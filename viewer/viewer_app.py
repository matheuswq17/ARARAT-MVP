from pathlib import Path
import sys
import os
import argparse
import json
import csv
import traceback
from datetime import datetime
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, Ellipse, Rectangle
import SimpleITK as sitk

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from shared import dicom_io
    from .exporters import roi_export
    from .exporters import mask_export
    from . import gt_labels
    from .inference_bridge import predict_for_export_folder
except (ImportError, ValueError) as e:
    try:
        from shared import dicom_io
        from viewer.exporters import roi_export
        from viewer.exporters import mask_export
        from viewer import gt_labels
        from viewer.inference_bridge import predict_for_export_folder
    except ImportError as e2:
        print(f"Erro fatal ao importar dependencias: {e2}")
        sys.exit(1)

class ViewerApp:
    def _load_config(self):
        """Carrega configurações persistentes (ex: data_root) de arquivo local."""
        config_path = os.path.join(current_dir, "config_local.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_config(self):
        """Salva configurações persistentes em arquivo local."""
        config_path = os.path.join(current_dir, "config_local.json")
        try:
            config = {
                "data_root": self.input_root,
                "samples_root": getattr(self, 'samples_root', None)
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except:
            pass

    def __init__(self, dicom_dir=None, data_root=None, series_hint="t2tsetra"):
        # Carregar config se argumentos forem vazios
        config = self._load_config()
        if not data_root and not dicom_dir and "data_root" in config:
            data_root = config["data_root"]
            print(f"[INFO] Carregando ultimo data_root da config: {data_root}")
        
        self.samples_root = config.get("samples_root")
        
        self.input_root = os.path.abspath(data_root) if data_root else (os.path.abspath(dicom_dir) if dicom_dir else None)
        self.dicom_root = self.input_root
        self.series_hint = series_hint
        
        # Salvar se definimos um novo root (e ele existe)
        if self.input_root and os.path.exists(self.input_root):
            # Se samples_root não veio da config, tenta inferir
            if not self.samples_root:
                self.discover_workspace()
            self._save_config()
        
        # Cache de Séries (LRU Simples)
        self._series_cache = {} # {(case_name, series_idx): (sitk_img, np_vol, meta)}
        self._cache_order = []
        self._max_cache_size = 6

        # Dados do Workspace (Cases)
        self.cases_list = []
        self.current_case_idx = -1
        self.is_samples_mode = False
        self.patient_select_mode = False # Novo modo de navegação
        
        # Dados do Case Atual
        self.series_list = []
        self.t2_quick = {'axial': None, 'coronal': None, 'sagittal': None} # indices
        self.current_series_idx = 0
        self.series_page = 0
        self.series_per_page = 20
        
        # Dados da Serie Atual
        self.sitk_img = None
        self.np_vol = None
        self.meta = None
        self.sitk_img_disp = None
        self.np_vol_disp = None
        self.meta_disp = None
        self.roi_status = {} # {lesion_id: "OK"/"PARTIAL"/"OUT"}
        self.rois_by_patient = {} # {patient_id: [rois]}
        
        self.current_slice = 0
        self.max_slice = 0
        self.center_voxel = [0, 0, 0]
        self.center_mm = None
        self.active_view = "axial"
        self.main_view = "axial"
        self.crop_mode = "CROP"
        self.crop_mm = {
            "axial": (180.0, 180.0),
            "coronal": (220.0, 140.0),
            "sagittal": (220.0, 140.0),
        }
        self.view_zoom = {"axial": 1.0, "coronal": 1.0, "sagittal": 1.0}
        self.view_pan = {"axial": (0.0, 0.0), "coronal": (0.0, 0.0), "sagittal": (0.0, 0.0)}
        self._pan_drag = {"active": False, "plane": None, "start_xy": None, "start_pan": None}
        self.win = {"axial": None, "coronal": None, "sagittal": None}
        self.level = {"axial": None, "coronal": None, "sagittal": None}
        self._wl_drag = {"active": False, "plane": None, "start_xy": None, "base_win": None, "base_level": None}
        self.view_state = {}
        for plane in ["axial", "coronal", "sagittal"]:
            self.view_state[plane] = {
                "mode": "FULL",
                "xlim": None,
                "ylim": None,
                "zoom": 1.0,
                "pan": (0.0, 0.0),
            }
        self.dev_layout_debug = False
        self.interp_mode = "bilinear"
        self._last_view_limits = {p: None for p in ["axial", "coronal", "sagittal"]}
        self.slot_main = [0.27, 0.55, 0.50, 0.40]
        self.slot_bl = [0.27, 0.05, 0.24, 0.40]
        self.slot_br = [0.53, 0.05, 0.24, 0.40]
        
        # ROI Candidata (Preview)
        self.candidate_center = None # (i, j, k) - None se seguindo mouse
        self.mouse_pos = (0, 0) # (i, j) atual do mouse
        self.radius_mm = 5.0
        self.is_locked = False # Se o clique travou o centro
        
        # Modos e Seleção
        self.mode = "NORMAL" # "NORMAL", "SERIES_SELECT", "CASE_SELECT"
        self.series_input_str = ""
        self.case_input_str = ""
        
        # Performance (Blitting)
        self.background = None
        self.preview_artists = [] # Mantido por compatibilidade temporaria se necessário
        self._persistent_artists = {
            'line': None,
            'ellipse': None,
            'text': None
        }
        
        self.rois = []
        self.lesion_counter = 1
        self.last_message = "Pronto"
        self.last_key = None
        self.gt_patient_id = None
        self.gt_label_source = None
        self.show_help = False
        self.last_preds = []
        self.roi_pred_map = {}
        self.show_predictions_panel = False
        self.toast_message = None
        self.toast_until = 0.0
        self.toast_artist = None
        self.show_gt = False
        self.gt_lesions = []
        self.gt_threshold_mm = 10.0
        
        self.fig = None
        self.ax_axial = None
        self.ax_sag = None
        self.ax_cor = None
        self.ax_info = None
        self.ax_sidebar = None
        self.ax = None
        
        # Export Info
        self.last_export_dir = None
        
        # Logo Branding & Icon
        self.logo_img = None
        assets_dir = os.path.join(current_dir, "assets")
        logo_path = os.path.join(assets_dir, "ararat_logo.png")
        self.ico_path = os.path.join(assets_dir, "ararat_logo.ico")
        
        if os.path.exists(logo_path):
            try:
                self.logo_img = mpimg.imread(logo_path)
                # Gerar .ico se não existir
                if not os.path.exists(self.ico_path):
                    img = Image.open(logo_path)
                    img.save(self.ico_path, format='ICO', sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)])
                    print(f"[INFO] Icone gerado em {self.ico_path}")
            except Exception as e:
                print(f"[WARNING] Erro ao processar logo/icone: {e}")
        else:
            print(f"[WARNING] Logo nao encontrada em {logo_path}")

        # Garantir pastas de export
        self.export_dir = os.path.join(project_root, "exports")
        self.roi_img_dir = os.path.join(self.export_dir, "roi_images")
        os.makedirs(self.roi_img_dir, exist_ok=True)
        
        # Inicialização
        self.discover_workspace()
        if self.is_samples_mode:
            self.load_case(0)
        else:
            self.discover_series()
            self.load_current_series()

    def discover_workspace(self):
        """Detecta se input_root é uma raiz de casos ou um caso específico."""
        if not self.input_root or not os.path.exists(self.input_root):
            self.cases_list = []
            self.last_message = "Nenhum data_root definido. Pressione O para selecionar."
            return

        print(f"Analisando workspace: {self.input_root}")
        
        # 1. Detectar se é modo SAMPLES ou SINGLE CASE com UX à prova de erro
        samples_dir = os.path.join(self.input_root, "SAMPLES")
        
        # Caso A: Selecionou PROSTATEx root (contém subpasta "SAMPLES")
        if os.path.exists(samples_dir) and os.path.isdir(samples_dir):
            print(f"[INFO] Root PROSTATEx detectado: {self.input_root}")
            self.samples_root = samples_dir
            self.is_samples_mode = True
        else:
            # Caso B: Selecionou a própria pasta SAMPLES (contém subpastas case*/patient*)
            # Verificamos se há subpastas que contêm DICOMs
            try:
                subdirs = sorted([d for d in os.listdir(self.input_root) 
                                 if os.path.isdir(os.path.join(self.input_root, d))])
            except:
                subdirs = []
            
            has_valid_subdirs = False
            for d in subdirs:
                if dicom_io.list_case_series(os.path.join(self.input_root, d)):
                    has_valid_subdirs = True
                    break
            
            if has_valid_subdirs:
                print(f"[INFO] Pasta SAMPLES detectada: {self.input_root}")
                self.samples_root = self.input_root
                self.input_root = os.path.dirname(self.samples_root) # Ajustar data_root para o pai
                self.is_samples_mode = True
            else:
                # Caso C: Tentar ver se é um caso único direto
                if dicom_io.list_case_series(self.input_root):
                    print("[INFO] Modo SINGLE CASE detectado.")
                    self.is_samples_mode = False
                    self.cases_list = [os.path.basename(self.input_root)]
                    self.current_case_idx = 0
                    return
                else:
                    # Caso D: Pasta Inválida
                    print(f"[WARNING] Pasta invalida: {self.input_root}")
                    self.last_message = "Pasta invalida. Selecione PROSTATEx (com SAMPLES) ou a propria SAMPLES."
                    self.cases_list = []
                    self.is_samples_mode = False
                    return

        # Se chegamos aqui, estamos em SAMPLES mode
        if self.is_samples_mode:
            self.discover_patients()

    def discover_patients(self):
        """Descobre pacientes na samples_root de forma estável."""
        if not self.samples_root or not os.path.exists(self.samples_root):
            return
            
        try:
            subdirs = sorted([d for d in os.listdir(self.samples_root) 
                             if os.path.isdir(os.path.join(self.samples_root, d))])
        except:
            subdirs = []
            
        valid_cases = []
        for d in subdirs:
            if dicom_io.list_case_series(os.path.join(self.samples_root, d)):
                valid_cases.append(d)
        
        if valid_cases:
            print(f"[INFO] {len(valid_cases)} casos encontrados em {self.samples_root}")
            self.cases_list = valid_cases
            # Se já tivermos um caso carregado que está na lista, manter o índice
            # Caso contrário, resetar para o primeiro
            if self.current_case_idx < 0 or self.current_case_idx >= len(self.cases_list):
                self.current_case_idx = 0
            
            # Garantir que dicom_root reflete o caso atual
            self.dicom_root = os.path.join(self.samples_root, self.cases_list[self.current_case_idx])
        else:
            self.last_message = "Nenhum caso com DICOM encontrado na pasta selecionada."
            self.cases_list = []
            self.is_samples_mode = False

    def next_patient(self, delta):
        """Navega para o paciente anterior/próximo."""
        if not self.cases_list:
            return
        
        new_idx = (self.current_case_idx + delta) % len(self.cases_list)
        self.load_case(new_idx)
        self.last_message = f"Paciente: {self.cases_list[new_idx]}"
        if self.fig: self.update_plot()

    def _get_autosave_path(self, case_name):
        """Retorna o caminho do arquivo rois_latest.json para um paciente."""
        if not self.input_root:
            return None
        return os.path.join(self.export_dir, case_name, "rois_latest.json")

    def _autosave_rois(self):
        """Salva as ROIs do paciente atual no arquivo rois_latest.json."""
        if not self.cases_list or self.current_case_idx < 0:
            return
            
        case_name = self.cases_list[self.current_case_idx]
        output_path = self._get_autosave_path(case_name)
        if not output_path:
            return
            
        try:
            roi_export.save_roi_json(output_path, case_name, self.rois, self.input_root)
            # Nota: HUD message opcional para não poluir muito, 
            # mas o requisito pede "ROI confirmada (autosave OK)"
        except Exception as e:
            print(f"[ERROR] Falha no autosave: {e}")

    def _autoload_rois(self, case_name):
        """Carrega as ROIs do paciente atual se rois_latest.json existir."""
        path = self._get_autosave_path(case_name)
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    
                loaded_rois = []
                # Converter formato do JSON exportado de volta para o formato interno
                for r in data.get("rois", []):
                    # Tentar extrair k do slice_index_k se existir, senão usa voxel[2]
                    # Note: roi_export simplificado pode não ter todos os campos
                    center_mm = r.get("center_xyz_mm")
                    if not center_mm: continue
                    
                    # Precisamos dos center_voxel. Como não temos a geometria aqui facilmente sem a série,
                    # o ideal é que o rois_latest.json tenha o center_voxel original.
                    # Mas o roi_export.py atual só salva center_xyz_mm.
                    # VAMOS MELHORAR O roi_export ou salvar o center_voxel aqui.
                    
                    roi = {
                        "id": r.get("id", f"L{len(loaded_rois)+1}"),
                        "center_mm": center_mm,
                        "radius_mm": r.get("radius_mm", 5.0),
                        "series_uid": r.get("series_instance_uid", "UNKNOWN"),
                        # Placeholder para center_voxel, será recalculado no load_current_series
                        "center_voxel": r.get("center_ijk", [0, 0, 0]) 
                    }
                    loaded_rois.append(roi)
                
                if loaded_rois:
                    self.rois_by_patient[case_name] = loaded_rois
                    self.last_message = "ROIs carregadas de rois_latest.json"
                    # Atualizar contador de lesões
                    try:
                        last_num = int(loaded_rois[-1]['id'][1:])
                        self.lesion_counter = last_num + 1
                    except:
                        self.lesion_counter = len(loaded_rois) + 1
                    return True
            except Exception as e:
                print(f"[ERROR] Falha ao carregar rois_latest: {e}")
        return False

    def load_case(self, case_idx):
        """Troca o caso ativo (hot-swap)."""
        if not (0 <= case_idx < len(self.cases_list)):
            return False
            
        new_case_name = self.cases_list[case_idx]
        print(f"\n[HOT-SWAP] Trocando para caso: {new_case_name}")
        
        # Alerta se houver ROIs não exportadas
        if self.rois:
            self.last_message = "Aviso: ROIs nao exportadas. Pressione E para exportar."
        
        # Atualizar dicom_root
        if self.is_samples_mode:
            self.dicom_root = os.path.join(self.samples_root, new_case_name)
        
        self.current_case_idx = case_idx
        
        # Reset de Estado Robusto
        self.series_list = []
        self.current_series_idx = 0
        self.series_page = 0
        self.current_slice = 0
        
        # Gerenciar ROIs por paciente
        # Se já estiver no dict rois_by_patient, carregar de lá
        if new_case_name in self.rois_by_patient:
            self.rois = self.rois_by_patient[new_case_name]
            # Atualizar lesion_counter
            if self.rois:
                try:
                    last_num = int(self.rois[-1]['id'][1:])
                    self.lesion_counter = last_num + 1
                except:
                    self.lesion_counter = len(self.rois) + 1
            else:
                self.lesion_counter = 1
        else:
            # Tentar carregar do disco (autosave)
            if self._autoload_rois(new_case_name):
                self.rois = self.rois_by_patient[new_case_name]
            else:
                self.rois = []
                self.lesion_counter = 1
        
        self.roi_status = {}
        self.candidate_center = None
        self.is_locked = False
        self.mode = "NORMAL"
        self.case_input_str = ""
        self.series_input_str = ""
        self.last_message = f"Caso {new_case_name} carregado."
        self.show_gt = False
        self.gt_lesions = []
        
        # Recarregar séries do novo caso
        self.discover_series()
        self.load_current_series()
        
        # Se UI já existir, atualizar
        if self.fig:
            self.update_plot()
        return True

    def _load_gt_for_case(self):
        if not self.cases_list or self.current_case_idx < 0:
            self.gt_lesions = []
            return
        case_name = self.cases_list[self.current_case_idx]
        try:
            patient_id = gt_labels.resolve_patient_id(case_name, self.input_root)
        except Exception:
            patient_id = None
        self.gt_patient_id = patient_id
        if not patient_id:
            self.gt_lesions = []
            self.last_message = f"GT indisponivel: nao foi possivel mapear {case_name}"
            return
        try:
            self.gt_lesions = gt_labels.get_gt_for_case(patient_id, self.input_root)
        except Exception:
            self.gt_lesions = []
        self.gt_label_source = self.gt_lesions[0].get("source") if self.gt_lesions else None

    def discover_series(self):
        """Busca series no dicom_root atual."""
        if not self.dicom_root or not os.path.exists(self.dicom_root):
            print(f"[WARNING] dicom_root invalido para busca de series: {self.dicom_root}")
            self.series_list = []
            return

        print(f"Buscando series em: {self.dicom_root}...")
        self.series_list = dicom_io.list_case_series(self.dicom_root)
        if not self.series_list:
            print(f"\n[AVISO] Nenhuma serie DICOM valida encontrada em {self.dicom_root}")
            return
        
        # Reset T2 quick
        self.t2_quick = {'axial': None, 'coronal': None, 'sagittal': None}
        
        print("\n[DEBUG] Analisando candidatos T2 QUICK:")
        
        # Detectar Tri-planar T2
        # Critério: t2 + orientação + maior num_slices (se houver empate)
        for orient in ['axial', 'coronal', 'sagittal']:
            candidates = []
            for idx, s in enumerate(self.series_list):
                # Imprimir info de cada série candidata T2
                if s['is_t2']:
                    print(f"  - Serie: {s['series_name']} | Orient: {s['orientation']} | Slices: {s['num_slices']}")
                
                if s['is_t2'] and s['orientation'] == orient:
                    candidates.append((idx, s))
            
            if candidates:
                # Escolher a que tem mais slices (mais provável ser a principal)
                best_idx = max(candidates, key=lambda x: x[1]['num_slices'])[0]
                self.t2_quick[orient] = best_idx
                print(f"  => [VITORIA] T2 {orient.upper()} detectada: {self.series_list[best_idx]['series_name']}")
            else:
                print(f"  => [AVISO] T2 {orient.upper()} nao encontrada")

        # Escolher serie default (Axial T2)
        if self.t2_quick['axial'] is not None:
            self.current_series_idx = self.t2_quick['axial']
            return
            
        # Fallback: Serie com mais slices
        self.current_series_idx = self.series_list.index(max(self.series_list, key=lambda x: x['num_slices']))

    def validate_rois_for_current_series(self):
        """Valida se cada ROI confirmada esta dentro do volume atual."""
        if not self.meta or not self.rois:
            self.roi_status = {}
            return {}

        new_status = {}
        out_list = []
        
        for roi in self.rois:
            lid = roi['id']
            rmm = roi['radius_mm']
            
            # Converter centro do mundo para voxel na série atual
            vi, vj, vk = dicom_io.mm_to_voxel(roi['center_mm'][0], 
                                            roi['center_mm'][1], 
                                            roi['center_mm'][2], 
                                            self.meta)
            
            # Limites do volume
            sz_k, sz_j, sz_i = self.np_vol.shape
            
            # Check centro
            is_center_in = (0 <= vi < sz_i and 0 <= vj < sz_j and 0 <= vk < sz_k)
            
            if not is_center_in:
                status = "OUT"
                out_list.append(lid)
            else:
                # Check parcial (bounding box simplificada em voxel)
                # Converter raio mm para voxels (aproximado por eixo)
                ri = rmm / self.meta['spacing'][0]
                rj = rmm / self.meta['spacing'][1]
                rk = rmm / self.meta['spacing'][2]
                
                is_partial = (
                    vi - ri < 0 or vi + ri >= sz_i or
                    vj - rj < 0 or vj + rj >= sz_j or
                    vk - rk < 0 or vk + rk >= sz_k
                )
                status = "PARTIAL" if is_partial else "OK"
            
            new_status[lid] = status
            
        self.roi_status = new_status
        
        # Atualizar mensagem se houver OUT
        if out_list:
            self.last_message = f"Atencao: ROIs fora do volume: {', '.join(out_list)}"
        
        return new_status

    def load_current_series(self, center_mm=None):
        """Carrega os dados da serie selecionada no momento."""
        if not self.series_list or self.current_series_idx >= len(self.series_list):
            self.sitk_img, self.np_vol, self.meta = None, None, None
            self.sitk_img_disp, self.np_vol_disp, self.meta_disp = None, None, None
            self.max_slice = 0
            self.current_slice = 0
            self.center_voxel = [0, 0, 0]
            self.center_mm = None
            return
        self._load_gt_for_case()
        case_name = self.cases_list[self.current_case_idx] if self.cases_list else "unknown"
        s_idx = self.current_series_idx
        cache_key = (case_name, s_idx)

        # 1. Tentar Cache
        if cache_key in self._series_cache:
            print(f"[DEBUG] Cache HIT: {cache_key}")
            self.sitk_img, self.np_vol, self.meta = self._series_cache[cache_key]
            # Atualizar ordem do LRU
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            self.last_message = f"Pronto (cache)"
        else:
            s = self.series_list[s_idx]
            self.last_message = f"Carregando {s['series_name']}..."
            if self.fig: self.update_plot()
            
            print(f"\n[INFO] Carregando serie: {s['series_name']} ({s['orientation']})")
            try:
                self.sitk_img, self.np_vol, self.meta = dicom_io.load_dicom_series_by_path(
                    s['series_dir'], s['series_uid']
                )
                
                # Adicionar ao Cache
                self._series_cache[cache_key] = (self.sitk_img, self.np_vol, self.meta)
                self._cache_order.append(cache_key)
                if len(self._cache_order) > self._max_cache_size:
                    oldest = self._cache_order.pop(0)
                    if oldest in self._series_cache:
                        del self._series_cache[oldest]
                        print(f"[DEBUG] Cache Evict: {oldest}")
                self.last_message = "Pronto"
                    
            except Exception as e:
                self.last_message = f"ERRO ao carregar serie (ver terminal)"
                print(f"\n[ERRO] Falha ao carregar serie {s['series_name']}: {e}")
                traceback.print_exc()
                if self.fig: self.update_plot()
                return

        self.max_slice = self.np_vol.shape[0] - 1
        self._prepare_display_volume()
        
        # Validar ROIs para esta série
        self.validate_rois_for_current_series()
        
        # Atualizar center_voxel das ROIs para a geometria da série atual
        # Isso garante que a renderização (desenho dos círculos) seja precisa
        for roi in self.rois:
            v = dicom_io.mm_to_voxel(roi['center_mm'][0], roi['center_mm'][1], roi['center_mm'][2], self.meta)
            roi['center_voxel'] = [int(round(v[0])), int(round(v[1])), int(round(v[2]))]
        
        # Posicionamento do centro
        sz_k, sz_j, sz_i = self.np_vol.shape
        if center_mm:
            vi, vj, vk = dicom_io.mm_to_voxel(center_mm[0], center_mm[1], center_mm[2], self.meta)
            i = int(round(max(0, min(vi, sz_i - 1))))
            j = int(round(max(0, min(vj, sz_j - 1))))
            k = int(round(max(0, min(vk, sz_k - 1))))
            self._set_center_voxel(i, j, k)
        elif self.center_mm is not None:
            vi, vj, vk = dicom_io.mm_to_voxel(self.center_mm[0], self.center_mm[1], self.center_mm[2], self.meta)
            i = int(round(max(0, min(vi, sz_i - 1))))
            j = int(round(max(0, min(vj, sz_j - 1))))
            k = int(round(max(0, min(vk, sz_k - 1))))
            self._set_center_voxel(i, j, k)
        else:
            i = sz_i // 2
            j = sz_j // 2
            k = sz_k // 2
            self._set_center_voxel(i, j, k)
        for plane in ["axial", "coronal", "sagittal"]:
            self._reset_wl_for_plane(plane, use_full_volume=True)
            state = self.view_state.get(plane)
            if state is not None:
                state["mode"] = "FULL"
                state["xlim"] = None
                state["ylim"] = None
                state["zoom"] = 1.0
                state["pan"] = (0.0, 0.0)
        self.crop_mode = "CROP"
        
        self.candidate_center = None
        self.is_locked = False
        self.last_message = "Pronto"
        if self.fig: self.update_plot()

    def _set_center_voxel(self, i, j, k):
        if self.np_vol is None or self.meta is None:
            self.center_voxel = [int(i), int(j), int(k)]
            self.current_slice = int(k)
            return
        sz_k, sz_j, sz_i = self.np_vol.shape
        i = int(max(0, min(i, sz_i - 1)))
        j = int(max(0, min(j, sz_j - 1)))
        k = int(max(0, min(k, sz_k - 1)))
        self.center_voxel = [i, j, k]
        self.current_slice = k
        x_mm, y_mm, z_mm = dicom_io.voxel_to_mm(i, j, k, self.meta)
        self.center_mm = [float(x_mm), float(y_mm), float(z_mm)]

    def _move_center_slice(self, plane, delta):
        if self.np_vol is None:
            return
        sz_k, sz_j, sz_i = self.np_vol.shape
        i, j, k = self.center_voxel
        if plane == "axial":
            k = int(max(0, min(k + delta, sz_k - 1)))
        elif plane == "sagittal":
            i = int(max(0, min(i + delta, sz_i - 1)))
        elif plane == "coronal":
            j = int(max(0, min(j + delta, sz_j - 1)))
        self._set_center_voxel(i, j, k)

    def run(self):
        # Desativar hotkeys padrão do matplotlib que podem causar conflitos
        plt.rcParams['keymap.save'] = ''
        plt.rcParams['keymap.fullscreen'] = ''
        plt.rcParams['keymap.home'] = ''
        plt.rcParams['keymap.back'] = ''
        plt.rcParams['keymap.forward'] = ''
        plt.rcParams['keymap.pan'] = ''
        plt.rcParams['keymap.zoom'] = ''
        plt.rcParams['keymap.quit'] = ''

        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor("#222222")
        self.ax_sidebar = self.fig.add_axes([0.03, 0.05, 0.22, 0.90])
        self.ax_info = self.fig.add_axes([0.80, 0.05, 0.17, 0.90])
        self.ax_axial = self.fig.add_axes(self.slot_main)
        self.ax_cor = self.fig.add_axes(self.slot_bl)
        self.ax_sag = self.fig.add_axes(self.slot_br)
        self.ax = self.ax_axial

        if self.ax_sidebar:
            self.ax_sidebar.set_facecolor("#303030")
            self.ax_sidebar.axis('off')
        if self.ax_info:
            self.ax_info.set_facecolor("#202020")
            self.ax_info.axis('off')
        for ax in [self.ax_axial, self.ax_cor, self.ax_sag]:
            if ax:
                ax.set_facecolor("black")

        # Desativar toolbar para evitar pan/zoom acidental
        self.fig.canvas.toolbar.pack_forget()

        self.fig.canvas.manager.set_window_title("ARARAT Viewer")
        
        # Definir icone da janela (se o ico existir)
        if os.path.exists(self.ico_path):
            try:
                # No Windows, tentamos usar o método do manager para definir o ícone
                manager = self.fig.canvas.manager
                if hasattr(manager, 'window'):
                    # Dependendo do backend (TkAgg é o padrão no Windows)
                    try:
                        manager.window.iconbitmap(self.ico_path)
                    except:
                        manager.set_window_icon(self.ico_path)
                else:
                    manager.set_window_icon(self.ico_path)
            except Exception as e:
                print(f"[WARNING] Nao foi possivel definir o icone da janela: {e}")

        self._apply_slots()
        self.update_plot()
        
        # conectar eventos
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('draw_event', self.on_draw)
        self.fig.canvas.mpl_connect('resize_event', self.on_resize)
        
        plt.show()

    def on_draw(self, event):
        return

    def on_resize(self, event):
        self._apply_slots()

    def on_button_release(self, event):
        if self._pan_drag["active"]:
            self._pan_drag["active"] = False
            self._pan_drag["plane"] = None
            self._pan_drag["start_xy"] = None
            self._pan_drag["start_pan"] = None
        if self._wl_drag["active"]:
            self._wl_drag["active"] = False
            self._wl_drag["plane"] = None
            self._wl_drag["start_xy"] = None
            self._wl_drag["base_win"] = None
            self._wl_drag["base_level"] = None

    def _draw_preview_fast(self):
        """Desenha o preview da ROI usando blitting e artists persistentes."""
        if self.background is None or self.ax is None or self.meta is None:
            return

        # Restaurar background
        self.fig.canvas.restore_region(self.background)

        # Determinar posição e parâmetros
        if self.is_locked and self.candidate_center:
            ci, cj, ck = self.candidate_center
            color = 'red'
            linestyle = '--'
            alpha = 1.0
        else:
            ci, cj = self.mouse_pos
            ck = self.current_slice
            color = 'red'
            linestyle = ':'
            alpha = 0.5

        # Calcular interseção da esfera
        p_center = dicom_io.voxel_to_mm(ci, cj, ck, self.meta)
        p_here = dicom_io.voxel_to_mm(ci, cj, self.current_slice, self.meta)
        dz_mm = abs(p_here[2] - p_center[2])
        
        show_roi = dz_mm < self.radius_mm

        # 1. Gerenciar Crosshair (Line2D)
        if self._persistent_artists['line'] is None:
            line, = self.ax.plot([ci], [cj], marker='+', color=color, markersize=10, 
                               animated=True, visible=False)
            self._persistent_artists['line'] = line
        
        line = self._persistent_artists['line']
        if show_roi:
            line.set_data([ci], [cj])
            line.set_color(color)
            line.set_alpha(alpha)
            line.set_visible(True)
        else:
            line.set_visible(False)

        # 2. Gerenciar Elipse (Patch)
        if self._persistent_artists['ellipse'] is None:
            ellipse = Ellipse((ci, cj), width=1, height=1, fill=False, color=color, 
                             linewidth=2, animated=True, visible=False)
            self.ax.add_patch(ellipse)
            self._persistent_artists['ellipse'] = ellipse
        
        ellipse = self._persistent_artists['ellipse']
        if show_roi:
            r_slice_mm = (self.radius_mm**2 - dz_mm**2)**0.5
            r_px_x = r_slice_mm / self.meta['spacing'][0]
            r_px_y = r_slice_mm / self.meta['spacing'][1]
            ellipse.set_center((ci, cj))
            ellipse.set_width(r_px_x * 2)
            ellipse.set_height(r_px_y * 2)
            ellipse.set_color(color)
            ellipse.set_linestyle(linestyle)
            ellipse.set_alpha(alpha)
            ellipse.set_visible(True)
        else:
            ellipse.set_visible(False)

        # Redesenhar artists
        if line.get_visible():
            self.ax.draw_artist(line)
        if ellipse.get_visible():
            self.ax.draw_artist(ellipse)

        self.fig.canvas.blit(self.ax.bbox)

    def _draw_roi_sphere(self, center_ijk, radius_mm, color, label=None, linestyle='-', alpha=1.0, roi_mm=None):
        if self.meta is None:
            return
            
        # Se roi_mm for fornecido, usamos ele para converter para voxel da série atual
        if roi_mm:
            ci, cj, ck = dicom_io.mm_to_voxel(roi_mm[0], roi_mm[1], roi_mm[2], self.meta)
        else:
            ci, cj, ck = center_ijk
        
        # Calcular dz_mm (distancia entre o slice do centro e o slice atual)
        p_center = dicom_io.voxel_to_mm(ci, cj, ck, self.meta)
        p_here = dicom_io.voxel_to_mm(ci, cj, self.current_slice, self.meta)
        dz_mm = abs(p_here[2] - p_center[2])
        
        if dz_mm < radius_mm:
            r_slice_mm = (radius_mm**2 - dz_mm**2)**0.5
            r_px_x = r_slice_mm / self.meta['spacing'][0]
            r_px_y = r_slice_mm / self.meta['spacing'][1]
            self.ax.plot(ci, cj, marker='+', color=color, markersize=10, alpha=alpha)
            ellipse = Ellipse((ci, cj), width=r_px_x*2, height=r_px_y*2, 
                             fill=False, color=color, linestyle=linestyle, alpha=alpha, linewidth=2)
            self.ax.add_patch(ellipse)
            if label:
                self.ax.text(ci+2, cj+2, label, color=color, fontweight='bold', fontsize=8)

    def _prepare_display_volume(self):
        self.sitk_img_disp = None
        self.np_vol_disp = None
        self.meta_disp = None
        if self.sitk_img is None or self.np_vol is None or self.meta is None:
            return
        try:
            sx, sy, sz = self.meta["spacing"]
        except Exception:
            sx, sy, sz = self.sitk_img.GetSpacing()
        s_min = min(float(sx), float(sy), float(sz))
        target_spacing = (s_min, s_min, s_min)
        size_x, size_y, size_z = self.sitk_img.GetSize()
        new_size = [
            int(round(size_x * sx / target_spacing[0])),
            int(round(size_y * sy / target_spacing[1])),
            int(round(size_z * sz / target_spacing[2])),
        ]
        resampled = sitk.Resample(
            self.sitk_img,
            new_size,
            sitk.Transform(),
            sitk.sitkBSpline,
            self.sitk_img.GetOrigin(),
            target_spacing,
            self.sitk_img.GetDirection(),
            0.0,
            self.sitk_img.GetPixelID(),
        )
        self.sitk_img_disp = resampled
        self.np_vol_disp = sitk.GetArrayFromImage(resampled)
        self.meta_disp = {
            "origin": resampled.GetOrigin(),
            "spacing": resampled.GetSpacing(),
            "direction": resampled.GetDirection(),
            "size": resampled.GetSize(),
        }

    def _get_wl_for_plane(self, plane):
        w = self.win.get(plane)
        l = self.level.get(plane)
        return w, l

    def _reset_wl_for_plane(self, plane, use_full_volume=False):
        if self.np_vol is None:
            self.win[plane] = None
            self.level[plane] = None
            return
        vol = self.np_vol
        if not use_full_volume:
            sz_k, sz_j, sz_i = vol.shape
            i, j, k = self.center_voxel
            i = int(max(0, min(i, sz_i - 1)))
            j = int(max(0, min(j, sz_j - 1)))
            k = int(max(0, min(k, sz_k - 1)))
            if plane == "axial":
                slice_img = vol[k, :, :]
            elif plane == "sagittal":
                slice_img = vol[:, :, i]
            else:
                slice_img = vol[:, j, :]
            data = slice_img.astype(float).ravel()
        else:
            data = vol.astype(float).ravel()
        if data.size == 0:
            self.win[plane] = None
            self.level[plane] = None
            return
        p1, p99 = np.percentile(data, [2, 98])
        w = float(max(p99 - p1, 1e-3))
        l = float((p99 + p1) * 0.5)
        self.win[plane] = w
        self.level[plane] = l

    def _get_help_text(self):
        """Gera o texto do help com alinhamento perfeito."""
        def fmt_line(cmd, desc, col=18):
            return f"{cmd.ljust(col)} : {desc}"

        lines = [
            "=== COMANDOS DO SISTEMA ===",
            "",
            "NAVEGACAO",
            fmt_line("Scroll / Drag", "zoom/pan (painel ativo)"),
            fmt_line("Arrows", "trocar slice"),
            fmt_line("[ / ]", "paginar series"),
            fmt_line("1..9", "atalho serie"),
            fmt_line("Ctrl + G", "ir para serie N"),
            fmt_line("Ctrl + Up/Dn", "navegar paciente"),
            fmt_line("C", "ir para paciente N"),
            fmt_line("A / K / S", "alternar painel AX/COR/SAG"),
            fmt_line("O", "abrir data_root"),
            "",
            "ROI (LESAO)",
            fmt_line("Clique esq.", "travar centro"),
            fmt_line("Enter", "confirmar ROI"),
            fmt_line("X", "limpar selecao"),
            fmt_line("+ / -", "ajustar raio"),
            fmt_line("Del", "deletar ultima"),
            "",
            "GERAL",
            fmt_line("E", "EXPORTAR & INFERIR"),
            fmt_line("F", "ABRIR PASTA EXPORT"),
            fmt_line("J / Ctrl+S", "Salvar JSON (apenas)"),
            fmt_line("V", "VALIDAR ROIs (log)"),
            fmt_line("G", "toggle GT (gabarito)"),
            fmt_line("Shift+G", "ir para lesao GT"),
            fmt_line("P", "toggle Painel Predicoes"),
            fmt_line("H", "mostrar/ocultar help"),
            fmt_line("D", "toggle modo DEV"),
            fmt_line("R / Shift+R", "reset view / reset all"),
            fmt_line("Z", "toggle Zoom/Crop mode"),
            fmt_line("I", "toggle interpolacao"),
            fmt_line("Botao dir + drag", "ajustar brilho/contraste"),
            fmt_line("Q", "sair")
        ]
        return "\n".join(lines)

    def _render_mpr_view(self, ax, plane):
        if ax is None or self.meta is None or self.np_vol is None:
            return
        vol = self.np_vol_disp if self.np_vol_disp is not None else self.np_vol
        meta_d = self.meta_disp if self.meta_disp is not None else self.meta
        sz_k, sz_j, sz_i = vol.shape
        try:
            sx, sy, sz = meta_d["spacing"]
        except Exception:
            sx, sy, sz = (1.0, 1.0, 1.0)
        if self.center_mm is not None:
            cx_mm, cy_mm, cz_mm = self.center_mm
            vi, vj, vk = dicom_io.mm_to_voxel(cx_mm, cy_mm, cz_mm, meta_d)
        else:
            i0, j0, k0 = self.center_voxel
            vx_mm, vy_mm, vz_mm = dicom_io.voxel_to_mm(i0, j0, k0, meta_d)
            vi, vj, vk = dicom_io.mm_to_voxel(vx_mm, vy_mm, vz_mm, meta_d)
        vi_i = int(max(0, min(round(vi), sz_i - 1)))
        vj_j = int(max(0, min(round(vj), sz_j - 1)))
        vk_k = int(max(0, min(round(vk), sz_k - 1)))
        if plane == "axial":
            slice_index = vk_k
            slice_img = vol[slice_index, :, :]
            max_x = sz_i * sx
            max_y = sz_j * sy
            center_x = cx_mm
            center_y = cy_mm
        elif plane == "sagittal":
            slice_index = vi_i
            slice_img = vol[:, :, slice_index]
            max_x = sz_j * sy
            max_y = sz_k * sz
            center_x = cy_mm
            center_y = cz_mm
        elif plane == "coronal":
            slice_index = vj_j
            slice_img = vol[:, slice_index, :]
            max_x = sz_i * sx
            max_y = sz_k * sz
            center_x = cx_mm
            center_y = cz_mm
        else:
            return
        state = self.view_state.get(plane)
        win, level = self._get_wl_for_plane(plane)
        if win is not None and level is not None:
            vmin = level - win / 2.0
            vmax = level + win / 2.0
        else:
            vmin, vmax = None, None
        extent = [0.0, max_x, max_y, 0.0]
        ax.imshow(
            slice_img,
            cmap="gray",
            origin="upper",
            extent=extent,
            interpolation=self.interp_mode,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_autoscale_on(False)
        if state is None:
            mode = "FULL"
            xlim, ylim = (0.0, max_x), (max_y, 0.0)
        else:
            mode = state.get("mode", "FULL")
            xlim = state.get("xlim")
            ylim = state.get("ylim")
            if xlim is None or ylim is None:
                if mode == "CROP":
                    crop_w, crop_h = self.crop_mm.get(plane, (max_x, max_y))
                    half_w = crop_w * 0.5
                    half_h = crop_h * 0.5
                    x0 = max(0.0, center_x - half_w)
                    x1 = min(max_x, center_x + half_w)
                    y0 = max(0.0, center_y - half_h)
                    y1 = min(max_y, center_y + half_h)
                    xlim = (x0, x1)
                    ylim = (y1, y0)
                else:
                    xlim = (0.0, max_x)
                    ylim = (max_y, 0.0)
                state["xlim"] = xlim
                state["ylim"] = ylim
        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_anchor("C")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
        if self.dev_layout_debug and state is not None:
            bbox = ax.get_window_extent().bounds if self.fig and self.fig.canvas else (0.0, 0.0, 0.0, 0.0)
            xlim_now = ax.get_xlim()
            ylim_now = ax.get_ylim()
            dbg = (
                f"bbox={bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}\n"
                f"xlim={xlim_now[0]:.1f},{xlim_now[1]:.1f}\n"
                f"ylim={ylim_now[0]:.1f},{ylim_now[1]:.1f}\n"
                f"mode={state.get('mode','')}"
            )
            ax.text(
                0.02,
                0.02,
                dbg,
                transform=ax.transAxes,
                fontsize=6,
                color="cyan",
                verticalalignment="bottom",
                family="monospace",
            )
            last = self._last_view_limits.get(plane)
            cur = (round(xlim_now[0], 3), round(xlim_now[1], 3), round(ylim_now[0], 3), round(ylim_now[1], 3))
            if last != cur:
                print(f"[VIEW] {plane} xlim={xlim_now} ylim={ylim_now} mode={state.get('mode')}")
                self._last_view_limits[plane] = cur
        self._draw_crosshair(ax, plane)
        self._draw_rois_on_plane(ax, plane, slice_index)
        self._draw_gt_on_plane(ax, plane, slice_index)
        if self.fig and self.dev_layout_debug:
            w_px, h_px = self._get_axes_px(ax)
            if w_px and h_px:
                if ax is self.ax_axial:
                    slot_name = "MAIN"
                elif ax is self.ax_cor:
                    slot_name = "BL"
                elif ax is self.ax_sag:
                    slot_name = "BR"
                else:
                    slot_name = ""
                if slot_name:
                    txt = f"{slot_name} {w_px}x{h_px}"
                    ax.text(
                        0.02,
                        0.90,
                        txt,
                        transform=ax.transAxes,
                        fontsize=8,
                        color="#bbbbbb",
                        verticalalignment="top",
                        horizontalalignment="left",
                        family="monospace",
                        alpha=0.9,
                    )
        self._draw_orientation_labels(ax, plane)
        if ax is self.ax_axial:
            ax.set_position(self.slot_main)
        elif ax is self.ax_cor:
            ax.set_position(self.slot_bl)
        elif ax is self.ax_sag:
            ax.set_position(self.slot_br)

    def _get_layout_assignment(self):
        mv = self.main_view or "axial"
        if mv == "coronal":
            return {
                "main": "coronal",
                "bottom_left": "axial",
                "bottom_right": "sagittal",
            }
        if mv == "sagittal":
            return {
                "main": "sagittal",
                "bottom_left": "axial",
                "bottom_right": "coronal",
            }
        return {
            "main": "axial",
            "bottom_left": "coronal",
            "bottom_right": "sagittal",
        }

    def _plane_for_axes(self, ax):
        if ax is None:
            return None
        slots = self._get_layout_assignment()
        if ax is self.ax_axial:
            return slots["main"]
        if ax is self.ax_cor:
            return slots["bottom_left"]
        if ax is self.ax_sag:
            return slots["bottom_right"]
        return None

    def _apply_slots(self):
        if self.ax_axial:
            self.ax_axial.set_position(self.slot_main)
        if self.ax_cor:
            self.ax_cor.set_position(self.slot_bl)
        if self.ax_sag:
            self.ax_sag.set_position(self.slot_br)

    def _get_axes_px(self, ax):
        if not self.fig or ax is None:
            return 0, 0
        try:
            bbox = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            w_px = int(round(bbox.width * self.fig.dpi))
            h_px = int(round(bbox.height * self.fig.dpi))
            return w_px, h_px
        except Exception:
            return 0, 0
    
    def _draw_orientation_labels(self, ax, plane):
        meta = self.meta_disp if self.meta_disp is not None else self.meta
        if meta is None or ax is None:
            return
        try:
            direction = meta.get("direction")
        except Exception:
            direction = None
        use_fallback = False
        labels_lr = ("L", "R")
        labels_ap = ("A", "P")
        labels_si = ("S", "I")
        def axis_letter(vec):
            idx = int(np.argmax(np.abs(vec)))
            sign = np.sign(vec[idx])
            if idx == 0:
                return ("L" if sign > 0 else "R", "R" if sign > 0 else "L")
            elif idx == 1:
                return ("P" if sign > 0 else "A", "A" if sign > 0 else "P")
            else:
                return ("S" if sign > 0 else "I", "I" if sign > 0 else "S")
        if direction and isinstance(direction, (list, tuple)) and len(direction) == 9:
            d = np.array(direction, dtype=float).reshape(3, 3)
            if plane == "axial":
                u_vec = d[:, 0]
                v_vec = d[:, 1]
            elif plane == "sagittal":
                u_vec = d[:, 1]
                v_vec = d[:, 2]
            elif plane == "coronal":
                u_vec = d[:, 0]
                v_vec = d[:, 2]
            else:
                return
            left_right = axis_letter(u_vec)
            top_bottom = axis_letter(v_vec)
            left_label, right_label = left_right
            top_label, bottom_label = top_bottom[1], top_bottom[0]
        else:
            use_fallback = True
            if plane == "axial":
                left_label, right_label = labels_lr
                top_label, bottom_label = labels_ap
            elif plane == "sagittal":
                left_label, right_label = labels_ap
                top_label, bottom_label = labels_si
            elif plane == "coronal":
                left_label, right_label = labels_lr
                top_label, bottom_label = labels_si
            else:
                return
        ax.text(0.02, 0.50, left_label, transform=ax.transAxes, color="#dddddd", fontsize=10, alpha=0.7, ha="left", va="center", family="monospace")
        ax.text(0.98, 0.50, right_label, transform=ax.transAxes, color="#dddddd", fontsize=10, alpha=0.7, ha="right", va="center", family="monospace")
        ax.text(0.50, 0.98, top_label, transform=ax.transAxes, color="#dddddd", fontsize=10, alpha=0.7, ha="center", va="top", family="monospace")
        ax.text(0.50, 0.02, bottom_label, transform=ax.transAxes, color="#dddddd", fontsize=10, alpha=0.7, ha="center", va="bottom", family="monospace")
        if use_fallback and self.dev_layout_debug:
            ax.text(0.98, 0.02, "ORIENT FALLBACK", transform=ax.transAxes, color="cyan", fontsize=7, alpha=0.7, ha="right", va="bottom", family="monospace")

    def _debug_slot_sizes(self):
        if not self.fig:
            return
        slots = [("MAIN", self.ax_axial), ("BL", self.ax_cor), ("BR", self.ax_sag)]
        measures = []
        for name, ax in slots:
            if ax is None:
                measures.append(f"{name}=None")
                continue
            w_px, h_px = self._get_axes_px(ax)
            if w_px and h_px:
                measures.append(f"{name}={w_px}x{h_px}")
            else:
                measures.append(f"{name}=?x?")
        if measures:
            print("[SLOTS]", " | ".join(measures))

    def _style_panel(self, ax, plane):
        if ax is None:
            return
        name = "AXIAL" if plane == "axial" else ("SAGITTAL" if plane == "sagittal" else "CORONAL")
        is_active = (plane == self.active_view)
        title_color = "yellow" if is_active else "white"
        if plane == "axial":
            base_color = "#d84a4a"
        elif plane == "sagittal":
            base_color = "#f1c40f"
        else:
            base_color = "#4caf50"
        label = name
        if self.np_vol is not None:
            sz_k, sz_j, sz_i = self.np_vol.shape
            i, j, k = self.center_voxel
            if plane == "axial":
                pos = int(max(0, min(k, sz_k - 1))) + 1
                total = sz_k
                label = f"{name} k={pos}/{total}"
            elif plane == "coronal":
                pos = int(max(0, min(j, sz_j - 1))) + 1
                total = sz_j
                label = f"{name} j={pos}/{total}"
            else:
                pos = int(max(0, min(i, sz_i - 1))) + 1
                total = sz_i
                label = f"{name} i={pos}/{total}"
        else:
            label = f"{name} [NO DATA]"
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            fontsize=10,
            color=title_color,
            fontfamily="monospace",
            bbox=dict(facecolor=base_color, alpha=0.9, edgecolor="none", boxstyle="round,pad=0.3"),
            verticalalignment="top",
            horizontalalignment="left",
        )
        for side, sp in ax.spines.items():
            sp.set_linewidth(1.5 if is_active else 1.0)
            if side == "top":
                sp.set_edgecolor(base_color)
            else:
                sp.set_edgecolor("yellow" if is_active else "#444")

    def _draw_crosshair(self, ax, plane):
        if self.np_vol is None or self.meta is None:
            return
        i, j, k = self.center_voxel
        try:
            sx, sy, sz = self.meta["spacing"]
        except Exception:
            try:
                sx, sy, sz = self.sitk_img.GetSpacing()
            except Exception:
                sx, sy, sz = (1.0, 1.0, 1.0)
        if plane == "axial":
            x = i * sx
            y = j * sy
        elif plane == "sagittal":
            x = j * sy
            y = k * sz
        elif plane == "coronal":
            x = i * sx
            y = k * sz
        else:
            return
        ax.axvline(x, color="yellow", linestyle="--", linewidth=0.8)
        ax.axhline(y, color="yellow", linestyle="--", linewidth=0.8)

    def _get_roi_draw_params(self, roi):
        color = "lime"
        text_label = roi["id"]
        pred = self.roi_pred_map.get(roi["id"]) if self.roi_pred_map else None
        if pred:
            cat = pred.get("risk_category", "")
            perc = pred.get("risk_percent", pred.get("prob_pos", 0.0) * 100.0)
            if cat == "Baixo":
                color = "lime"
            elif cat == "Intermediário":
                color = "yellow"
            elif cat == "Alto":
                color = "red"
            else:
                color = "cyan"
            text_label = f"{roi['id']} {perc:.0f}% {cat}"
        return color, text_label

    def _draw_rois_on_plane(self, ax, plane, slice_index):
        if self.np_vol is None or self.meta is None:
            return
        if not self.rois:
            return
        sz_k, sz_j, sz_i = self.np_vol.shape
        try:
            sx, sy, sz = self.meta["spacing"]
        except Exception:
            try:
                sx, sy, sz = self.sitk_img.GetSpacing()
            except Exception:
                sx, sy, sz = (1.0, 1.0, 1.0)
        for roi in self.rois:
            color, text_label = self._get_roi_draw_params(roi)
            ci, cj, ck = roi["center_voxel"]
            r_mm = roi["radius_mm"]
            if plane == "axial":
                d_mm = abs((slice_index - ck) * sz)
                if d_mm > r_mm:
                    continue
                r2d_mm = (r_mm * r_mm - d_mm * d_mm) ** 0.5
                x = ci * sx
                y = cj * sy
            elif plane == "sagittal":
                d_mm = abs((slice_index - ci) * sx)
                if d_mm > r_mm:
                    continue
                r2d_mm = (r_mm * r_mm - d_mm * d_mm) ** 0.5
                x = cj * sy
                y = ck * sz
            elif plane == "coronal":
                d_mm = abs((slice_index - cj) * sy)
                if d_mm > r_mm:
                    continue
                r2d_mm = (r_mm * r_mm - d_mm * d_mm) ** 0.5
                x = ci * sx
                y = ck * sz
            else:
                continue
            ax.plot(x, y, marker="+", color=color, markersize=8, alpha=0.8)
            ellipse = Ellipse(
                (x, y),
                width=r2d_mm * 2.0,
                height=r2d_mm * 2.0,
                fill=False,
                color=color,
                linestyle="-",
                alpha=0.8,
                linewidth=1.5,
            )
            ax.add_patch(ellipse)
            ax.text(x + 2, y + 2, text_label, color=color, fontsize=7)

    def _draw_gt_on_plane(self, ax, plane, slice_index):
        if not self.show_gt or not self.gt_lesions or self.meta is None or self.np_vol is None:
            return
        sz_k, sz_j, sz_i = self.np_vol.shape
        try:
            sx, sy, sz = self.meta["spacing"]
        except Exception:
            try:
                sx, sy, sz = self.sitk_img.GetSpacing()
            except Exception:
                sx, sy, sz = (1.0, 1.0, 1.0)
        for idx, lesion in enumerate(self.gt_lesions, 1):
            x_mm, y_mm, z_mm = lesion["xyz_mm"]
            vi, vj, vk = dicom_io.mm_to_voxel(x_mm, y_mm, z_mm, self.meta)
            vi_int = int(round(vi))
            vj_int = int(round(vj))
            vk_int = int(round(vk))
            if not (0 <= vk_int < sz_k and 0 <= vi_int < sz_i and 0 <= vj_int < sz_j):
                continue
            if plane == "axial":
                if vk_int != slice_index:
                    continue
                x = vi_int * sx
                y = vj_int * sy
            elif plane == "sagittal":
                if vi_int != slice_index:
                    continue
                x = vj_int * sy
                y = vk_int * sz
            elif plane == "coronal":
                if vj_int != slice_index:
                    continue
                x = vi_int * sx
                y = vk_int * sz
            else:
                continue
            lid = lesion.get("lesion_id") or f"L{idx}"
            label = f"GT {lid}"
            clinsig = lesion.get("clinsig")
            zone = lesion.get("zone")
            extra = []
            if clinsig is not None:
                extra.append(f"ClinSig={clinsig}")
            if zone is not None:
                extra.append(f"zone={zone}")
            ggg = lesion.get("ggg")
            isup = lesion.get("isup")
            if not extra:
                if ggg:
                    extra.append(f"GGG={ggg}")
                if isup:
                    extra.append(f"ISUP={isup}")
            if extra:
                label += " | " + " | ".join(extra)
            ax.plot(x, y, marker="x", color="magenta", markersize=8, linewidth=1.5)
            ax.text(
                x + 2,
                y + 2,
                label,
                color="magenta",
                fontsize=7,
                fontweight="bold",
            )

    def update_plot(self):
        if self.fig is None:
            return
        self._apply_slots()
        if self.ax_axial:
            self.ax_axial.clear()
            self.ax_axial.set_facecolor("black")
            self.ax_axial.set_xticks([])
            self.ax_axial.set_yticks([])
        if self.ax_sag:
            self.ax_sag.clear()
            self.ax_sag.set_facecolor("black")
            self.ax_sag.set_xticks([])
            self.ax_sag.set_yticks([])
        if self.ax_cor:
            self.ax_cor.clear()
            self.ax_cor.set_facecolor("black")
            self.ax_cor.set_xticks([])
            self.ax_cor.set_yticks([])
        if self.ax_info:
            self.ax_info.clear()
            self.ax_info.set_facecolor("#202020")
            self.ax_info.axis('off')
        if hasattr(self, "ax_sidebar") and self.ax_sidebar:
            self.ax_sidebar.clear()
            self.ax_sidebar.set_facecolor("#303030")
            self.ax_sidebar.axis('off')

        if self.np_vol is not None:
            sz_k, sz_j, sz_i = self.np_vol.shape
            i, j, k = self.center_voxel
            i = int(max(0, min(i, sz_i - 1)))
            j = int(max(0, min(j, sz_j - 1)))
            k = int(max(0, min(k, sz_k - 1)))
            self._set_center_voxel(i, j, k)
            layout = self._get_layout_assignment()
            self._render_mpr_view(self.ax_axial, layout["main"])
            self._render_mpr_view(self.ax_cor, layout["bottom_left"])
            self._render_mpr_view(self.ax_sag, layout["bottom_right"])

        layout = self._get_layout_assignment()
        self._style_panel(self.ax_axial, layout["main"])
        self._style_panel(self.ax_cor, layout["bottom_left"])
        self._style_panel(self.ax_sag, layout["bottom_right"])

        case_name = self.cases_list[self.current_case_idx] if self.cases_list else "None"
        if self.mode == "SERIES_SELECT":
            mode_str = f"GO TO SERIES: {self.series_input_str}_ (Enter confirm, Esc cancel)"
        elif self.mode == "CASE_SELECT":
            mode_str = f"GO TO PATIENT: {self.case_input_str}_ (Enter confirm, Esc cancel)"
        else:
            mode_str = "LOCKED" if self.is_locked else "PREVIEW"
        if self.series_list and self.current_series_idx < len(self.series_list):
            s = self.series_list[self.current_series_idx]
            cv_i, cv_j, cv_k = self.center_voxel
            line1 = f"CASE: {case_name} | SERIES: {s['series_name'][:20]} ({s['orientation'].upper()})"
            line2 = f"CENTER (i,j,k)=({cv_i},{cv_j},{cv_k}) | R: {self.radius_mm:.1f} mm | MODE: {mode_str}"
        else:
            line1 = f"CASE: {case_name} | NO SERIES LOADED"
            line2 = f"MODE: {mode_str}"
        hud_text = f"{line1}\n{line2}"
        if self.last_message:
            hud_text += f"\nLAST: {self.last_message}"
        if self.last_key:
            hud_text += f"\nKEY: {self.last_key}"
        if self.ax_axial:
            self.ax_axial.text(
                0.01,
                1.01,
                hud_text,
                transform=self.ax_axial.transAxes,
                verticalalignment="bottom",
                horizontalalignment="left",
                family="monospace",
                fontsize=8,
                color="white",
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.4"),
            )

        if self.dev_layout_debug:
            self._debug_slot_sizes()

        if self.toast_artist is not None:
            try:
                self.toast_artist.remove()
            except Exception:
                pass
            self.toast_artist = None
        if self.toast_message and time.time() < self.toast_until:
            self.toast_artist = self.fig.text(
                0.5,
                0.02,
                self.toast_message,
                transform=self.fig.transFigure,
                ha="center",
                va="bottom",
                family="monospace",
                fontsize=10,
                color="yellow",
                bbox=dict(facecolor="black", alpha=0.8, edgecolor="yellow", boxstyle="round,pad=0.4"),
            )
        elif self.toast_message and time.time() >= self.toast_until:
            self.toast_message = None

        total_series_pages = (len(self.series_list) - 1) // self.series_per_page + 1 if self.series_list else 0
        sidebar_text = ""
        if self.cases_list:
            sidebar_text += "=== PATIENTS ===\n"
            for i, c in enumerate(self.cases_list):
                mark = ">" if i == self.current_case_idx else " "
                sidebar_text += f"{mark}[{i+1}] {c[:15]}\n"
            sidebar_text += "\n"
        if self.series_list:
            sidebar_text += "=== T2 QUICK ===\n"
            for orient in ['axial', 'coronal', 'sagittal']:
                idx = self.t2_quick[orient]
                mark = ">" if idx == self.current_series_idx and idx is not None else " "
                key_map = {'axial': 'A', 'coronal': 'K', 'sagittal': 'S'}
                key = key_map[orient]
                if idx is not None:
                    name = self.series_list[idx]['series_name'][:12]
                    sidebar_text += f"{mark}({key}) {orient[:3].upper()}: {name}\n"
                else:
                    sidebar_text += f"   ({key}) {orient[:3].upper()}: -\n"
            sidebar_text += "\n"
            sidebar_text += f"=== SERIES ({self.series_page+1}/{total_series_pages}) ===\n"
            start_idx = self.series_page * self.series_per_page
            end_idx = min(start_idx + self.series_per_page, len(self.series_list))
            for i in range(start_idx, end_idx):
                ser = self.series_list[i]
                mark = ">" if i == self.current_series_idx else " "
                sidebar_text += f"{mark}[{i+1}] {ser['series_name'][:12]} ({ser['orientation'][0].upper()})\n"
            sidebar_text += "\nUse [ ] to page\n"
        sidebar_text += "\n=== ROIs ===\n"
        if not self.rois:
            sidebar_text += "Nenhuma ROI.\n"
        else:
            for roi in self.rois:
                lid = roi["id"]
                status = self.roi_status.get(lid, "??")
                sidebar_text += (
                    f"{lid} | S:{roi['center_voxel'][2]} | R:{roi['radius_mm']:.1f} | {status}\n"
                    f"  Pos:({roi['center_voxel'][0]},{roi['center_voxel'][1]})\n"
                )
        sidebar_text += "\n=== PREDICOES ===\n"
        if self.show_predictions_panel and self.last_preds:
            for p in self.last_preds:
                lesion = p.get("lesion", "?")
                perc = p.get("risk_percent", p.get("prob_pos", 0.0) * 100.0)
                cat = p.get("risk_category", "")
                label = p.get("pred_label", "")
                sidebar_text += f"{lesion}: {perc:.0f}% ({cat}) -> {label}\n"
        else:
            sidebar_text += "Nenhuma.\n"
        sidebar_text += "\n=== GT ===\n"
        if self.show_gt:
            sidebar_text += f"ON ({len(self.gt_lesions)} lesoes)\n"
        else:
            sidebar_text += "OFF\n"
        if self.show_help:
            sidebar_text += "\n" + self._get_help_text()
        if hasattr(self, "ax_sidebar") and self.ax_sidebar:
            self.ax_sidebar.text(
                0.02,
                0.98,
                sidebar_text,
                transform=self.ax_sidebar.transAxes,
                verticalalignment="top",
                family="monospace",
                fontsize=8,
                color="white",
            )

        info_panel_text = ""
        if self.series_list and self.current_series_idx < len(self.series_list):
            s = self.series_list[self.current_series_idx]
            info_panel_text += f"CASE: {case_name}\n"
            info_panel_text += f"SERIES: {s['series_name']} ({s['orientation'].upper()})\n"
        else:
            info_panel_text += f"CASE: {case_name}\nNO SERIES LOADED\n"
        info_panel_text += f"ROIs: {len(self.rois)}\n"
        if self.show_predictions_panel and self.last_preds:
            info_panel_text += f"PREDICOES: {len(self.last_preds)}\n"
        if self.show_gt:
            info_panel_text += f"GT lesions: {len(self.gt_lesions)}\n"
        if self.show_gt:
            info_panel_text += "\nGT DETAIL:\n"
            info_panel_text += f"patient_id_resolved={self.gt_patient_id or 'None'}\n"
            info_panel_text += f"labels_source={self.gt_label_source or 'None'}\n"
            if not self.gt_lesions:
                info_panel_text += "GT indisponivel: labels nao encontradas ou mapping ausente\n"
            elif self.meta is None or self.np_vol is None:
                info_panel_text += "GT coords nao projetaveis\n"
            else:
                sz_k, sz_j, sz_i = self.np_vol.shape
                any_proj = False
                any_oob = False
                for idx, lesion in enumerate(self.gt_lesions, 1):
                    x, y, z = lesion["xyz_mm"]
                    vi, vj, vk = dicom_io.mm_to_voxel(x, y, z, self.meta)
                    vi_int = int(round(vi))
                    vj_int = int(round(vj))
                    vk_int = int(round(vk))
                    in_bounds = (
                        0 <= vk_int < sz_k and
                        0 <= vi_int < sz_i and
                        0 <= vj_int < sz_j
                    )
                    if in_bounds:
                        any_proj = True
                    else:
                        any_oob = True
                    lid = lesion.get("lesion_id") or f"L{idx}"
                    info_panel_text += f"{lid}: voxel=({vi_int},{vj_int},{vk_int}) slice={vk_int} in_bounds={str(in_bounds).lower()}\n"
                if not any_proj:
                    info_panel_text += "GT coords nao projetaveis\n"
                if any_oob:
                    info_panel_text += "GT fora do volume desta serie.\n"
        if self.show_gt and self.rois and self.gt_lesions:
            info_panel_text += "\nROI vs GT:\n"
            for roi in self.rois:
                best = None
                for lesion in self.gt_lesions:
                    x, y, z = lesion["xyz_mm"]
                    cx, cy, cz = roi["center_mm"]
                    dx = cx - x
                    dy = cy - y
                    dz = cz - z
                    dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                    if best is None or dist < best[0]:
                        best = (dist, lesion)
                if best is not None:
                    dist_mm, _ = best
                    status = "PERTO" if dist_mm <= self.gt_threshold_mm else "LONGE"
                    info_panel_text += f"{roi['id']} -> {dist_mm:.1f} mm ({status})\n"
        status_flags = []
        if self.last_export_dir:
            status_flags.append("Export OK")
        if self.last_preds:
            status_flags.append("Model OK")
        if self.show_gt and self.gt_lesions:
            status_flags.append("GT OK")
        if status_flags:
            info_panel_text += "\nStatus: " + ", ".join(status_flags) + "\n"
        if self.ax_info:
            self.ax_info.text(
                0.05,
                0.95,
                info_panel_text,
                transform=self.ax_info.transAxes,
                verticalalignment='top',
                family='monospace',
                fontsize=8,
                color="white",
            )

        self.fig.canvas.draw_idle()

    def _toggle_gt(self):
        if not self.cases_list or self.current_case_idx < 0:
            return
        if not self.gt_lesions:
            self._load_gt_for_case()
        self.show_gt = not self.show_gt
        if self.show_gt and not self.gt_lesions:
            self.last_message = "GT indisponivel"
        self.update_plot()

    def _jump_to_gt_slice(self):
        if not self.gt_lesions or self.meta is None or self.np_vol is None:
            return
        if self.rois:
            ref = self.rois[-1]["center_mm"]
        else:
            ref = self.gt_lesions[0]["xyz_mm"]
        best = None
        for lesion in self.gt_lesions:
            x, y, z = lesion["xyz_mm"]
            cx, cy, cz = ref
            dx = cx - x
            dy = cy - y
            dz = cz - z
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if best is None or dist < best[0]:
                best = (dist, lesion)
        if best is None:
            return
        x, y, z = best[1]["xyz_mm"]
        vi, vj, vk = dicom_io.mm_to_voxel(x, y, z, self.meta)
        i = int(round(vi))
        j = int(round(vj))
        k = int(round(vk))
        self._set_center_voxel(i, j, k)
        self.last_message = "Pulando para GT mais proxima"
        self.update_plot()

    def on_mouse_move(self, event):
        if self.mode in ["SERIES_SELECT", "CASE_SELECT"]:
            return
        if event.inaxes is None:
            return
        plane = self._plane_for_axes(event.inaxes)
        if plane is None:
            return
        if plane != self.active_view:
            self.active_view = plane
            if self.fig:
                self.update_plot()
        if self._pan_drag["active"] and self._pan_drag["plane"] == plane:
            ax = event.inaxes
            if ax is None:
                return
            start_x, start_y = self._pan_drag["start_xy"]
            dx_px = event.x - start_x
            dy_px = event.y - start_y
            x0, x1 = ax.get_xlim()
            y1, y0 = ax.get_ylim()
            width_px = max(ax.bbox.width, 1.0)
            height_px = max(ax.bbox.height, 1.0)
            mm_per_px_x = (x1 - x0) / width_px
            mm_per_px_y = (y0 - y1) / height_px if (y0 - y1) != 0 else 0.0
            dx_mm = -dx_px * mm_per_px_x
            dy_mm = -dy_px * mm_per_px_y
            state = self.view_state.get(plane)
            if state is not None and state.get("xlim") and state.get("ylim"):
                xlim0, xlim1 = state["xlim"]
                ylim0, ylim1 = state["ylim"]
                state["xlim"] = (xlim0 + dx_mm, xlim1 + dx_mm)
                state["ylim"] = (ylim0 + dy_mm, ylim1 + dy_mm)
            self.update_plot()
            return
        if self._wl_drag["active"] and self._wl_drag["plane"] == plane:
            start_x, start_y = self._wl_drag["start_xy"]
            base_win = self._wl_drag["base_win"]
            base_level = self._wl_drag["base_level"]
            if base_win is None or base_level is None:
                self._reset_wl_for_plane(plane, use_full_volume=False)
                base_win, base_level = self._get_wl_for_plane(plane)
                self._wl_drag["base_win"] = base_win
                self._wl_drag["base_level"] = base_level
            dx = event.x - start_x
            dy = event.y - start_y
            if base_win is None or base_level is None:
                return
            factor = 1.0 + dx * 0.01
            if factor <= 0.0:
                factor = 0.01
            new_win = max(base_win * factor, 1e-3)
            new_level = base_level - dy * (base_win * 0.01)
            self.win[plane] = new_win
            self.level[plane] = new_level
            self.update_plot()
            return

    def on_scroll(self, event):
        if self.mode in ["SERIES_SELECT", "CASE_SELECT"]:
            return
        if self.np_vol is None:
            return
        plane = self._plane_for_axes(event.inaxes)
        if plane is None:
            return
        if event.button == "up":
            factor = 1.1
        elif event.button == "down":
            factor = 1.0 / 1.1
        else:
            return
        state = self.view_state.get(plane)
        if state is None or not state.get("xlim") or not state.get("ylim"):
            return
        x0, x1 = state["xlim"]
        y0, y1 = state["ylim"]
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        half_w = 0.5 * (x1 - x0)
        half_h = 0.5 * (y1 - y0)
        new_half_w = half_w / factor
        new_half_h = half_h / factor
        state["xlim"] = (cx - new_half_w, cx + new_half_w)
        state["ylim"] = (cy - new_half_h, cy + new_half_h)
        state["zoom"] = max(1.0, state.get("zoom", 1.0) * factor)
        self.update_plot()

    def on_click(self, event):
        if self.mode in ["SERIES_SELECT", "CASE_SELECT"]:
            return
        if event.inaxes is None:
            return
        try:
            sx, sy, sz = self.meta["spacing"]
        except Exception:
            try:
                sx, sy, sz = self.sitk_img.GetSpacing()
            except Exception:
                sx, sy, sz = (1.0, 1.0, 1.0)
        plane = self._plane_for_axes(event.inaxes)
        if plane is None:
            return
        self.active_view = plane
        if event.dblclick and event.button == 1:
            state = self.view_state.get(plane)
            if state is not None:
                state["mode"] = "FULL"
                state["xlim"] = None
                state["ylim"] = None
                state["zoom"] = 1.0
                state["pan"] = (0.0, 0.0)
            self.update_plot()
            return
        if event.button == 2:
            self._pan_drag["active"] = True
            self._pan_drag["plane"] = plane
            self._pan_drag["start_xy"] = (event.x, event.y)
            self._pan_drag["start_pan"] = None
            return
        if event.button == 3:
            if self.meta is None or self.np_vol is None:
                return
            self._wl_drag["active"] = True
            self._wl_drag["plane"] = plane
            self._wl_drag["start_xy"] = (event.x, event.y)
            win, level = self._get_wl_for_plane(plane)
            if win is None or level is None:
                self._reset_wl_for_plane(plane, use_full_volume=False)
                win, level = self._get_wl_for_plane(plane)
            self._wl_drag["base_win"] = win
            self._wl_drag["base_level"] = level
            return
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.meta is None or self.np_vol is None:
            return
        i, j, k = self.center_voxel
        if plane == "axial":
            i = int(round(event.xdata / sx))
            j = int(round(event.ydata / sy))
        elif plane == "sagittal":
            j = int(round(event.xdata / sy))
            k = int(round(event.ydata / sz))
        elif plane == "coronal":
            i = int(round(event.xdata / sx))
            k = int(round(event.ydata / sz))
        self._set_center_voxel(i, j, k)
        self.candidate_center = [i, j, k]
        self.is_locked = True
        self.last_message = "Centro travado. Pressione Enter para confirmar."
        self.update_plot()

    def on_key(self, event):
        try:
            self.last_key = event.key
            print(f"[KEY] mode={self.mode} key={event.key}")
            if event.key == 'q':
                plt.close()
                return

            is_ctrl = event.key.startswith('ctrl+')

            if self.mode == "NORMAL" and not is_ctrl and event.key in ['a', 'k', 's']:
                mapping = {'a': 'axial', 'k': 'coronal', 's': 'sagittal'}
                target = mapping.get(event.key)
                if target:
                    self.main_view = target
                    self.active_view = target
                    self.last_message = f"Painel principal: {target.upper()}"
                    self.update_plot()
                    return

            if self.mode == "NORMAL" and event.key == 'g':
                self._toggle_gt()
                return

            if self.mode == "NORMAL" and event.key == 'G':
                self._jump_to_gt_slice()
                return

            # Modo Go-to Series (Ctrl+G)
            if self.mode == "NORMAL" and event.key == 'ctrl+g':
                self.mode = "SERIES_SELECT"
                self.series_input_str = ""
                self.update_plot()
                return

            # Modo Abrir Pasta (O)
            if self.mode == "NORMAL" and event.key == 'o':
                self.open_data_root()
                return

            # Modo de Seleção de Caso (Captura de Teclas)
            if self.mode == "CASE_SELECT":
                 if event.key == 'enter':
                     try:
                         n = int(self.case_input_str)
                         print(f"[DEBUG] Go-to case: {n}")
                         if self.load_case(n - 1):
                             self.mode = "NORMAL"
                         else:
                             self.last_message = f"Caso {n} invalido (1..{len(self.cases_list)})"
                     except ValueError:
                         self.last_message = "Entrada invalida (digite apenas numeros)"
                     
                     self.case_input_str = ""
                     self.update_plot()
                 elif event.key == 'escape':
                     self.mode = "NORMAL"
                     self.case_input_str = ""
                     self.last_message = "Selecao de caso cancelada"
                     self.update_plot()
                 elif event.key == 'backspace':
                     self.case_input_str = self.case_input_str[:-1]
                     self.update_plot()
                 elif event.key is not None and len(event.key) == 1 and event.key.isdigit():
                     self.case_input_str += event.key
                     self.update_plot()
                 return

             # Modo de Seleção de Série (Captura de Teclas)
            if self.mode == "SERIES_SELECT":
                if event.key == 'enter':
                    try:
                        n = int(self.series_input_str)
                        print(f"[DEBUG] Go-to series: {n}")
                        idx = n - 1
                        if 0 <= idx < len(self.series_list):
                            self.current_series_idx = idx
                            self.load_current_series()
                            self.last_message = f"Trocado para serie {n}"
                            self.series_page = idx // self.series_per_page
                            self.mode = "NORMAL"
                        else:
                            self.last_message = f"Indice {n} invalido (1..{len(self.series_list)})"
                    except ValueError:
                        self.last_message = "Entrada invalida (digite apenas numeros)"
                    
                    self.series_input_str = ""
                    self.update_plot()
                elif event.key == 'escape':
                    self.mode = "NORMAL"
                    self.series_input_str = ""
                    self.last_message = "Selecao cancelada"
                    self.update_plot()
                elif event.key == 'backspace':
                    self.series_input_str = self.series_input_str[:-1]
                    self.update_plot()
                elif event.key is not None and len(event.key) == 1 and event.key.isdigit():
                    self.series_input_str += event.key
                    self.update_plot()
                return

            # Controles Normais (Modo NORMAL)
            elif event.key == 'c':
                self.mode = "CASE_SELECT"
                self.case_input_str = ""
                self.update_plot()
            elif event.key == 'ctrl+up':
                self.next_patient(-1)
            elif event.key == 'ctrl+down':
                self.next_patient(1)
            elif event.key in ['up', 'right']:
                self._move_center_slice(self.active_view, 1)
                self.update_plot()
            elif event.key in ['down', 'left']:
                self._move_center_slice(self.active_view, -1)
                self.update_plot()
            elif event.key == 'z':
                for plane in ["axial", "coronal", "sagittal"]:
                    state = self.view_state.get(plane)
                    if state is not None:
                        state["mode"] = "CROP" if state.get("mode") != "CROP" else "FULL"
                        state["xlim"] = None
                        state["ylim"] = None
                self.update_plot()
            elif event.key == 'd':
                self.dev_layout_debug = not self.dev_layout_debug
                self.last_message = f"DEV DEBUG: {'ON' if self.dev_layout_debug else 'OFF'}"
                self.update_plot()
            elif event.key == 'i':
                self.interp_mode = "nearest" if self.interp_mode != "nearest" else "bilinear"
                self.last_message = f"Interpolacao: {self.interp_mode}"
                self.update_plot()
            elif event.key == 'r':
                state = self.view_state.get(self.active_view)
                if state is not None:
                    state["mode"] = "FULL"
                    state["xlim"] = None
                    state["ylim"] = None
                    state["zoom"] = 1.0
                    state["pan"] = (0.0, 0.0)
                self.update_plot()
            elif event.key == 'R':
                for plane in ["axial", "coronal", "sagittal"]:
                    state = self.view_state.get(plane)
                    if state is not None:
                        state["mode"] = "FULL"
                        state["xlim"] = None
                        state["ylim"] = None
                        state["zoom"] = 1.0
                        state["pan"] = (0.0, 0.0)
                self.update_plot()
            elif event.key in ['+', '=']:
                self.radius_mm += 0.5
                self.update_plot()
            elif event.key in ['-', '_']:
                self.radius_mm = max(0.5, self.radius_mm - 0.5)
                self.update_plot()
            elif event.key == 'x':
                self.is_locked = False
                self.candidate_center = None
                self.last_message = "Selecao limpa."
                self.update_plot()
            elif event.key == 'enter':
                self.confirm_roi()
            elif event.key == 'delete':
                self.delete_last_roi()
            elif event.key in ['j', 'ctrl+s']:
                self.save_json()
            elif event.key == 'e':
                self.export_all_to_pipeline()
            elif event.key == 'f':
                self.open_last_export_dir()
            elif event.key == 'v':
                self.validate_rois()
            elif event.key == 'p':
                self.show_predictions_panel = not self.show_predictions_panel
                if self.show_predictions_panel:
                    print("Pred panel enabled")
                else:
                    print("Pred panel disabled")
                self.update_plot()
            elif event.key == 'h':
                self.show_help = not self.show_help
                self.update_plot()
            elif event.key == ']':
                max_pages = (len(self.series_list) - 1) // self.series_per_page
                self.series_page = min(self.series_page + 1, max_pages)
                self.update_plot()
            elif event.key == '[':
                self.series_page = max(self.series_page - 1, 0)
                self.update_plot()
            elif event.key is not None and len(event.key) == 1 and event.key.isdigit() and event.key != '0':
                # Atalhos 1-9 (apenas se não for 0)
                idx = int(event.key) - 1
                if idx < len(self.series_list):
                    self.current_series_idx = idx
                    self.load_current_series()
                    self.last_message = f"Trocado para serie {idx+1}"
                    self.update_plot()
        except Exception:
            print("\n[ERRO] Excecao em on_key:")
            traceback.print_exc()

    def delete_last_roi(self):
        if not self.rois:
            self.last_message = "Nenhuma ROI para remover"
        else:
            removed = self.rois.pop()
            self.last_message = f"ROI removida: {removed['id']}"
            
            # Atualizar memória e autosave
            case_name = self.cases_list[self.current_case_idx]
            self.rois_by_patient[case_name] = self.rois
            self._autosave_rois()
            
            if not self.rois:
                self.lesion_counter = 1
            else:
                # Extrair numero do id "L3" -> 3
                try:
                    last_id = int(self.rois[-1]['id'][1:])
                    self.lesion_counter = last_id + 1
                except:
                    self.lesion_counter = len(self.rois) + 1
        self.update_plot()

    def confirm_roi(self):
        if not self.is_locked or not self.candidate_center:
            self.last_message = "AVISO: Trave o centro com clique primeiro!"
            self.update_plot()
            return
            
        i, j, k = self.candidate_center
        s = self.series_list[self.current_series_idx]
        
        # converter para MM
        x, y, z = dicom_io.voxel_to_mm(i, j, k, self.meta)
        
        roi = {
            "id": f"L{self.lesion_counter}",
            "center_voxel": [int(round(i)), int(round(j)), int(k)],
            "center_mm": [float(x), float(y), float(z)],
            "radius_mm": float(self.radius_mm),
            "series_uid": s['series_uid']
        }
        
        self.rois.append(roi)
        self.last_message = f"ROI L{self.lesion_counter} confirmada! (autosave OK)"
        
        # Atualizar memória e autosave
        case_name = self.cases_list[self.current_case_idx]
        self.rois_by_patient[case_name] = self.rois
        self._autosave_rois()
        
        # Exportar PNG e CSV
        self._export_roi_assets(roi)
        
        self.lesion_counter += 1
        self.is_locked = False
        self.candidate_center = None
        self.update_plot()

    def _export_roi_assets(self, roi):
        """Salva PNG do slice atual com a ROI e atualiza manifest.csv."""
        case_name = self.cases_list[self.current_case_idx]
        s = self.series_list[self.current_series_idx]
        plane = s['orientation']
        
        # 1. Salvar PNG
        # Vamos usar o canvas atual para capturar a imagem com as ROIs desenhadas
        # mas apenas do eixo principal.
        # Para garantir qualidade, vamos salvar o slice atual separadamente se necessário, 
        # mas o requisito pede "imagem do slice atual com a ROI desenhada".
        
        roi_id = roi['id']
        png_filename = f"case{case_name}_series{self.current_series_idx}_{plane}_slice{self.current_slice}_roi{roi_id}.png"
        png_path = os.path.join(self.roi_img_dir, png_filename)
        
        try:
            # Criar uma figura temporária para o export sem o HUD/Painel
            fig_tmp, ax_tmp = plt.subplots(figsize=(8, 8))
            ax_tmp.imshow(self.np_vol[self.current_slice, :, :], cmap='gray')
            
            # Desenhar a ROI confirmada (e outras se houver)
            # Para o export, vamos focar na ROI atual em destaque (vermelha ou verde)
            # Mas o requisito diz "com a ROI desenhada". Vamos desenhar a atual.
            ci, cj, ck = roi['center_voxel']
            r_mm = roi['radius_mm']
            
            # Cálculo de elipse para o export (mesma lógica do _draw_roi_sphere)
            p_center = dicom_io.voxel_to_mm(ci, cj, ck, self.meta)
            p_here = dicom_io.voxel_to_mm(ci, cj, self.current_slice, self.meta)
            dz_mm = abs(p_here[2] - p_center[2])
            
            if dz_mm < r_mm:
                r_slice_mm = (r_mm**2 - dz_mm**2)**0.5
                r_px_x = r_slice_mm / self.meta['spacing'][0]
                r_px_y = r_slice_mm / self.meta['spacing'][1]
                
                ellipse = Ellipse((ci, cj), width=r_px_x*2, height=r_px_y*2, 
                                 fill=False, color='lime', linewidth=2)
                ax_tmp.add_patch(ellipse)
                ax_tmp.plot(ci, cj, '+', color='lime', markersize=10)
                
            ax_tmp.text(5, 15, f"CASE: {case_name} | SERIES: {self.current_series_idx}\nSLICE: {self.current_slice} | ROI: {roi_id} | R: {r_mm}mm", 
                        color='yellow', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
            
            ax_tmp.axis('off')
            plt.tight_layout()
            fig_tmp.savefig(png_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig_tmp)
            
        except Exception as e:
            print(f"[ERROR] Falha ao exportar PNG: {e}")

        # 2. Atualizar CSV Manifest
        manifest_path = os.path.join(self.export_dir, "roi_manifest.csv")
        file_exists = os.path.isfile(manifest_path)
        
        try:
            with open(manifest_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "case", "plane", "series_index", "series_name", "slice_index", "x", "y", "radius_mm", "dicom_dir"])
                
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    case_name,
                    plane,
                    self.current_series_idx,
                    s['series_name'],
                    self.current_slice,
                    f"{roi['center_mm'][0]:.2f}",
                    f"{roi['center_mm'][1]:.2f}",
                    roi['radius_mm'],
                    self.dicom_root
                ])
        except Exception as e:
            print(f"[ERROR] Falha ao atualizar manifest.csv: {e}")

    def open_last_export_dir(self):
        """Abre a pasta do último export no explorador de arquivos."""
        if not self.last_export_dir or not os.path.exists(self.last_export_dir):
            self.last_message = "Nenhum export realizado nesta sessao."
            self.update_plot()
            return
            
        try:
            if sys.platform == 'win32':
                os.startfile(self.last_export_dir)
            elif sys.platform == 'darwin': # macOS
                import subprocess
                subprocess.Popen(['open', self.last_export_dir])
            else: # linux
                import subprocess
                subprocess.Popen(['xdg-open', self.last_export_dir])
            self.last_message = "Abrindo pasta de export..."
        except Exception as e:
            self.last_message = f"Erro ao abrir pasta: {str(e)[:20]}"
        self.update_plot()

    def open_data_root(self):
        """Abre seletor de pastas para mudar o data_root."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            new_root = filedialog.askdirectory(title="Selecione a PASTA RAIZ (ex: PROSTATEx ou SAMPLES)")
            root.destroy()
            
            if new_root:
                old_root = self.input_root
                self.input_root = os.path.abspath(new_root)
                # Resetar samples_root para forçar redescobrir se o usuário mudou de pasta
                self.samples_root = None 
                self.discover_workspace()
                
                if self.cases_list:
                    # Sucesso: Salvar e carregar primeiro caso
                    self._save_config()
                    self.load_case(0)
                    self.last_message = f"Raiz configurada: {len(self.cases_list)} pacientes."
                else:
                    # Falha: Reverter e avisar (discover_workspace já setou last_message)
                    self.input_root = old_root
                    self.discover_workspace()
                
                if self.fig: self.update_plot()
        except Exception as e:
            print(f"[WARNING] Erro ao abrir seletor de pastas: {e}")
            self.last_message = "Erro ao abrir seletor de pastas."
            if self.fig: self.update_plot()

    def export_all_to_pipeline(self):
        """Exporta ROIs em JSON e máscaras NIfTI para o pipeline."""
        if not self.rois:
            self.last_message = "AVISO: Nenhuma ROI para exportar."
            self.update_plot()
            return

        case_name = self.cases_list[self.current_case_idx]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        export_case_dir = os.path.join(self.export_dir, case_name, timestamp)
        os.makedirs(export_case_dir, exist_ok=True)

        try:
            # 1. Exportar JSON
            json_path = os.path.join(export_case_dir, "rois.json")
            roi_export.save_roi_json(json_path, case_name, self.rois, self.input_root)
            
            # 2. Exportar Máscaras NIfTI
            mask_export.export_roi_masks(export_case_dir, self.sitk_img, self.rois, case_name)
            
            self.last_message = f"Export OK: {case_name}/{timestamp}"
            self.last_export_dir = export_case_dir
            print(f"\n[INFO] Export Pipeline completo em: {export_case_dir}")
            
            # Inferência imediata usando ponte (features CSV + venv_infer)
            try:
                dicom_dir = Path(self.meta['series_dir']) if (self.meta and 'series_dir' in self.meta) else Path(self.dicom_root or '')
                export_dir = Path(self.last_export_dir)
                preds = predict_for_export_folder(dicom_dir=dicom_dir, export_dir=export_dir)
                if preds:
                    self.last_preds = preds
                    self.roi_pred_map = {}
                    for p in preds:
                        lid = p.get("lesion")
                        if lid:
                            self.roi_pred_map[lid] = p
                    if len(preds) == 1:
                        first = preds[0]
                        thr = first.get('thr_cv', first.get('threshold', 0.5))
                        lesion = first.get('lesion', 'ROI')
                        perc = first.get('risk_percent', first['prob_pos'] * 100.0)
                        cat = first.get('risk_category', '')
                        self.last_message = f"PRED {lesion}: {perc:.0f}% ({cat}) | thr={thr:.3f} -> {first['pred_label']}"
                    else:
                        parts = []
                        for p in preds:
                            lesion = p.get('lesion', '?')
                            perc = p.get('risk_percent', p['prob_pos'] * 100.0)
                            cat = p.get('risk_category', '')
                            parts.append(f"{lesion}={perc:.0f}%({cat})")
                        summary = ", ".join(parts)
                        self.last_message = f"PRED: {summary}"
                    self.toast_message = self.last_message
                    self.toast_until = time.time() + 8.0
                    print("HUD prediction toast enabled")
                else:
                    self.last_message = "INFER ERROR: sem resultados"
            except Exception as e:
                self.last_message = f"INFER ERROR: {e}"
        except Exception as e:
            self.last_message = f"ERRO no export: {str(e)[:20]}..."
            print(f"[ERROR] Falha no export pipeline: {e}")
            traceback.print_exc()
        
        self.update_plot()

    def validate_rois(self):
        """Valida se as ROIs intersectam o volume atual."""
        if not self.rois:
            self.last_message = "AVISO: Nenhuma ROI para validar."
            self.update_plot()
            return

        print("\n=== VALIDAÇÃO DE ROIS ===")
        all_valid = True
        for roi in self.rois:
            center_mm = roi['center_mm']
            radius_mm = roi['radius_mm']
            roi_id = roi['id']
            
            # Converter centro para voxel na imagem atual
            try:
                # Usar SimpleITK para transformar ponto físico em índice
                # Se estiver fora do bounding box físico, lança exceção ou retorna valores fora
                continuous_idx = self.sitk_img.TransformPhysicalPointToContinuousIndex(center_mm)
                size = self.sitk_img.GetSize() # (x, y, z)
                
                # Checar se o centro está dentro ou perto o suficiente para a esfera intersectar
                is_outside = False
                for i in range(3):
                    # Se o centro estiver mais longe que o raio das bordas do volume
                    # Simplificação: checar se o centro está nos limites [0, size]
                    if continuous_idx[i] < -0.5 or continuous_idx[i] > size[i] - 0.5:
                        is_outside = True
                        break
                
                if is_outside:
                    msg = f"WARN: ROI {roi_id} fora do volume!"
                    print(f"[WARNING] {msg} (Centro: {center_mm}, Volume Size: {size})")
                    self.last_message = msg
                    all_valid = False
            except Exception as e:
                msg = f"WARN: Erro ao validar {roi_id}"
                print(f"[ERROR] {msg}: {e}")
                self.last_message = msg
                all_valid = False

        if all_valid:
            self.last_message = "Todas as ROIs validas no volume atual."
            print("[INFO] Todas as ROIs estao dentro dos limites do volume.")
        
        self.update_plot()

    def save_json(self):
        """Exporta JSON completo e padronizado em exports/ com metadados geométricos."""
        case_name = self.cases_list[self.current_case_idx]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"roi_selection_{case_name}_{timestamp}.json"
        
        # Garantir diretório de export
        os.makedirs(self.export_dir, exist_ok=True)
        output_path = os.path.join(self.export_dir, filename)
        
        # Montar estrutura completa solicitada
        rois_data = []
        for roi in self.rois:
            # Encontrar série da ROI se possível (no momento o viewer assume a série atual para as ROIs mostradas)
            # Mas vamos pegar os dados salvos no objeto ROI
            s_uid = roi.get('series_uid')
            
            # Encontrar metadados da série (se não for a atual, teríamos que carregar, mas para o MVP
            # vamos assumir que as ROIs confirmadas têm os dados geométricos já calculados)
            
            roi_entry = {
                "id": roi['id'],
                "case_id": case_name,
                "series_uid": s_uid,
                "orientation": "UNKNOWN", # Fallback
                "slice_index_k": roi['center_voxel'][2],
                "center_ijk": roi['center_voxel'],
                "center_xyz_mm": roi['center_mm'],
                "radius_mm": roi['radius_mm'],
                "timestamp_iso": datetime.now().isoformat(),
                "image_geometry": {
                    "spacing_xyz": list(self.meta['spacing']),
                    "origin_xyz": list(self.meta['origin']),
                    "direction": list(self.meta['direction']),
                    "shape_ijk": list(self.meta['size'])
                }
            }
            
            # Tentar achar a orientação na lista de séries
            for s in self.series_list:
                if s['series_uid'] == s_uid:
                    roi_entry["orientation"] = s['orientation'].upper()
                    roi_entry["series_name"] = s['series_name']
                    break
            
            rois_data.append(roi_entry)

        data = {
            "app_version": "1.0.0-MVP",
            "data_root": self.input_root,
            "case_name": case_name,
            "rois": rois_data
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, sort_keys=True)
            self.last_message = f"JSON exportado: {filename}"
            print(f"[INFO] Export completo salvo em: {output_path}")
        except Exception as e:
            self.last_message = f"ERRO ao salvar JSON: {str(e)[:20]}..."
            print(f"[ERROR] Falha no export JSON: {e}")
        
        if self.fig: self.update_plot()

def main():
    parser = argparse.ArgumentParser(description="ARARAT Viewer MVP - ProstateX")
    parser.add_argument("--dicom_dir", help="Diretorio de um caso especifico (Single-case mode)")
    parser.add_argument("--data_root", "--samples_root", help="Diretorio raiz contendo varios casos (Samples mode)")
    parser.add_argument("--series_hint", default="t2tsetra", help="Hint para encontrar a serie (default: t2tsetra)")
    
    args = parser.parse_args()
    
    app = ViewerApp(dicom_dir=args.dicom_dir, data_root=args.data_root, series_hint=args.series_hint)
    app.run()

if __name__ == "__main__":
    main()
