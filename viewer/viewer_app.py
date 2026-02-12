import sys
import os
import argparse
import json
import csv
import traceback
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, Ellipse, Rectangle

# adicionar raiz do projeto ao path para importar shared
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from shared import dicom_io
except ImportError as e:
    print(f"Erro ao importar shared.dicom_io: {e}")
    sys.exit(1)

class ViewerApp:
    def __init__(self, dicom_dir=None, data_root=None, series_hint="t2tsetra"):
        self.input_root = os.path.abspath(data_root) if data_root else (os.path.abspath(dicom_dir) if dicom_dir else None)
        self.dicom_root = self.input_root
        self.series_hint = series_hint
        
        # Cache de Séries (LRU Simples)
        self._series_cache = {} # {(case_name, series_idx): (sitk_img, np_vol, meta)}
        self._cache_order = []
        self._max_cache_size = 6

        # Dados do Workspace (Cases)
        self.cases_list = []
        self.current_case_idx = -1
        self.is_samples_mode = False
        
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
        
        self.current_slice = 0
        self.max_slice = 0
        
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
        
        self.rois = [] # ROIs confirmadas
        self.lesion_counter = 1
        self.last_message = "Pronto"
        self.show_help = False
        
        self.fig = None
        self.ax = None
        self.ax_left = None
        self.ax_info = None
        
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
            print(f"[WARNING] Caminho invalido: {self.input_root}")
            self.cases_list = []
            return

        print(f"Analisando workspace: {self.input_root}")
        
        # Verificar se é um caso único (tem DICOMs ou subpastas de série)
        series_in_root = dicom_io.list_case_series(self.input_root)
        
        if series_in_root:
            print("[INFO] Modo SINGLE CASE detectado.")
            self.is_samples_mode = False
            self.cases_list = [os.path.basename(self.input_root)]
            self.current_case_idx = 0
        else:
            # Tentar encontrar casos (pastas que contenham séries)
            subdirs = sorted([d for d in os.listdir(self.input_root) 
                             if os.path.isdir(os.path.join(self.input_root, d))])
            
            valid_cases = []
            for d in subdirs:
                case_path = os.path.join(self.input_root, d)
                if dicom_io.list_case_series(case_path):
                    valid_cases.append(d)
            
            if valid_cases:
                print(f"[INFO] Modo SAMPLES detectado. {len(valid_cases)} casos validos encontrados.")
                self.is_samples_mode = True
                self.cases_list = valid_cases
                self.current_case_idx = 0
            else:
                print(f"[WARNING] Nenhun caso valido encontrado em {self.input_root}")
                self.cases_list = []
                self.is_samples_mode = False

    def load_case(self, case_idx):
        """Troca o caso ativo (hot-swap)."""
        if not (0 <= case_idx < len(self.cases_list)):
            return False
            
        new_case_name = self.cases_list[case_idx]
        print(f"\n[HOT-SWAP] Trocando para caso: {new_case_name}")
        
        # Atualizar dicom_root
        if self.is_samples_mode:
            self.dicom_root = os.path.join(self.input_root, new_case_name)
        
        self.current_case_idx = case_idx
        
        # Reset de Estado Robusto
        self.series_list = []
        self.current_series_idx = 0
        self.series_page = 0
        self.current_slice = 0
        self.rois = [] # Limpar ROIs ao trocar de caso no MVP
        self.lesion_counter = 1
        self.candidate_center = None
        self.is_locked = False
        self.mode = "NORMAL"
        self.case_input_str = ""
        self.series_input_str = ""
        self.last_message = f"Caso {new_case_name} carregado."
        
        # Recarregar séries do novo caso
        self.discover_series()
        self.load_current_series()
        
        # Se UI já existir, atualizar
        if self.fig:
            self.update_plot()
        return True

    def discover_series(self):
        print(f"Buscando series em: {self.dicom_root}...")
        self.series_list = dicom_io.list_case_series(self.dicom_root)
        if not self.series_list:
            print(f"\n[ERRO] Nenhuma serie DICOM valida encontrada em {self.dicom_root}")
            sys.exit(1)
        
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

    def load_current_series(self, center_mm=None):
        case_name = self.cases_list[self.current_case_idx]
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
        
        # Posicionamento do Slice
        if center_mm:
            _, _, k = dicom_io.mm_to_voxel(center_mm[0], center_mm[1], center_mm[2], self.meta)
            self.current_slice = int(round(max(0, min(k, self.max_slice))))
        else:
            self.current_slice = min(self.current_slice, self.max_slice)
            if self.current_slice < 0:
                self.current_slice = self.max_slice // 2
        
        self.candidate_center = None
        self.is_locked = False
        self.last_message = "Pronto"
        if self.fig: self.update_plot()

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

        # Usar gridspec para criar um layout com 3 colunas: Sidebar Left, Main Image, Info Right
        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1.2, 4, 1.2])
        
        self.ax_left = self.fig.add_subplot(gs[0])
        self.ax = self.fig.add_subplot(gs[1])
        self.ax_info = self.fig.add_subplot(gs[2])
        
        self.ax_left.axis('off')
        self.ax_info.axis('off') 

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

        self.update_plot()
        
        # conectar eventos
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('draw_event', self.on_draw)
        
        # Ajustar margens para o novo layout com sidebar
        plt.subplots_adjust(top=0.92, bottom=0.05, left=0.02, right=0.98, wspace=0.1)
        
        plt.show()

    def on_draw(self, event):
        """Captura o background para blitting após o primeiro draw."""
        if event is not None and event.canvas != self.fig.canvas:
            return
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self._draw_preview_fast()

    def _draw_preview_fast(self):
        """Desenha o preview da ROI usando blitting e artists persistentes."""
        if self.background is None or self.ax is None:
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
        """Desenha a interseção da esfera ROI com o slice atual, suportando multi-planos."""
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
            # Raio da seção circular no slice atual (em mm) usando Pitágoras
            r_slice_mm = (radius_mm**2 - dz_mm**2)**0.5
            
            # Converter para pixels (elipse se spacing x/y for diferente)
            r_px_x = r_slice_mm / self.meta['spacing'][0]
            r_px_y = r_slice_mm / self.meta['spacing'][1]
            
            # Desenhar crosshair (centro da esfera projetado)
            self.ax.plot(ci, cj, marker='+', color=color, markersize=10, alpha=alpha)
            
            # Desenhar elipse (interseção da esfera com o plano)
            ellipse = Ellipse((ci, cj), width=r_px_x*2, height=r_px_y*2, 
                             fill=False, color=color, linestyle=linestyle, alpha=alpha, linewidth=2)
            self.ax.add_patch(ellipse)
            
            if label:
                self.ax.text(ci+2, cj+2, label, color=color, fontweight='bold', fontsize=8)

    def _get_help_text(self):
        """Gera o texto do help com alinhamento perfeito."""
        def fmt_line(cmd, desc, col=18):
            return f"{cmd.ljust(col)} : {desc}"

        lines = [
            "=== COMANDOS DO SISTEMA ===",
            "",
            "NAVEGACAO",
            fmt_line("Scroll/Arrows", "trocar slice"),
            fmt_line("[ / ]", "paginar series"),
            fmt_line("1..9", "atalho serie"),
            fmt_line("Ctrl + G", "ir para serie N"),
            fmt_line("C + num + Enter", "ir para case N"),
            fmt_line("A / K / G", "T2 Axial/Cor/Sag"),
            "",
            "ROI (LESAO)",
            fmt_line("Clique esq.", "travar centro"),
            fmt_line("Enter / D", "confirmar ROI"),
            fmt_line("X", "limpar selecao"),
            fmt_line("+ / -", "ajustar raio"),
            fmt_line("Del", "deletar ultima"),
            "",
            "GERAL",
            fmt_line("Ctrl + S / J", "salvar JSON"),
            fmt_line("O", "abrir data_root"),
            fmt_line("H", "mostrar/ocultar help"),
            fmt_line("Q", "sair")
        ]
        return "\n".join(lines)

    def update_plot(self):
        # 1. Limpar eixos
        self.ax.clear()
        self.ax_info.clear()
        self.ax_left.clear()
        self.ax_info.axis('off')
        self.ax_left.axis('off')
        
        # Resetar artists persistentes pois o ax.clear() os removeu
        self._persistent_artists = {'line': None, 'ellipse': None, 'text': None}
        
        # 2. Desenhar imagem
        slice_img = self.np_vol[self.current_slice, :, :]
        height, width = slice_img.shape
        self.ax.imshow(slice_img, cmap='gray', extent=[0, width, height, 0])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.axis('off') 
        
        # 3. HUD (Acima da Imagem)
        s = self.series_list[self.current_series_idx]
        case_name = self.cases_list[self.current_case_idx]
        
        if self.mode == "SERIES_SELECT":
            mode_str = f"GO TO SERIES: {self.series_input_str}_ (Enter confirm, Esc cancel)"
        elif self.mode == "CASE_SELECT":
            mode_str = f"GO TO CASE: {self.case_input_str}_ (Enter confirm, Esc cancel)"
        else:
            mode_str = "LOCKED" if self.is_locked else "PREVIEW"
            
        line1 = f"CASE: {case_name} | SERIES: {s['series_name'][:20]} ({s['orientation'].upper()})"
        line2 = f"SLICE: {self.current_slice}/{self.max_slice} | R: {self.radius_mm:.1f} mm | MODE: {mode_str}"
        
        hud_text = f"{line1}\n{line2}"
        if self.last_message:
            hud_text += f"\nLAST: {self.last_message}"

        self.ax.text(0.01, 1.01, hud_text, transform=self.ax.transAxes, 
                     verticalalignment='bottom', horizontalalignment='left',
                     family='monospace', fontsize=8, color='white', fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.4'))

        # 4. Sidebar Esquerda (Logo + Help)
        if self.logo_img is not None:
            # Desenhar logo no topo da sidebar esquerda
            logo_inset = self.ax_left.inset_axes([0.1, 0.85, 0.8, 0.12])
            logo_inset.imshow(self.logo_img)
            logo_inset.axis('off')

        if self.show_help:
            # Bloco único de help na sidebar esquerda
            help_txt = self._get_help_text()
            self.ax_left.text(0.05, 0.82, help_txt, transform=self.ax_left.transAxes,
                             va="top", ha="left", family='monospace', fontsize=9, color='white',
                             bbox=dict(facecolor=(0,0,0,0.75), edgecolor="yellow", linewidth=1, boxstyle="round,pad=0.6"))
        else:
            self.ax_left.text(0.5, 0.82, "Pressione 'H'\npara ajuda", transform=self.ax_left.transAxes,
                             va="top", ha="center", fontsize=9, color='gray', family='monospace', 
                             fontweight='bold')

        # 5. Painel Direito (Cases, Series & ROIs)
        total_series_pages = (len(self.series_list) - 1) // self.series_per_page + 1
        
        info_panel_text = ""
        
        # Seção CASES
        if self.is_samples_mode:
            info_panel_text += "=== CASES ===\n"
            for i, c in enumerate(self.cases_list):
                mark = ">" if i == self.current_case_idx else " "
                info_panel_text += f"{mark}[{i+1}] {c[:15]}\n"
            info_panel_text += "\n"

        # Seção T2 QUICK
        info_panel_text += "=== T2 QUICK ===\n"
        for orient in ['axial', 'coronal', 'sagittal']:
            idx = self.t2_quick[orient]
            mark = ">" if idx == self.current_series_idx and idx is not None else " "
            # Mapeamento de teclas: A para axial, K para coronal, G para sagittal
            key_map = {'axial': 'A', 'coronal': 'K', 'sagittal': 'G'}
            key = key_map[orient]
            if idx is not None:
                name = self.series_list[idx]['series_name'][:12]
                info_panel_text += f"{mark}({key}) {orient[:3].upper()}: {name}\n"
            else:
                info_panel_text += f"   ({key}) {orient[:3].upper()}: -\n"
        info_panel_text += "\n"

        # Seção SERIES
        info_panel_text += f"=== SERIES ({self.series_page+1}/{total_series_pages}) ===\n"
        start_idx = self.series_page * self.series_per_page
        end_idx = min(start_idx + self.series_per_page, len(self.series_list))
        
        for i in range(start_idx, end_idx):
            ser = self.series_list[i]
            mark = ">" if i == self.current_series_idx else " "
            info_panel_text += f"{mark}[{i+1}] {ser['series_name'][:12]} ({ser['orientation'][0].upper()})\n"
        
        info_panel_text += "\nUse [ ] to page\n"
        
        # Seção ROIs
        info_panel_text += "\n=== ROIs ===\n"
        if not self.rois:
            info_panel_text += "Nenhuma ROI.\n"
        else:
            for roi in self.rois:
                i, j, k = roi['center_voxel']
                # Tentar encontrar o plano da serie original da ROI
                info_panel_text += (
                    f"#{roi['id']} | S:{k} | R:{roi['radius_mm']:.1f}\n"
                    f"  Pos:({int(i)},{int(j)})\n"
                )
        
        self.ax_info.text(0.05, 0.98, info_panel_text, transform=self.ax_info.transAxes,
                         verticalalignment='top', family='monospace', fontsize=8)

        # 6. Desenhar ROIs confirmadas (Renderização 3D Tri-planar)
        for roi in self.rois:
            self._draw_roi_sphere(
                roi['center_voxel'], 
                roi['radius_mm'], 
                color='lime', 
                label=roi['id'],
                linestyle='-',
                alpha=0.8,
                roi_mm=roi['center_mm']
            )

        self.fig.canvas.draw_idle()
        # Preview será desenhado via blitting no on_draw ou manualmente aqui se background já existe
        if self.background:
            self._draw_preview_fast()

    def on_mouse_move(self, event):
        if self.mode in ["SERIES_SELECT", "CASE_SELECT"]:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        
        self.mouse_pos = (event.xdata, event.ydata)
        
        if not self.is_locked:
            self._draw_preview_fast()

    def on_scroll(self, event):
        if self.mode in ["SERIES_SELECT", "CASE_SELECT"]:
            return
        if event.button == 'up':
            self.current_slice = min(self.current_slice + 1, self.max_slice)
        elif event.button == 'down':
            self.current_slice = max(self.current_slice - 1, 0)
        self.update_plot()

    def on_click(self, event):
        if self.mode in ["SERIES_SELECT", "CASE_SELECT"]:
            return
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
            
        self.candidate_center = [event.xdata, event.ydata, self.current_slice]
        self.is_locked = True
        self.last_message = "Centro travado. Pressione Enter para confirmar."
        self.update_plot()

    def on_key(self, event):
        try:
            if event.key == 'q':
                plt.close()
                return

            # Detectar Ctrl
            is_ctrl = event.key.startswith('ctrl+')

            # Atalhos de Navegação T2 Quick (A, K, G)
            if self.mode == "NORMAL" and not is_ctrl and event.key in ['a', 'k', 'g']:
                target = None
                if event.key == 'a': target = 'axial'
                elif event.key == 'k': target = 'coronal' # K de Coronal (C conflita com Case)
                elif event.key == 'g': target = 'sagittal' # G de SaGittal (S conflita com Save)
                
                if target:
                    if self.t2_quick[target] is not None:
                        last_center_mm = self.rois[-1]['center_mm'] if self.rois else None
                        self.current_series_idx = self.t2_quick[target]
                        self.load_current_series(center_mm=last_center_mm)
                        self.series_page = self.current_series_idx // self.series_per_page
                        self.last_message = f"Trocado para T2 {target.upper()}"
                    else:
                        self.last_message = f"T2 {target.upper()} nao disponivel neste case."
                    self.update_plot()
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
            elif event.key in ['up', 'right']:
                self.current_slice = min(self.current_slice + 1, self.max_slice)
                self.update_plot()
            elif event.key in ['down', 'left']:
                self.current_slice = max(self.current_slice - 1, 0)
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
        self.last_message = f"ROI L{self.lesion_counter} confirmada!"
        
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

    def open_data_root(self):
        """Abre seletor de pastas para mudar o data_root."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            new_root = filedialog.askdirectory(title="Selecione a pasta raiz de casos (data_root)")
            root.destroy()
            
            if new_root:
                self.input_root = os.path.abspath(new_root)
                self.dicom_root = self.input_root
                self.discover_workspace()
                if self.is_samples_mode:
                    self.load_case(0)
                else:
                    self.discover_series()
                    self.load_current_series()
                self.last_message = f"Nova raiz: {os.path.basename(self.input_root)}"
                if self.fig: self.update_plot()
        except Exception as e:
            print(f"[WARNING] Erro ao abrir seletor de pastas: {e}")
            self.last_message = "Erro ao abrir seletor de pastas."
            if self.fig: self.update_plot()

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
