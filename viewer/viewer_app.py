import sys
import os
import argparse
import json
import traceback
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

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
    def __init__(self, dicom_dir, series_hint="t2tsetra"):
        self.input_root = os.path.abspath(dicom_dir)
        self.dicom_root = self.input_root
        self.series_hint = series_hint
        
        # Dados do Workspace (Cases)
        self.cases_list = []
        self.current_case_idx = -1
        self.is_samples_mode = False
        
        # Dados do Case Atual
        self.series_list = []
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
        
        self.fig = None
        self.ax = None
        
        # Inicialização
        self.discover_workspace()
        if self.is_samples_mode:
            self.load_case(0)
        else:
            self.discover_series()
            self.load_current_series()

    def discover_workspace(self):
        """Detecta se dicom_dir é uma raiz de casos ou um caso específico."""
        print(f"Analisando workspace: {self.input_root}")
        subdirs = [d for d in os.listdir(self.input_root) if os.path.isdir(os.path.join(self.input_root, d))]
        
        # Heurística: se tem subpastas "caseX", é modo SAMPLES
        cases = sorted([d for d in subdirs if d.lower().startswith("case")])
        if cases:
            print(f"[INFO] Modo SAMPLES detectado. {len(cases)} casos encontrados.")
            self.is_samples_mode = True
            self.cases_list = cases
        else:
            print("[INFO] Modo SINGLE CASE detectado.")
            self.is_samples_mode = False
            self.cases_list = [os.path.basename(self.input_root)]
            self.current_case_idx = 0

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
        
        # Escolher serie default
        # 1) T2 axial (hint)
        for idx, s in enumerate(self.series_list):
            if self.series_hint.lower() in s['series_name'].lower() and s['orientation'] == 'axial':
                self.current_series_idx = idx
                return
        
        # 2) Qualquer T2
        for idx, s in enumerate(self.series_list):
            if s['is_t2']:
                self.current_series_idx = idx
                return
        
        # 3) Serie com mais slices
        self.current_series_idx = self.series_list.index(max(self.series_list, key=lambda x: x['num_slices']))

    def load_current_series(self):
        s = self.series_list[self.current_series_idx]
        print(f"\n[INFO] Carregando serie: {s['series_name']} ({s['orientation']})")
        try:
            self.sitk_img, self.np_vol, self.meta = dicom_io.load_dicom_series_by_path(
                s['series_dir'], s['series_uid']
            )
            self.max_slice = self.np_vol.shape[0] - 1
            # Tentar preservar o slice atual se estiver no limite da nova serie
            self.current_slice = min(self.current_slice, self.max_slice)
            if self.current_slice < 0:
                self.current_slice = self.max_slice // 2
            
            # Resetar estado de ROI candidata ao trocar de serie
            self.candidate_center = None
            self.is_locked = False
            
        except Exception as e:
            print(f"\n[ERRO] Falha ao carregar serie {s['series_name']}: {e}")
            traceback.print_exc()
            sys.exit(1)

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

        # Usar gridspec para criar um painel lateral
        self.fig = plt.figure(figsize=(14, 10))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[4, 1.2])
        
        self.ax = self.fig.add_subplot(gs[0])
        self.ax_info = self.fig.add_subplot(gs[1])
        self.ax_info.axis('off') 

        # Desativar toolbar para evitar pan/zoom acidental
        self.fig.canvas.toolbar.pack_forget()

        self.fig.canvas.manager.set_window_title(f"ARARAT Viewer - {os.path.basename(self.dicom_root)}")
        
        self.update_plot()
        
        # conectar eventos
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('draw_event', self.on_draw)
        
        plt.tight_layout()
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

    def _draw_roi_sphere(self, center_ijk, radius_mm, color, label=None, linestyle='-', alpha=1.0):
        """Desenha a interseção da esfera ROI com o slice atual."""
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

    def update_plot(self):
        # 1. Limpar eixos
        self.ax.clear()
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # Resetar artists persistentes pois o ax.clear() os removeu
        self._persistent_artists = {'line': None, 'ellipse': None, 'text': None}
        
        # 2. Desenhar imagem
        slice_img = self.np_vol[self.current_slice, :, :]
        height, width = slice_img.shape
        self.ax.imshow(slice_img, cmap='gray', extent=[0, width, height, 0])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.axis('off') 
        
        # 3. HUD (Top-Left)
        s = self.series_list[self.current_series_idx]
        case_name = self.cases_list[self.current_case_idx]
        
        if self.mode == "SERIES_SELECT":
            mode_str = f"GO TO SERIES: {self.series_input_str}_ (Enter to confirm, Esc to cancel)"
        elif self.mode == "CASE_SELECT":
            mode_str = f"GO TO CASE: {self.case_input_str}_ (Enter to confirm, Esc to cancel)"
        else:
            mode_str = "LOCKED (Enter/a to confirm)" if self.is_locked else "PREVIEW (Click to lock)"
            
        hud_text = (
            f"CASE: {case_name} | SERIE: {s['series_name']} [{s['orientation'].upper()}]\n"
            f"SLICE: {self.current_slice}/{self.max_slice} | R: {self.radius_mm:.1f} mm\n"
            f"MODE: {mode_str}\n"
            "nav: scroll/arrows | c: go-to case | g: go-to series | confirm: Enter/a | clear: x | radius: +/- | save: s | quit: q"
        )
        if self.last_message:
            hud_text += f"\nLAST: {self.last_message}"

        self.ax.text(0.01, 0.99, hud_text, transform=self.ax.transAxes, 
                     verticalalignment='top', horizontalalignment='left',
                     family='monospace', fontsize=9, color='white', fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))

        # 4. Painel Lateral (Cases, Series & ROIs)
        total_series_pages = (len(self.series_list) - 1) // self.series_per_page + 1
        
        info_panel_text = ""
        
        # Se estiver no modo SAMPLES, mostrar lista de casos
        if self.is_samples_mode:
            info_panel_text += "=== CASES ===\n"
            # Mostrar os casos ao redor do atual (limitar para não estourar painel)
            for i, c in enumerate(self.cases_list):
                mark = ">" if i == self.current_case_idx else " "
                info_panel_text += f"{mark}[{i+1}] {c}\n"
            info_panel_text += "\n"

        info_panel_text += f"=== SERIES (Pg {self.series_page+1}/{total_series_pages}) ===\n"
        start_idx = self.series_page * self.series_per_page
        end_idx = min(start_idx + self.series_per_page, len(self.series_list))
        
        for i in range(start_idx, end_idx):
            ser = self.series_list[i]
            mark = ">" if i == self.current_series_idx else " "
            info_panel_text += f"{mark}[{i+1}] {ser['series_name'][:12]} ({ser['orientation'][0].upper()})\n"
        
        info_panel_text += "\nUse [ ] to page\n"
        info_panel_text += "\n=== ROIs ===\n"
        if not self.rois:
            info_panel_text += "Nenhuma ROI confirmada."
        else:
            for roi in self.rois:
                i, j, k = roi['center_voxel']
                info_panel_text += (
                    f"#{roi['id']} | S:{k} | R:{roi['radius_mm']:.1f}\n"
                    f"  Pos: ({i},{j})\n"
                )
        
        self.ax_info.text(0.05, 0.95, info_panel_text, transform=self.ax_info.transAxes,
                         verticalalignment='top', family='monospace', fontsize=8)

        # 6. Desenhar ROIs confirmadas (Renderização 3D)
        for roi in self.rois:
            if roi.get('series_uid') == s['series_uid']:
                self._draw_roi_sphere(
                    roi['center_voxel'], 
                    roi['radius_mm'], 
                    color='lime', 
                    label=roi['id'],
                    linestyle='-',
                    alpha=0.8
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
            if event.key == 'c':
                print("[DEBUG] Enter CASE_SELECT")
                self.mode = "CASE_SELECT"
                self.case_input_str = ""
                self.update_plot()
            elif event.key == 'g':
                print("[DEBUG] Enter SERIES_SELECT")
                self.mode = "SERIES_SELECT"
                self.series_input_str = ""
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
            elif event.key in ['enter', 'a']:
                self.confirm_roi()
            elif event.key == 'd':
                self.delete_last_roi()
            elif event.key == 's':
                self.save_json()
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
        self.lesion_counter += 1
        self.is_locked = False
        self.candidate_center = None
        self.update_plot()

    def save_json(self):
        case_name = self.cases_list[self.current_case_idx]
        if self.is_samples_mode:
            filename = f"roi_selection_{case_name}.json"
        else:
            filename = "roi_selection.json"
            
        output_path = os.path.join(os.path.dirname(__file__), filename)
        s = self.series_list[self.current_series_idx]
        
        data = {
            "case_root": self.dicom_root,
            "case_name": case_name,
            "current_series": {
                "uid": s['series_uid'],
                "dir": s['series_dir'],
                "name": s['series_name'],
                "orientation": s['orientation']
            },
            "rois": self.rois
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.last_message = f"JSON salvo: {filename}"
        except Exception as e:
            self.last_message = f"ERRO ao salvar JSON: {str(e)[:20]}..."
        self.update_plot()

def main():
    parser = argparse.ArgumentParser(description="ARARAT Viewer MVP - ProstateX")
    parser.add_argument("--dicom_dir", required=True, help="Diretorio raiz do caso ou serie")
    parser.add_argument("--series_hint", default="t2tsetra", help="Hint para encontrar a serie (default: t2tsetra)")
    
    args = parser.parse_args()
    
    app = ViewerApp(args.dicom_dir, args.series_hint)
    app.run()

if __name__ == "__main__":
    main()
