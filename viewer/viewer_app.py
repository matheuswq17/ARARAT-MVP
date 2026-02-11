import sys
import os
import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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
        self.dicom_root = os.path.abspath(dicom_dir)
        self.series_hint = series_hint
        self.sitk_img = None
        self.np_vol = None
        self.meta = None
        
        self.current_slice = 0
        self.max_slice = 0
        
        self.current_selection = None # (i, j)
        self.radius_mm = 5.0
        
        self.rois = []
        self.lesion_counter = 1
        self.last_message = "Pronto"
        
        self.fig = None
        self.ax = None
        
        # carregar dados
        self.load_data()

    def load_data(self):
        print(f"Buscando serie '{self.series_hint}' em: {self.dicom_root}...")
        try:
            self.sitk_img, self.np_vol, self.meta = dicom_io.load_dicom_series(
                self.dicom_root, self.series_hint
            )
            self.max_slice = self.np_vol.shape[0] - 1
            self.current_slice = self.max_slice // 2
            
            print(f"\n[INFO] Serie carregada com sucesso!")
            print(f"Diretorio: {self.meta['series_dir']}")
            print(f"ID Serie:  {self.meta['series_id']}")
            print(f"Slices:    {self.meta['n_files']}")
            print(f"Shape:     {self.np_vol.shape}")
            
        except Exception as e:
            print(f"\n[ERRO] Falha ao carregar DICOM: {e}")
            sys.exit(1)

    def run(self):
        # Janela única com HUD sobreposto
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title(f"ARARAT Viewer - {os.path.basename(self.meta['series_dir'])}")
        
        self.update_plot()
        
        # conectar eventos
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        plt.tight_layout()
        plt.show()

    def update_plot(self):
        self.ax.clear()
        
        slice_img = self.np_vol[self.current_slice, :, :]
        self.ax.imshow(slice_img, cmap='gray')
        self.ax.axis('off') # Esconder eixos para focar na imagem
        
        # --- HUD / Overlay ---
        hud_text = (
            f"Slice: {self.current_slice}/{self.max_slice}\n"
            f"Radius: {self.radius_mm:.1f} mm | ROIs: {len(self.rois)}\n"
            "Scroll/Arrows: nav | Click: center | +/-: radius | a: add | s: save | q: quit"
        )
        if self.last_message:
            hud_text += f"\nLast: {self.last_message}"

        # Adicionar o HUD no topo com caixa semi-transparente
        self.ax.text(0.02, 0.98, hud_text, transform=self.ax.transAxes, 
                     verticalalignment='top', family='monospace', fontsize=10,
                     color='white', fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.5'))

        # desenhar selecao atual
        if self.current_selection:
            ci, cj = self.current_selection
            self.ax.plot(ci, cj, 'r+', markersize=12)
            
            # raio em pixels (aproximado pelo spacing x)
            radius_px = self.radius_mm / self.meta['spacing'][0]
            circ = Circle((ci, cj), radius_px, fill=False, color='red', linewidth=2)
            self.ax.add_patch(circ)

        # desenhar ROIs salvas
        for roi in self.rois:
            if roi['center_ijk'][2] == self.current_slice:
                ri, rj, _ = roi['center_ijk']
                self.ax.plot(ri, rj, 'gx', markersize=8)
                self.ax.text(ri+2, rj+2, f"L{roi['lesion_id']}", color='lime', fontweight='bold')
                
                r_px = roi['radius_mm'] / self.meta['spacing'][0]
                c = Circle((ri, rj), r_px, fill=False, color='lime', linestyle='--', alpha=0.6)
                self.ax.add_patch(c)

        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.button == 'up':
            self.current_slice = min(self.current_slice + 1, self.max_slice)
        elif event.button == 'down':
            self.current_slice = max(self.current_slice - 1, 0)
        self.last_message = f"Slice alterado para {self.current_slice}"
        self.update_plot()

    def on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
            
        self.current_selection = (event.xdata, event.ydata)
        self.last_message = f"Centro marcado: ({event.xdata:.1f}, {event.ydata:.1f})"
        self.update_plot()

    def on_key(self, event):
        if event.key == 'q':
            plt.close()
        elif event.key in ['up', 'right']:
            self.current_slice = min(self.current_slice + 1, self.max_slice)
            self.last_message = f"Slice: {self.current_slice}"
            self.update_plot()
        elif event.key in ['down', 'left']:
            self.current_slice = max(self.current_slice - 1, 0)
            self.last_message = f"Slice: {self.current_slice}"
            self.update_plot()
        elif event.key in ['+', '=']:
            self.radius_mm += 0.5
            self.last_message = f"Raio: {self.radius_mm:.1f} mm"
            self.update_plot()
        elif event.key in ['-', '_']:
            self.radius_mm = max(0.5, self.radius_mm - 0.5)
            self.last_message = f"Raio: {self.radius_mm:.1f} mm"
            self.update_plot()
        elif event.key == 'a':
            self.add_roi()
        elif event.key == 's':
            self.save_json()

    def add_roi(self):
        if not self.current_selection:
            self.last_message = "AVISO: Selecione um ponto primeiro!"
            self.update_plot()
            return
            
        i, j = self.current_selection
        k = self.current_slice
        
        # converter para MM
        x, y, z = dicom_io.voxel_to_mm(i, j, k, self.meta)
        
        roi = {
            "lesion_id": self.lesion_counter,
            "center_ijk": [int(round(i)), int(round(j)), int(k)],
            "center_mm": [float(x), float(y), float(z)],
            "radius_mm": float(self.radius_mm)
        }
        
        self.rois.append(roi)
        self.last_message = f"ROI L{self.lesion_counter} adicionada!"
        self.lesion_counter += 1
        self.current_selection = None
        self.update_plot()

    def save_json(self):
        output_path = os.path.join(os.path.dirname(__file__), "roi_selection.json")
        
        data = {
            "dicom_root": self.dicom_root,
            "series_dir": self.meta['series_dir'],
            "series_id": self.meta['series_id'],
            "rois": self.rois
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.last_message = f"JSON salvo: {os.path.basename(output_path)}"
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
