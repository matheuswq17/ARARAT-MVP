import os
import numpy as np
import SimpleITK as sitk

def export_roi_masks(output_dir, reference_sitk_img, rois, case_id):
    """
    Gera máscaras NIfTI 3D para cada ROI, alinhadas à imagem de referência.
    
    Args:
        output_dir (str): Diretório onde as máscaras serão salvas
        reference_sitk_img (sitk.Image): Imagem SimpleITK de referência (geometria)
        rois (list): Lista de ROIs confirmadas
        case_id (str): ID do caso para nomeação
        
    Returns:
        list: Caminhos das máscaras geradas
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_paths = []
    
    # Obter dados da geometria da referência
    spacing = reference_sitk_img.GetSpacing()
    origin = reference_sitk_img.GetOrigin()
    direction = reference_sitk_img.GetDirection()
    size = reference_sitk_img.GetSize()
    
    # Criar um grid de coordenadas físicas para o volume (operação pesada se for muito grande)
    # Alternativa: Iterar sobre os voxels e converter para mm
    
    # Vamos criar uma máscara vazia (numpy)
    # Note: SimpleITK usa (x, y, z), numpy usa (z, y, x)
    mask_np = np.zeros(sitk.GetArrayFromImage(reference_sitk_img).shape, dtype=np.uint8)
    
    for roi in rois:
        roi_id = roi['id']
        center_mm = np.array(roi['center_mm'])
        radius_mm = roi['radius_mm']
        
        # Otimização: Apenas iterar no bounding box da ROI em voxels
        # Converter centro mm para voxel na imagem de referência
        center_voxel = reference_sitk_img.TransformPhysicalPointToContinuousIndex(center_mm)
        
        # Estimar BB em voxels
        pad_vox = [int(np.ceil(radius_mm / s)) for s in spacing]
        
        z_start = max(0, int(round(center_voxel[2] - pad_vox[2])))
        z_end = min(mask_np.shape[0], int(round(center_voxel[2] + pad_vox[2] + 1)))
        
        y_start = max(0, int(round(center_voxel[1] - pad_vox[1])))
        y_end = min(mask_np.shape[1], int(round(center_voxel[1] + pad_vox[1] + 1)))
        
        x_start = max(0, int(round(center_voxel[0] - pad_vox[0])))
        x_end = min(mask_np.shape[2], int(round(center_voxel[0] + pad_vox[0] + 1)))
        
        # Criar máscara individual para esta ROI
        roi_mask_np = np.zeros_like(mask_np)
        
        # Iterar apenas no sub-volume
        for z in range(z_start, z_end):
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    # Obter coordenada física do voxel (x, y, z)
                    phys_pt = reference_sitk_img.TransformIndexToPhysicalPoint((x, y, z))
                    dist = np.linalg.norm(np.array(phys_pt) - center_mm)
                    if dist <= radius_mm:
                        roi_mask_np[z, y, x] = 1
        
        # Salvar NIfTI para esta ROI
        roi_mask_sitk = sitk.GetImageFromArray(roi_mask_np)
        roi_mask_sitk.CopyInformation(reference_sitk_img)
        
        filename = f"mask_{roi_id}.nii.gz"
        filepath = os.path.join(output_dir, filename)
        sitk.WriteImage(roi_mask_sitk, filepath)
        generated_paths.append(filepath)
        
    return generated_paths
