import os
import json
from datetime import datetime

def save_roi_json(output_path, case_id, rois, data_root, app_version="1.0.0-MVP"):
    """
    Exporta ROIs para um JSON estruturado.
    
    Args:
        output_path (str): Caminho completo para o arquivo .json
        case_id (str): ID/Nome do caso
        rois (list): Lista de dicionários de ROI do ViewerApp
        data_root (str): Raiz dos dados DICOM
        app_version (str): Versão da aplicação
    """
    rois_data = []
    for roi in rois:
        roi_entry = {
            "id": roi['id'],
            "case_id": case_id,
            "series_instance_uid": roi.get('series_uid', 'UNKNOWN'),
            "center_xyz_mm": roi['center_mm'],
            "radius_mm": roi['radius_mm'],
            "timestamp_iso": datetime.now().isoformat()
        }
        rois_data.append(roi_entry)

    data = {
        "app_version": app_version,
        "export_timestamp": datetime.now().isoformat(),
        "data_root": data_root,
        "case_id": case_id,
        "rois": rois_data
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    
    return output_path
