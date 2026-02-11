import sys
import os
import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sys.stderr.write("\nERRO CRITICO: SimpleITK nao encontrado.\n")
    sys.stderr.write("Por favor, instale usando: pip install SimpleITK\n\n")
    sys.exit(1)

def resolve_series_dir(root_dir, series_hint="t2tsetra"):
    """
    Localiza a melhor pasta de serie DICOM dentro de root_dir.
    
    A) Se root_dir ja contem uma serie DICOM valida, usa ela.
    B) Caso contrario, busca pastas cujo nome contem series_hint (case-insensitive).
       Escolhe a que tiver mais arquivos DICOM.
    C) Fallback: primeira pasta com DICOM encontrada.
    
    Returns:
        tuple: (series_dir, series_id, n_files)
    """
    reader = sitk.ImageSeriesReader()
    
    def get_valid_series(d):
        try:
            ids = reader.GetGDCMSeriesIDs(d)
            if ids:
                # retorna o primeiro id e o numero de arquivos para esse id
                sid = ids[0]
                files = reader.GetGDCMSeriesFileNames(d, sid)
                return sid, len(files)
        except:
            pass
        return None, 0

    # caso a: root_dir ja e a serie
    sid, n_files = get_valid_series(root_dir)
    if sid:
        return root_dir, sid, n_files

    # caso b: buscar por series_hint
    candidates = []
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            if series_hint.lower() in d.lower():
                full_path = os.path.join(root, d)
                sid, n_files = get_valid_series(full_path)
                if sid:
                    candidates.append((full_path, sid, n_files))
    
    if candidates:
        # escolher o candidato com maior n_files
        best = max(candidates, key=lambda x: x[2])
        return best

    # caso c: fallback total
    print(f"[AVISO] Nenhuma serie contendo '{series_hint}' encontrada. Buscando qualquer serie DICOM...")
    for root, dirs, files in os.walk(root_dir):
        sid, n_files = get_valid_series(root)
        if sid:
            print(f"[AVISO] Usando serie encontrada em: {root}")
            return root, sid, n_files
            
    raise ValueError(f"Nenhuma serie DICOM valida encontrada em {root_dir}")

def load_dicom_series(root_dir, series_hint="t2tsetra"):
    """
    Carrega uma serie DICOM resolvendo o diretorio automaticamente.
    
    Returns:
        tuple: (sitk_img, np_vol, meta_dict)
    """
    series_dir, series_id, n_files = resolve_series_dir(root_dir, series_hint)
    
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(series_dir, series_id)
    reader.SetFileNames(dicom_names)
    
    sitk_img = reader.Execute()
    np_vol = sitk.GetArrayFromImage(sitk_img)
    
    meta_dict = {
        "origin": sitk_img.GetOrigin(),
        "spacing": sitk_img.GetSpacing(),
        "direction": sitk_img.GetDirection(),
        "size": sitk_img.GetSize(),
        "series_dir": series_dir,
        "series_id": series_id,
        "n_files": n_files
    }
    
    return sitk_img, np_vol, meta_dict

def voxel_to_mm(i, j, k, meta):
    """
    Converte voxel (i, j, k) -> mm (x, y, z)
    """
    origin = meta["origin"]
    spacing = meta["spacing"]
    direction = meta["direction"]
    
    vx = i * spacing[0]
    vy = j * spacing[1]
    vz = k * spacing[2]
    
    dx = direction[0]*vx + direction[1]*vy + direction[2]*vz
    dy = direction[3]*vx + direction[4]*vy + direction[5]*vz
    dz = direction[6]*vx + direction[7]*vy + direction[8]*vz
    
    return (origin[0] + dx, origin[1] + dy, origin[2] + dz)

def mm_to_voxel(x, y, z, meta):
    """
    Converte mm (x, y, z) -> voxel (i, j, k)
    """
    origin = meta["origin"]
    spacing = meta["spacing"]
    direction = meta["direction"]
    
    px = x - origin[0]
    py = y - origin[1]
    pz = z - origin[2]
    
    rx = direction[0]*px + direction[3]*py + direction[6]*pz
    ry = direction[1]*px + direction[4]*py + direction[7]*pz
    rz = direction[2]*px + direction[5]*py + direction[8]*pz
    
    i = int(round(rx / spacing[0]))
    j = int(round(ry / spacing[1]))
    k = int(round(rz / spacing[2]))
    
    return (i, j, k)
