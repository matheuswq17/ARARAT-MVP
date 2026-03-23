import argparse
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
print(f"[validate_gt] repo_root={repo_root}")
sys.path.insert(0, str(repo_root))
shared_path = repo_root / "shared" / "dicom_io.py"
print(f"[validate_gt] shared/dicom_io.py exists={shared_path.exists()}")

from shared import dicom_io
from viewer import gt_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--dicom_dir", required=True)
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    case_name = args.case
    dicom_dir = os.path.abspath(args.dicom_dir)

    print(f"data_root: {data_root}")
    print(f"case_name: {case_name}")
    try:
        patient_id = gt_labels.resolve_patient_id(case_name, data_root)
    except Exception as e:
        patient_id = None
        print(f"resolve_patient_id error: {e}")

    if not patient_id:
        print(f"patient_id resolvido: None (GT indisponivel: nao foi possivel mapear {case_name})")
        return

    print(f"patient_id resolvido: {patient_id}")

    gt_list = gt_labels.get_gt_for_case(patient_id, data_root)
    if not gt_list:
        print("Nenhuma lesao GT encontrada para este patient_id (labels nao encontradas).")
        return

    sitk_img, np_vol, meta = dicom_io.load_dicom_series(dicom_dir)
    sz_k, sz_j, sz_i = np_vol.shape

    print(f"Volume: shape={np_vol.shape} series_dir={meta.get('series_dir')}")

    for idx, lesion in enumerate(gt_list, 1):
        x, y, z = lesion["xyz_mm"]
        vi, vj, vk = dicom_io.mm_to_voxel(x, y, z, meta)
        vi_int = int(round(vi))
        vj_int = int(round(vj))
        vk_int = int(round(vk))
        in_bounds = (
            0 <= vk_int < sz_k and
            0 <= vi_int < sz_i and
            0 <= vj_int < sz_j
        )
        print("----")
        print(f"lesion_id: {lesion.get('lesion_id') or f'GT{idx}'}")
        print(f"GGG: {lesion.get('ggg')} ISUP: {lesion.get('isup')}")
        print(f"ClinSig: {lesion.get('clinsig')} zone: {lesion.get('zone')}")
        print(f"xyz_mm: ({x:.3f}, {y:.3f}, {z:.3f})")
        print(f"voxel: ({vi_int}, {vj_int}, {vk_int})")
        print(f"slice: {vk_int}")
        print(f"in_bounds: {in_bounds}")
        print(f"source: {lesion.get('source')}")


if __name__ == "__main__":
    main()
