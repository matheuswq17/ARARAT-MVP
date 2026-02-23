import os
import csv
import json
import re


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

_GT_CACHE = {}
_LABELS_LOADED = False
_LAST_DATA_ROOT_KEY = None


def _normalize_name(name):
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _normalize_patient_id(value):
    s = str(value).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return digits
    return s.lower()


def _safe_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        txt = str(value).strip()
        if not txt:
            return None
        txt = txt.replace(",", ".")
        return float(txt)
    except Exception:
        return None


def _parse_xyz_from_row(row, field_map):
    x_names = ["x", "worldx", "posx", "xmm"]
    y_names = ["y", "worldy", "posy", "ymm"]
    z_names = ["z", "worldz", "posz", "zmm"]

    x_val = None
    y_val = None
    z_val = None

    for key in x_names:
        if key in field_map:
            x_val = _safe_float(row.get(field_map[key]))
            break
    for key in y_names:
        if key in field_map:
            y_val = _safe_float(row.get(field_map[key]))
            break
    for key in z_names:
        if key in field_map:
            z_val = _safe_float(row.get(field_map[key]))
            break

    if x_val is not None and y_val is not None and z_val is not None:
        return (x_val, y_val, z_val)

    for norm, orig in field_map.items():
        v = row.get(orig)
        if not isinstance(v, str):
            continue
        txt = v.strip()
        if not txt:
            continue
        inner = txt.strip("()[] ")
        parts = re.split(r"[,\s;]+", inner)
        if len(parts) >= 3:
            xv = _safe_float(parts[0])
            yv = _safe_float(parts[1])
            zv = _safe_float(parts[2])
            if xv is not None and yv is not None and zv is not None:
                return (xv, yv, zv)

    return None


def _extract_patient_id(row, field_map):
    candidates = [
        "patientid",
        "patient_id",
        "patient",
        "prostatexid",
        "prostatex",
        "proxid",
        "case",
        "caseid",
    ]
    for key in candidates:
        if key in field_map:
            v = row.get(field_map[key])
            if v is not None and str(v).strip():
                return str(v).strip()
    return None


def _extract_lesion_id(row, field_map):
    candidates = ["finding", "findingid", "lesion", "lesionid", "fid", "roi", "roi_id", "id"]
    for key in candidates:
        if key in field_map:
            v = row.get(field_map[key])
            if v is not None and str(v).strip():
                return str(v).strip()
    return None


def _extract_grade(row, field_map):
    ggg = None
    isup = None

    ggg_candidates = ["ggg", "gradegroup", "gleasongradegroup"]
    isup_candidates = ["isup", "gg", "ggroup"]

    for key in ggg_candidates:
        if key in field_map:
            v = row.get(field_map[key])
            if v is not None and str(v).strip():
                ggg = str(v).strip()
                break

    for key in isup_candidates:
        if key in field_map:
            v = row.get(field_map[key])
            if v is not None and str(v).strip():
                isup = str(v).strip()
                break

    return ggg, isup


def _build_entry(row, field_map, source_path):
    patient_id = _extract_patient_id(row, field_map)
    if not patient_id:
        return None

    xyz = _parse_xyz_from_row(row, field_map)
    if not xyz:
        return None

    lesion_id = _extract_lesion_id(row, field_map)
    ggg, isup = _extract_grade(row, field_map)

    clinsig = None
    zone = None

    if "clinsig" in field_map:
        v = row.get(field_map["clinsig"])
        if v is not None and str(v).strip():
            clinsig = str(v).strip()

    if "zone" in field_map:
        v = row.get(field_map["zone"])
        if v is not None and str(v).strip():
            zone = str(v).strip()

    return {
        "patient_id": patient_id,
        "patient_key": _normalize_patient_id(patient_id),
        "lesion_id": lesion_id,
        "xyz_mm": xyz,
        "ggg": ggg,
        "isup": isup,
        "clinsig": clinsig,
        "zone": zone,
        "source": source_path,
        "row": row,
    }


def resolve_patient_id(case_name, data_root):
    if not case_name:
        return None
    s = str(case_name).strip()
    if s.lower().startswith("prostatex"):
        return s
    if data_root:
        base = os.path.abspath(data_root)
        map_path = os.path.join(base, "SAMPLES", "sample_case_map.json")
        if os.path.exists(map_path):
            try:
                with open(map_path, "r") as f:
                    mp = json.load(f)
                if isinstance(mp, dict):
                    v = mp.get(s)
                    if v:
                        return str(v).strip()
            except Exception:
                pass
    manifest = os.path.join(project_root, "exports", "roi_manifest.csv")
    if os.path.exists(manifest):
        try:
            with open(manifest, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                fmap = {_normalize_name(name): name for name in fieldnames}
                case_keys = ["casename", "case", "samplecase"]
                pid_keys = ["patientid", "patient_id", "patient", "prostatexid", "prostatex"]
                case_col = None
                pid_col = None
                for k in case_keys:
                    if k in fmap:
                        case_col = fmap[k]
                        break
                for k in pid_keys:
                    if k in fmap:
                        pid_col = fmap[k]
                        break
                if case_col and pid_col:
                    for row in reader:
                        v_case = row.get(case_col)
                        if v_case and str(v_case).strip() == s:
                            v_pid = row.get(pid_col)
                            if v_pid and str(v_pid).strip():
                                return str(v_pid).strip()
        except Exception:
            pass
    return None


def _parse_csv_labels(path):
    entries = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        field_map = {_normalize_name(name): name for name in fieldnames}
        for row in reader:
            entry = _build_entry(row, field_map, path)
            if entry is not None:
                entries.append(entry)
    return entries


def _parse_json_labels(path):
    entries = []
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        iterable = data
    elif isinstance(data, dict):
        if isinstance(data.get("lesions"), list):
            iterable = data["lesions"]
        elif isinstance(data.get("items"), list):
            iterable = data["items"]
        else:
            iterable = list(data.values())
    else:
        return entries

    for item in iterable:
        if not isinstance(item, dict):
            continue
        field_map = {_normalize_name(name): name for name in item.keys()}
        entry = _build_entry(item, field_map, path)
        if entry is not None:
            entries.append(entry)

    return entries


def _scan_label_files(data_root):
    paths = []
    root_key = os.path.abspath(data_root) if data_root else None
    if not root_key:
        return paths
    dirs = []
    base = root_key
    dirs.append(os.path.join(base, "LABELS"))
    dirs.append(os.path.join(base, "labels"))
    parent = os.path.dirname(base)
    dirs.append(os.path.join(parent, "LABELS"))
    dirs.append(os.path.join(parent, "labels"))
    priority_names = [
        "prostatex-findings-train.csv",
        "prostatex-findings-test.csv",
        "prostatex_findings.csv",
        "labels.csv",
        "labels.json",
    ]
    for d in dirs:
        if not os.path.isdir(d):
            continue
        entries = os.listdir(d)
        lower_map = {name.lower(): name for name in entries}
        for wanted in priority_names:
            if wanted in lower_map:
                paths.append(os.path.join(d, lower_map[wanted]))
                return paths
    return paths


def _ensure_labels_loaded(data_root):
    global _LABELS_LOADED, _GT_CACHE, _LAST_DATA_ROOT_KEY

    key = os.path.abspath(data_root) if data_root else None
    if _LABELS_LOADED and key == _LAST_DATA_ROOT_KEY:
        return

    _GT_CACHE = {}
    paths = _scan_label_files(data_root)
    if not paths:
        _LABELS_LOADED = True
        _LAST_DATA_ROOT_KEY = key
        return
    path = paths[0]
    try:
        if path.lower().endswith(".csv"):
            entries = _parse_csv_labels(path)
        elif path.lower().endswith(".json"):
            entries = _parse_json_labels(path)
        else:
            entries = []
    except Exception:
        entries = []
    for e in entries:
        k = e["patient_key"]
        if k not in _GT_CACHE:
            _GT_CACHE[k] = []
        _GT_CACHE[k].append(e)
    _LABELS_LOADED = True
    _LAST_DATA_ROOT_KEY = key


def get_gt_for_case(case_name, data_root=None):
    _ensure_labels_loaded(data_root)
    if not case_name:
        return []
    key = _normalize_patient_id(case_name)
    entries = _GT_CACHE.get(key, [])
    result = []
    counter = 1
    for e in entries:
        lesion_id = e.get("lesion_id")
        if not lesion_id:
            lesion_id = f"GT{counter}"
            counter += 1
        result.append(
            {
                "patient_id": e["patient_id"],
                "lesion_id": lesion_id,
                "xyz_mm": e["xyz_mm"],
                "ggg": e.get("ggg"),
                "isup": e.get("isup"),
                "clinsig": e.get("clinsig"),
                "zone": e.get("zone"),
                "source": e.get("source"),
            }
        )
    return result
