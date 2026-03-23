import os
import csv
import json
import re
try:
    from . import path_utils
except Exception:
    from viewer import path_utils


_GT_CACHE = {}
_LABELS_LOADED = False
_LAST_DATA_ROOT_KEY = None
_LABELS_SOURCE = None
_LABELS_ERROR = None
_LABELS_STATS = None


def _normalize_name(name):
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _normalize_patient_id(value):
    s = str(value).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        if (s.strip().isdigit() or "prostatex" in s.lower()) and len(digits) < 4:
            return digits.zfill(4)
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
    if s.strip().isdigit():
        return f"ProstateX-{s.strip().zfill(4)}"

    m = re.search(r"^prostatex[-_\s]*(\d{1,4})$", s.lower())
    if m:
        digits4 = m.group(1).zfill(4)
        return f"ProstateX-{digits4}"

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
    manifest = str(path_utils.resolve_writable_path("exports", "roi_manifest.csv"))
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
    last_exc = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                field_map = {_normalize_name(name): name for name in fieldnames}
                for row in reader:
                    entry = _build_entry(row, field_map, path)
                    if entry is not None:
                        entries.append(entry)
            return entries
        except UnicodeDecodeError as e:
            last_exc = e
            entries = []
            continue
        except Exception as e:
            last_exc = e
            entries = []
            break
    if last_exc is not None:
        raise last_exc
    return entries


def _parse_json_labels(path):
    entries = []
    last_exc = None
    data = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                data = json.load(f)
            break
        except UnicodeDecodeError as e:
            last_exc = e
            continue
        except Exception as e:
            last_exc = e
            break
    if data is None:
        if last_exc is not None:
            raise last_exc
        return entries

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
    dirs = []
    seen_dirs = set()

    def add_dir(d):
        if not d:
            return
        k = os.path.abspath(str(d))
        if k in seen_dirs:
            return
        seen_dirs.add(k)
        dirs.append(k)

    root_key = os.path.abspath(data_root) if data_root else None
    if root_key:
        base = root_key
        add_dir(os.path.join(base, "LABELS"))
        add_dir(os.path.join(base, "labels"))
        parent = os.path.dirname(base)
        add_dir(os.path.join(parent, "LABELS"))
        add_dir(os.path.join(parent, "labels"))

    add_dir(path_utils.resolve_writable_path("LABELS"))
    add_dir(path_utils.resolve_path("data", "PROSTATEx", "LABELS"))

    priority_names = [
        "labels_merged.csv",
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


def _load_labels_from_candidates(paths):
    last_error = None
    for path in paths:
        try:
            if not os.path.exists(path):
                continue
            if path.lower().endswith(".csv"):
                entries = _parse_csv_labels(path)
            elif path.lower().endswith(".json"):
                entries = _parse_json_labels(path)
            else:
                entries = []
            if not entries:
                continue
            gt_cache = {}
            for e in entries:
                k = e["patient_key"]
                if k not in gt_cache:
                    gt_cache[k] = []
                gt_cache[k].append(e)
            return gt_cache, path, None
        except Exception as e:
            last_error = str(e)
            continue
    if last_error is None:
        last_error = "labels_nao_encontradas_ou_invalidas"
    return {}, None, last_error


def _ensure_labels_loaded(data_root):
    global _LABELS_LOADED, _GT_CACHE, _LAST_DATA_ROOT_KEY, _LABELS_SOURCE, _LABELS_ERROR, _LABELS_STATS

    key = os.path.abspath(data_root) if data_root else None
    if _LABELS_LOADED and key == _LAST_DATA_ROOT_KEY:
        return

    _GT_CACHE = {}
    _LABELS_SOURCE = None
    _LABELS_ERROR = None
    _LABELS_STATS = None

    paths = _scan_label_files(data_root)
    gt_cache, source, err = _load_labels_from_candidates(paths)
    _GT_CACHE = gt_cache
    _LABELS_SOURCE = source
    _LABELS_ERROR = err
    _LABELS_STATS = {
        "patients": len(_GT_CACHE),
        "lesions": sum(len(v) for v in _GT_CACHE.values()) if _GT_CACHE else 0,
    }
    _LABELS_LOADED = True
    _LAST_DATA_ROOT_KEY = key


def preload_labels(data_root=None):
    _ensure_labels_loaded(data_root)
    ok = bool(_GT_CACHE)
    return {
        "ok": ok,
        "source": _LABELS_SOURCE,
        "error": _LABELS_ERROR,
        "stats": _LABELS_STATS,
    }


def get_labels_status():
    return {
        "loaded": _LABELS_LOADED,
        "source": _LABELS_SOURCE,
        "error": _LABELS_ERROR,
        "stats": _LABELS_STATS,
    }


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
