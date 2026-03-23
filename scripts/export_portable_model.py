from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib


def _to_float_list(values) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for v in values:
        if v is None:
            out.append(None)
            continue
        try:
            fv = float(v)
        except Exception:
            out.append(None)
            continue
        if fv != fv:
            out.append(None)
            continue
        out.append(fv)
    return out


def export_model(model_dir: Path) -> Path:
    meta_path = model_dir / "meta.json"
    model_path = model_dir / "model.joblib"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    features = meta.get("features") or []
    if not isinstance(features, list) or not features:
        raise RuntimeError("meta.json sem lista 'features' válida.")

    pipe = joblib.load(str(model_path))

    imputer = pipe.named_steps.get("imputer")
    scaler = pipe.named_steps.get("scaler")
    clf = pipe.named_steps.get("clf")

    if imputer is None or scaler is None or clf is None:
        raise RuntimeError("Pipeline inesperado: precisa ter steps imputer/scaler/clf.")

    stats = getattr(imputer, "statistics_", None)
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    coef = getattr(clf, "coef_", None)
    intercept = getattr(clf, "intercept_", None)
    classes = getattr(clf, "classes_", None)

    if stats is None or mean is None or scale is None or coef is None or intercept is None:
        raise RuntimeError("Modelo não contém parâmetros necessários (statistics_/mean_/scale_/coef_/intercept_).")

    if len(stats) != len(features):
        raise RuntimeError("Dimensão do imputer não bate com meta.json.")

    stats_list = _to_float_list(stats)
    keep_indices = [i for i, v in enumerate(stats_list) if v is not None]
    if len(mean) != len(keep_indices) or len(scale) != len(keep_indices):
        raise RuntimeError("Dimensão do scaler não bate com meta.json/imputer.")

    coef_row = coef[0] if hasattr(coef, "__len__") else coef
    if hasattr(coef_row, "__len__") and len(coef_row) != len(keep_indices):
        raise RuntimeError("Dimensão do classificador não bate com meta.json/imputer.")
    intercept_val = intercept[0] if hasattr(intercept, "__len__") else intercept

    payload: Dict[str, Any] = {
        "format": "ararat_portable_logreg_v1",
        "features": list(features),
        "keep_indices": list(keep_indices),
        "imputer": {
            "strategy": "median",
            "statistics": stats_list,
        },
        "scaler": {
            "mean": _to_float_list(mean),
            "scale": _to_float_list(scale),
        },
        "logreg": {
            "coef": _to_float_list(coef_row),
            "intercept": float(intercept_val),
            "classes": [int(x) for x in (classes.tolist() if hasattr(classes, "tolist") else (classes or [0, 1]))],
        },
    }

    out_path = model_dir / "model_portable.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "inference" / "models" / "v1_prostatex"
    out = export_model(model_dir)
    print(str(out))


if __name__ == "__main__":
    main()
