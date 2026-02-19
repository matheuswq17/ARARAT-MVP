from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

import SimpleITK as sitk
try:
    from radiomics import featureextractor
except Exception:
    featureextractor = None

try:
    import yaml  # PyYAML
except Exception:
    yaml = None


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def load_meta(model_dir: Path) -> dict:
    meta_yaml = model_dir / "meta.yaml"
    meta_json = model_dir / "meta.json"

    if meta_yaml.exists() and _read_text(meta_yaml).strip():
        if yaml is None:
            raise RuntimeError("PyYAML não instalado, mas meta.yaml existe. Instale pyyaml ou use meta.json.")
        meta = yaml.safe_load(_read_text(meta_yaml))
        return meta

    if meta_json.exists() and _read_text(meta_json).strip():
        meta = json.loads(_read_text(meta_json))
        return meta

    raise FileNotFoundError(f"Nenhum meta.yaml/meta.json válido em: {model_dir}")


def load_schema(model_dir: Path) -> dict | None:
    p = model_dir / "schema.json"
    if p.exists() and _read_text(p).strip():
        return json.loads(_read_text(p))
    return None


def ensure_sklearn_compatible():
    try:
        import sklearn  # noqa: F401
    except Exception:
        raise RuntimeError("scikit-learn ausente. Use o venv de inferência (Python >=3.11).")
    from sklearn import __version__ as skv
    if not skv.startswith("1.8."):
        raise RuntimeError(f"scikit-learn {skv} incompatível com o modelo. Use o venv de inferência (Python >=3.11) com scikit-learn==1.8.0.")


def load_dicom_series(dicom_dir: Path) -> sitk.Image:
    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM dir não existe: {dicom_dir}")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise ValueError(f"Nenhuma série DICOM encontrada em: {dicom_dir}")

    files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(files)
    img = reader.Execute()
    return img


def load_mask(mask_path: Path) -> sitk.Image:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask não existe: {mask_path}")
    m = sitk.ReadImage(str(mask_path))
    # força binário
    m = sitk.BinaryThreshold(m, 1, 1, 1, 0)
    m = sitk.Cast(m, sitk.sitkUInt8)
    return m


def align_mask_to_image(mask: sitk.Image, image: sitk.Image) -> sitk.Image:
    # Se geometria não bater, resample mask -> image
    if (
        mask.GetSize() != image.GetSize()
        or mask.GetSpacing() != image.GetSpacing()
        or mask.GetOrigin() != image.GetOrigin()
        or mask.GetDirection() != image.GetDirection()
    ):
        mask = sitk.Resample(
            mask,
            image,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
    return mask


def extract_radiomics(image: sitk.Image, mask: sitk.Image, params_yaml: Path) -> dict:
    if not params_yaml.exists() or not _read_text(params_yaml).strip():
        raise FileNotFoundError(f"radiomics_params.yaml ausente ou vazio: {params_yaml}")

    if featureextractor is None:
        raise RuntimeError("PyRadiomics não instalado. Use modo --features_csv ou instale pyradiomics.")

    extractor = featureextractor.RadiomicsFeatureExtractor(str(params_yaml))
    out = extractor.execute(image, mask)

    feats = {}
    for k, v in out.items():
        if k.startswith("diagnostics"):
            continue
        try:
            feats[k] = float(v)
        except Exception:
            # se vier algo não numérico, ignora
            pass
    return feats


def build_X(features_order: list[str], extracted: dict) -> pd.DataFrame:
    row = {}
    for f in features_order:
        row[f] = extracted.get(f, np.nan)

    X = pd.DataFrame([row], columns=features_order)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def predict_proba(model, X: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1][0]
        return float(p)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)[0]
        p = 1.0 / (1.0 + np.exp(-s))
        return float(p)
    raise TypeError("Modelo não suporta predict_proba nem decision_function.")


def infer_one(model_dir: Path, dicom_dir: Path, mask_path: Path, params_yaml: Path) -> dict:
    meta = load_meta(model_dir)
    schema = load_schema(model_dir)

    features_order = meta.get("features")
    if not features_order:
        # tenta schema
        if schema and "features" in schema:
            if isinstance(schema["features"][0], dict):
                features_order = [x["name"] for x in schema["features"]]
            else:
                features_order = list(schema["features"])
    if not features_order:
        raise ValueError("Lista de features não encontrada no meta.* ou schema.json")

    model_file = meta.get("model_file", "model.joblib")
    model_path = model_dir / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"model_file não encontrado: {model_path}")

    thr = float(meta.get("threshold_default", meta.get("thr_cv", 0.5)))
    pos_label = int(meta.get("pos_label", 1))

    image = load_dicom_series(dicom_dir)
    mask = load_mask(mask_path)
    mask = align_mask_to_image(mask, image)

    extracted = extract_radiomics(image, mask, params_yaml)
    X = build_X(features_order, extracted)

    ensure_sklearn_compatible()
    model = joblib.load(str(model_path))
    p = predict_proba(model, X)
    pred = int(p >= thr)

    return {
        "model": meta.get("name", model_dir.name),
        "model_path": str(model_path),
        "dicom_dir": str(dicom_dir),
        "mask_path": str(mask_path),
        "pos_label": pos_label,
        "threshold": thr,
        "proba_pos": p,
        "pred_label": pred,
        "missing_features": [f for f in features_order if f not in extracted],
    }


def find_masks_in_export(export_dir: Path) -> list[Path]:
    # pega mask_*.nii.gz dentro do export_dir
    masks = sorted(export_dir.glob("mask_*.nii.gz"))
    return masks


def infer_from_features_csv(model_dir: Path, csv_path: Path, row_index: int, out_json: Path | None) -> None:
    ensure_sklearn_compatible()
    meta = load_meta(model_dir)
    features_order = meta.get("features")
    if not features_order:
        schema = load_schema(model_dir)
        if schema and "features" in schema:
            if isinstance(schema["features"][0], dict):
                features_order = [x["name"] for x in schema["features"]]
            else:
                features_order = list(schema["features"])
    if not features_order:
        raise ValueError("Lista de features não encontrada no meta.* ou schema.json")
    model_file = meta.get("model_file", "model.joblib")
    model_path = model_dir / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"model_file não encontrado: {model_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in features_order if c not in df.columns]
    if missing:
        raise ValueError(f"Features ausentes no CSV: {missing}")
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index fora dos limites: {row_index} de 0..{len(df)-1}")
    X = df.loc[[row_index], features_order].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    model = joblib.load(str(model_path))
    thr = float(meta.get("thr_cv", meta.get("threshold_default", 0.5)))
    p = predict_proba(model, X)
    pred = int(p >= thr)
    features_used = {k: (None if pd.isna(v) else float(v)) for k, v in zip(features_order, X.iloc[0].tolist())}
    out = {
        "model": meta.get("name", model_dir.name),
        "prob_pos": float(p),
        "thr_cv": float(thr),
        "pred_label": int(pred),
        "features_used": features_used,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    s = json.dumps(out, ensure_ascii=False, indent=2)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(s, encoding="utf-8")
    else:
        print(s)


def main():
    repo_root = Path(__file__).resolve().parents[1]  # .../inference/infer_cli.py -> repo root
    default_model_dir = repo_root / "inference" / "models" / "v1_prostatex"
    default_params = repo_root / "radiomics_params.yaml"

    ap = argparse.ArgumentParser(description="ARARAT - Inference CLI (radiomics + model.joblib)")
    ap.add_argument("--features_csv", type=str, default=None, help="CSV com features já extraídas.")
    ap.add_argument("--row_index", type=int, default=0, help="Índice da linha do CSV.")
    ap.add_argument("--dicom_dir", type=str, default=None, help="Pasta contendo a série DICOM (um único series por pasta é recomendado).")
    ap.add_argument("--mask", type=str, default=None, help="Caminho para uma máscara .nii.gz (ex: mask_L1.nii.gz).")
    ap.add_argument("--export_dir", type=str, default=None, help="Pasta de export do viewer (contendo mask_*.nii.gz).")
    ap.add_argument("--model_dir", type=str, default=str(default_model_dir), help="Pasta do modelo (meta + joblib).")
    ap.add_argument("--params", type=str, default=str(default_params), help="radiomics_params.yaml")
    ap.add_argument("--out_json", type=str, default=None, help="Arquivo JSON de saída (modo CSV).")
    ap.add_argument("--write_json", action="store_true", help="Salva inference_results.json dentro do export_dir (se fornecido).")

    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    params_yaml = Path(args.params).resolve()
    if args.features_csv:
        csv_path = Path(args.features_csv).resolve()
        out_json = Path(args.out_json).resolve() if args.out_json else None
        infer_from_features_csv(model_dir, csv_path, int(args.row_index), out_json)
        return

    if args.export_dir:
        export_dir = Path(args.export_dir).resolve()
        if not export_dir.exists():
            raise FileNotFoundError(f"export_dir não existe: {export_dir}")

        if not args.dicom_dir:
            raise ValueError("Usando --export_dir, você também precisa passar --dicom_dir (série DICOM correspondente).")

        dicom_dir = Path(args.dicom_dir).resolve()
        masks = find_masks_in_export(export_dir)
        if not masks:
            raise FileNotFoundError(f"Nenhuma mask_*.nii.gz encontrada em: {export_dir}")

        results = []
        for m in masks:
            results.append(infer_one(model_dir, dicom_dir, m, params_yaml))

        print(json.dumps(results, ensure_ascii=False, indent=2))

        if args.write_json:
            outp = export_dir / "inference_results.json"
            outp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] Salvo: {outp}")

        return

    # modo single
    if not args.dicom_dir or not args.mask:
        raise ValueError("Use --dicom_dir e --mask, ou então use --export_dir (com --dicom_dir).")

    dicom_dir = Path(args.dicom_dir).resolve()
    mask_path = Path(args.mask).resolve()

    result = infer_one(model_dir, dicom_dir, mask_path, params_yaml)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
