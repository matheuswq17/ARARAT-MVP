from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# Viewer env (.venv39) tem SimpleITK + pyradiomics
import SimpleITK as sitk
from radiomics import featureextractor


@dataclass
class InferenceConfig:
    repo_root: Path
    infer_python: Path           # .venv_infer/Scripts/python.exe
    model_dir: Path              # inference/models/v1_prostatex
    meta_path: Path              # inference/models/v1_prostatex/meta.json
    radiomics_params: Path       # radiomics_params.yaml (na raiz do repo)


def _find_repo_root() -> Path:
    # assume: este arquivo está em <repo>/viewer/inference_bridge.py
    return Path(__file__).resolve().parents[1]


def load_config() -> InferenceConfig:
    repo = _find_repo_root()

    # prioridade: config_local.json pode sobrescrever caminhos (opcional)
    cfg_path = repo / "config_local.json"
    infer_py = repo / ".venv_infer" / "Scripts" / "python.exe"
    model_dir = repo / "inference" / "models" / "v1_prostatex"
    meta_path = model_dir / "meta.json"
    radiomics_params = repo / "radiomics_params.yaml"

    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(cfg, dict):
                if cfg.get("infer_python"):
                    infer_py = Path(cfg["infer_python"]).expanduser().resolve()
                if cfg.get("model_dir"):
                    model_dir = Path(cfg["model_dir"]).expanduser().resolve()
                    meta_path = model_dir / "meta.json"
        except Exception:
            pass

    return InferenceConfig(
        repo_root=repo,
        infer_python=infer_py,
        model_dir=model_dir,
        meta_path=meta_path,
        radiomics_params=radiomics_params,
    )


def _load_meta(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json não encontrado: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def extract_radiomics_features(dicom_dir: Path, mask_path: Path, params_yaml: Path) -> Dict[str, Any]:
    """
    Lê a série DICOM e a máscara NIfTI e extrai radiomics via PyRadiomics.
    Retorna dict com chaves tipo 'original_firstorder_Median', etc.
    """
    if not dicom_dir.exists():
        raise FileNotFoundError(f"dicom_dir não existe: {dicom_dir}")
    if not mask_path.exists():
        raise FileNotFoundError(f"mask não existe: {mask_path}")
    if not params_yaml.exists():
        raise FileNotFoundError(f"radiomics_params.yaml não existe: {params_yaml}")

    reader = sitk.ImageSeriesReader()
    series_files = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not series_files:
        raise FileNotFoundError(f"Nenhum arquivo DICOM encontrado em: {dicom_dir}")
    reader.SetFileNames(series_files)
    image = reader.Execute()

    mask = sitk.ReadImage(str(mask_path))
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    extractor = featureextractor.RadiomicsFeatureExtractor(str(params_yaml))
    result = extractor.execute(image, mask)

    feats: Dict[str, Any] = {}
    for k, v in result.items():
        if str(k).startswith("diagnostics_"):
            continue
        # tenta converter pra float quando possível
        try:
            if hasattr(v, "item"):
                v = v.item()
            feats[str(k)] = float(v)
        except Exception:
            feats[str(k)] = v
    return feats


def _write_one_row_csv(cols: List[str], values: Dict[str, Any], out_csv: Path) -> None:
    row = {}
    for c in cols:
        v = values.get(c, np.nan)
        # garante numérico quando der
        try:
            row[c] = float(v) if v is not None else np.nan
        except Exception:
            row[c] = np.nan
    if "tumor_vs_não_tumor" in row:
        row["tumor_vs_não_tumor"] = 1.0
    df = pd.DataFrame([row])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")


def run_infer_cli_from_csv(cfg: InferenceConfig, features_csv: Path, out_json: Path) -> Dict[str, Any]:
    """
    Chama o infer_cli (modo A) dentro do .venv_infer e lê o JSON de saída.
    """
    if not cfg.infer_python.exists():
        raise FileNotFoundError(
            f"Python de inferência não encontrado: {cfg.infer_python}\n"
            f"Você criou o .venv_infer?"
        )
    if not features_csv.exists():
        raise FileNotFoundError(f"features_csv não encontrado: {features_csv}")

    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(cfg.infer_python),
        "-m",
        "inference.infer_cli",
        "--features_csv",
        str(features_csv),
        "--row_index",
        "0",
        "--model_dir",
        str(cfg.model_dir),
        "--out_json",
        str(out_json),
    ]

    p = subprocess.run(
        cmd,
        cwd=str(cfg.repo_root),
        capture_output=True,
        text=True,
        shell=False,
    )

    if p.returncode != 0:
        raise RuntimeError(
            "Falha ao rodar infer_cli.\n"
            f"CMD: {' '.join(cmd)}\n\n"
            f"STDOUT:\n{p.stdout}\n\n"
            f"STDERR:\n{p.stderr}"
        )

    if not out_json.exists():
        raise RuntimeError(f"infer_cli terminou mas não gerou JSON: {out_json}")

    return json.loads(out_json.read_text(encoding="utf-8"))


def predict_for_export_folder(
    dicom_dir: Path,
    export_dir: Path,
    masks_glob: str = "mask_*.nii.gz",
) -> List[Dict[str, Any]]:
    """
    Para cada mask_*.nii.gz em export_dir:
      1) extrai radiomics no env do Viewer (.venv39)
      2) cria features_<mask>.csv (1 linha, colunas do meta.json)
      3) chama infer_cli no .venv_infer e gera pred_<mask>.json
    Retorna lista com os dicts de predição.
    """
    cfg = load_config()
    meta = _load_meta(cfg.meta_path)

    cols = meta.get("features")
    if not cols or not isinstance(cols, list):
        raise ValueError("meta.json não tem lista 'features' válida.")

    export_dir = export_dir.resolve()
    masks = sorted(export_dir.glob(masks_glob))
    if not masks:
        raise FileNotFoundError(f"Nenhuma máscara encontrada em {export_dir} com glob {masks_glob}")

    def _mask_base_name(p: Path) -> str:
        n = p.name
        if n.endswith(".nii.gz"):
            return n[:-7]
        if n.endswith(".nii"):
            return n[:-4]
        return p.stem

    def _lesion_label(base: str) -> str:
        return base[5:] if base.startswith("mask_") else base

    preds: List[Dict[str, Any]] = []
    for m in masks:
        feats = extract_radiomics_features(dicom_dir=Path(dicom_dir), mask_path=m, params_yaml=cfg.radiomics_params)

        # monta CSV 1-linha no formato do modelo
        base = _mask_base_name(m)
        csv_path = export_dir / f"features_{base}.csv"
        _write_one_row_csv(cols=cols, values=feats, out_csv=csv_path)

        # chama inferência no .venv_infer
        out_json = export_dir / f"pred_{base}.json"
        pred = run_infer_cli_from_csv(cfg=cfg, features_csv=csv_path, out_json=out_json)

        pred["mask_base"] = base
        pred["lesion"] = _lesion_label(base)
        preds.append(pred)

    return preds
