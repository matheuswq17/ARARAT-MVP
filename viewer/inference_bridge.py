from __future__ import annotations

import json
import subprocess
import hashlib
import logging
import shutil
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import joblib

import SimpleITK as sitk
from radiomics import featureextractor
try:
    from . import path_utils
except Exception:
    from viewer import path_utils


logger = logging.getLogger("ararat.inference")


def _ensure_radiomics_validation_files() -> None:
    param_schema_yaml = """# Parameters schema
name: Parameter schema
desc: This schema defines what arguments may be present in the parameters file that can be passed to the pyradiomics package.
type: map
mapping:
  setting: &settings
    type: map
    mapping:
      minimumROIDimensions:
        type: int
        range:
          min: 1
          max: 3
      minimumROISize:
        type: int
        range:
          min-ex: 0
      geometryTolerance:
        type: float
        range:
          min-ex: 0
      correctMask:
        type: bool
      additionalInfo:
        type: bool
      label:
        type: int
        range:
          min-ex: 0
      label_channel:
        type: int
        range:
          min: 0
      binWidth:
        type: float
        range:
          min-ex: 0
      binCount:
        type: int
        range:
          min-ex: 0
      normalize:
        type: bool
      normalizeScale:
        type: float
        range:
          min-ex: 0
      removeOutliers:
        type: float
        range:
          min-ex: 0
      resampledPixelSpacing:
        seq:
          - type: float
            range:
              min: 0
      interpolator:
        type: any
        func: checkInterpolator
      padDistance:
        type: int
        range:
          min: 0
      distances:
        seq:
          - type: int
            range:
              min-ex: 0
      force2D:
        type: bool
      force2Ddimension:
        type: int
        range:
          min: 0
          max: 2
      resegmentRange:
        seq:
          - type: float
      resegmentMode:
        type: str
        enum: ['absolute', 'relative', 'sigma']
      resegmentShape:
        type: bool
      preCrop:
        type: bool
      sigma:
        seq:
          - type: float
            range:
              min-ex: 0
      start_level:
        type: int
        range:
          min: 0
      level:
        type: int
        range:
          min-ex: 0
      wavelet:
        type: str
        func: checkWavelet
      gradientUseSpacing:
        type: bool
      lbp2DRadius:
        type: float
        range:
          min-ex: 0
      lbp2DSamples:
        type: int
        range:
          min: 1
      lbp2DMethod:
        type: str
        enum: ['default', 'ror', 'uniform', 'var']
      lbp3DLevels:
        type: int
        range:
          min: 1
      lbp3DIcosphereRadius:
        type: float
        range:
          min-ex: 0
      lbp3DIcosphereSubdivision:
        type: int
        range:
          min: 0
      voxelArrayShift:
        type: int
      symmetricalGLCM:
        type: bool
      weightingNorm:
        type: any
        func: checkWeighting
      gldm_a:
        type: int
        range:
          min: 0

  voxelSetting:
    type: map
    mapping:
      kernelRadius:
        type: int
        range:
          min-ex: 0
      maskedKernel:
        type: bool
      initValue:
        type: float
      voxelBatch:
        type: int
        range:
          min-ex: 0

  featureClass:
    type: map
    func: checkFeatureClass
    matching-rule: 'any'
    mapping:
      regex;(.+):
        type: any

  imageType:
    type: map
    func: checkImageType
    matching-rule: 'any'
    mapping:
       regex;(.+): *settings
"""

    schema_funcs_py = """import pywt
import six

from radiomics import getFeatureClasses, getImageTypes

featureClasses = getFeatureClasses()
imageTypes = getImageTypes()

def checkWavelet(value, rule_obj, path):
  if not isinstance(value, six.string_types):
    raise TypeError('Wavelet not expected type (str)')
  wavelist = pywt.wavelist()
  if value not in wavelist:
    raise ValueError('Wavelet "%s" not available in pyWavelets %s' % (value, wavelist))
  return True


def checkInterpolator(value, rule_obj, path):
  if value is None:
    return True
  if isinstance(value, six.string_types):
    enum = {'sitkNearestNeighbor',
            'sitkLinear',
            'sitkBSpline',
            'sitkGaussian',
            'sitkLabelGaussian',
            'sitkHammingWindowedSinc',
            'sitkCosineWindowedSinc',
            'sitkWelchWindowedSinc',
            'sitkLanczosWindowedSinc',
            'sitkBlackmanWindowedSinc'}
    if value not in enum:
      raise ValueError('Interpolator value "%s" not valid, possible values: %s' % (value, enum))
  elif isinstance(value, int):
    if value < 1 or value > 10:
      raise ValueError('Intepolator value %i, must be in range of [1-10]' % (value))
  else:
    raise TypeError('Interpolator not expected type (str or int)')
  return True


def checkWeighting(value, rule_obj, path):
  if value is None:
    return True
  elif isinstance(value, six.string_types):
    enum = ['euclidean', 'manhattan', 'infinity', 'no_weighting']
    if value not in enum:
      raise ValueError('WeightingNorm value "%s" not valid, possible values: %s' % (value, enum))
  else:
    raise TypeError('WeightingNorm not expected type (str or None)')
  return True


def checkFeatureClass(value, rule_obj, path):
  global featureClasses
  if value is None:
    raise TypeError('featureClass dictionary cannot be None value')
  for className, features in six.iteritems(value):
    if className not in featureClasses.keys():
      raise ValueError(
        'Feature Class %s is not recognized. Available feature classes are %s' % (className, list(featureClasses.keys())))
    if features is not None:
      if not isinstance(features, list):
        raise TypeError('Value of feature class %s not expected type (list)' % (className))
      unrecognizedFeatures = set(features) - set(featureClasses[className].getFeatureNames())
      if len(unrecognizedFeatures) > 0:
        raise ValueError('Feature Class %s contains unrecognized features: %s' % (className, str(unrecognizedFeatures)))

  return True


def checkImageType(value, rule_obj, path):
  global imageTypes
  if value is None:
    raise TypeError('imageType dictionary cannot be None value')

  for im_type in value:
    if im_type not in imageTypes:
      raise ValueError('Image Type %s is not recognized. Available image types are %s' %
                       (im_type, imageTypes))

  return True
"""

    try:
        import radiomics
        import radiomics.featureextractor as fe
    except Exception:
        return

    try:
        schema_file, schema_funcs = radiomics.getParameterValidationFiles()
    except Exception:
        return

    schema_p = Path(schema_file)
    funcs_p = Path(schema_funcs)
    if schema_p.exists() and funcs_p.exists():
        return

    writable_dir = path_utils.resolve_writable_path("radiomics", "schemas")
    try:
        writable_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    dst_schema = writable_dir / "paramSchema.yaml"
    dst_funcs = writable_dir / "schemaFuncs.py"

    try:
        if not dst_schema.exists():
            dst_schema.write_text(param_schema_yaml, encoding="utf-8")
        if not dst_funcs.exists():
            dst_funcs.write_text(schema_funcs_py, encoding="utf-8")
    except Exception:
        return

    if not dst_schema.exists() or not dst_funcs.exists():
        return

    def _patched():
        return str(dst_schema), str(dst_funcs)

    try:
        radiomics.getParameterValidationFiles = _patched
        fe.getParameterValidationFiles = _patched
    except Exception:
        return


_PORTABLE_MODEL_CACHE: Optional[Dict[str, Any]] = None
_PORTABLE_MODEL_CACHE_KEY: Optional[str] = None


def _load_portable_model(model_dir: Path) -> Optional[Dict[str, Any]]:
    global _PORTABLE_MODEL_CACHE, _PORTABLE_MODEL_CACHE_KEY
    p = model_dir / "model_portable.json"
    if not p.exists():
        return None
    key = str(p.resolve())
    if _PORTABLE_MODEL_CACHE is not None and _PORTABLE_MODEL_CACHE_KEY == key:
        return _PORTABLE_MODEL_CACHE
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != "ararat_portable_logreg_v1":
        return None
    _PORTABLE_MODEL_CACHE = data
    _PORTABLE_MODEL_CACHE_KEY = key
    return data


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    if x != x or x == float("inf") or x == float("-inf"):
        return float("nan")
    return x


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _predict_proba_portable(portable: Dict[str, Any], features_row: Dict[str, Any]) -> float:
    feats_all = portable.get("features") or []
    keep_indices = portable.get("keep_indices") or []
    stats = ((portable.get("imputer") or {}).get("statistics")) or []
    mean = ((portable.get("scaler") or {}).get("mean")) or []
    scale = ((portable.get("scaler") or {}).get("scale")) or []
    coef = ((portable.get("logreg") or {}).get("coef")) or []
    intercept = float((portable.get("logreg") or {}).get("intercept", 0.0))

    if not isinstance(feats_all, list) or not isinstance(keep_indices, list):
        return 0.0

    x_keep: List[float] = []
    for j, idx in enumerate(keep_indices):
        if not isinstance(idx, int) or idx < 0 or idx >= len(feats_all):
            return 0.0
        name = feats_all[idx]
        raw = features_row.get(name)
        v = _safe_float(raw)
        if v != v:
            if idx < len(stats) and stats[idx] is not None:
                v = float(stats[idx])
            else:
                v = 0.0
        mu = float(mean[j]) if j < len(mean) and mean[j] is not None else 0.0
        sd = float(scale[j]) if j < len(scale) and scale[j] is not None else 1.0
        if sd == 0.0:
            x_keep.append(0.0)
        else:
            x_keep.append((v - mu) / sd)

    z = intercept
    for j, xv in enumerate(x_keep):
        w = float(coef[j]) if j < len(coef) and coef[j] is not None else 0.0
        z += w * xv
    return float(_sigmoid(z))


@dataclass
class InferenceConfig:
    repo_root: Path
    infer_python: Path           # .venv_infer/Scripts/python.exe
    model_dir: Path              # inference/models/v1_prostatex
    meta_path: Path              # inference/models/v1_prostatex/meta.json
    radiomics_params: Path       # radiomics_params.yaml (na raiz do repo)


def _find_repo_root() -> Path:
    return path_utils.get_app_root()


def load_config() -> InferenceConfig:
    repo = _find_repo_root()

    cfg_path = path_utils.get_config_path()
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


def _run_infer_inprocess(cfg: InferenceConfig, features_csv: Path, out_json: Path) -> Dict[str, Any]:
    df = pd.read_csv(features_csv)
    if df.empty:
        raise RuntimeError("features_csv vazio.")
    meta = _load_meta(cfg.meta_path)
    thr = float(meta.get("thr_cv", meta.get("threshold_default", 0.5)))
    model_file = meta.get("model_file", "model.joblib")
    model_path = cfg.model_dir / model_file
    cols = meta.get("features")
    if not cols or not isinstance(cols, list):
        cols = list(df.columns)
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"model_file não encontrado: {model_path}")
        model = joblib.load(str(model_path))
        X = df.loc[[0], cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[:, 1][0])
        elif hasattr(model, "decision_function"):
            s = float(model.decision_function(X)[0])
            prob = float(1.0 / (1.0 + math.exp(-s)))
        else:
            raise TypeError("Modelo não suporta predict_proba nem decision_function.")
        pred = int(prob >= thr)
        out = {
            "model": meta.get("name", cfg.model_dir.name),
            "prob_pos": float(prob),
            "thr_cv": float(thr),
            "pred_label": int(pred),
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out
    except Exception:
        logger.exception("infer_inprocess_joblib_falhou model_path=%s", model_path)

    portable = _load_portable_model(cfg.model_dir)
    if portable is None:
        raise RuntimeError("Modelo joblib falhou e model_portable.json ausente para fallback.")
    row = df.iloc[0].to_dict()
    prob = _predict_proba_portable(portable, row)
    pred = int(prob >= thr)
    out = {
        "model": meta.get("name", cfg.model_dir.name),
        "prob_pos": float(prob),
        "thr_cv": float(thr),
        "pred_label": int(pred),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _load_meta(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json não encontrado: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


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

    _ensure_radiomics_validation_files()
    extractor = featureextractor.RadiomicsFeatureExtractor(str(params_yaml))
    result = extractor.execute(image, mask)

    feats: Dict[str, Any] = {}
    for k, v in result.items():
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
    if not features_csv.exists():
        raise FileNotFoundError(f"features_csv não encontrado: {features_csv}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    if (not cfg.infer_python.exists()) or path_utils.is_frozen():
        logger.warning("infer_python_indisponivel_ou_frozen fallback_local infer_python=%s", cfg.infer_python)
        return _run_infer_inprocess(cfg=cfg, features_csv=features_csv, out_json=out_json)

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
        logger.error("falha_infer_cli_subprocess stdout=%s stderr=%s", p.stdout, p.stderr)
        return _run_infer_inprocess(cfg=cfg, features_csv=features_csv, out_json=out_json)

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
    logger.info("paths model_dir=%s meta=%s radiomics=%s infer_python=%s", cfg.model_dir, cfg.meta_path, cfg.radiomics_params, cfg.infer_python)
    meta = _load_meta(cfg.meta_path)

    cols = meta.get("features")
    if not cols or not isinstance(cols, list):
        raise ValueError("meta.json não tem lista 'features' válida.")

    model_file = cfg.model_dir / meta.get("model_file", "model.joblib")
    if not model_file.exists():
        raise FileNotFoundError(f"model.joblib não encontrado: {model_file}")
    if not cfg.radiomics_params.exists():
        raise FileNotFoundError(f"radiomics_params.yaml não existe: {cfg.radiomics_params}")

    model_sha = _sha256_of_file(model_file)
    meta_sha = _sha256_of_file(cfg.meta_path)
    params_sha = _sha256_of_file(cfg.radiomics_params)

    thresholds_path = cfg.model_dir / "thresholds.json"
    thresholds_cfg: Optional[Dict[str, Any]] = None
    if thresholds_path.exists():
        thresholds_cfg = json.loads(thresholds_path.read_text(encoding="utf-8"))

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

        base = _mask_base_name(m)
        csv_path = export_dir / f"features_{base}.csv"
        _write_one_row_csv(cols=cols, values=feats, out_csv=csv_path)

        out_json = export_dir / f"pred_{base}.json"
        pred = run_infer_cli_from_csv(cfg=cfg, features_csv=csv_path, out_json=out_json)

        pred["mask_base"] = base
        pred["lesion"] = _lesion_label(base)
        prob = float(pred.get("prob_pos", 0.0))
        thr = float(pred.get("thr_cv", meta.get("thr_cv", meta.get("threshold_default", 0.5))))

        risk_category = None
        thresholds_source = "thr_cv"
        if thresholds_cfg and isinstance(thresholds_cfg, dict):
            bins = thresholds_cfg.get("bins") or []
            labels = thresholds_cfg.get("labels") or []
            if isinstance(bins, list) and isinstance(labels, list) and len(labels) == len(bins) + 1:
                thresholds_source = "thresholds.json"
                if prob < float(bins[0]):
                    risk_category = labels[0]
                elif len(bins) == 1 or prob < float(bins[1]):
                    risk_category = labels[1]
                else:
                    risk_category = labels[-1]
        if risk_category is None:
            risk_category = "Positivo" if prob >= thr else "Negativo"

        pred["risk_category"] = risk_category
        pred["risk_percent"] = prob * 100.0
        pred["thresholds_source"] = thresholds_source
        pred["lesion_id"] = pred["lesion"]
        pred["mask_file"] = m.name
        pred["features_file"] = csv_path.name
        pred["model_version"] = {
            "model_joblib_sha256": model_sha,
            "meta_json_sha256": meta_sha,
            "radiomics_params_sha256": params_sha,
        }
        pred["series_dir"] = str(dicom_dir)
        pred["export_dir"] = str(export_dir)
        try:
            pred["radiomics_params"] = str(cfg.radiomics_params.relative_to(cfg.repo_root))
        except ValueError:
            pred["radiomics_params"] = str(cfg.radiomics_params)
        meta_feats = pred.get("features_metadata") or {}
        meta_feats["tumor_vs_não_tumor"] = {"constant_feature": True, "value": 1.0}
        pred["features_metadata"] = meta_feats

        out_json.write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
        preds.append(pred)

    summary_path = export_dir / "case_summary.json"
    summary = {
        "case": export_dir.parent.name,
        "export_dir": str(export_dir),
        "n_lesions": len(preds),
        "lesions": preds,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return preds
