# ARARAT MVP – Inferência

## Ambiente de Inferência
- Executar de uma janela do PowerShell na raiz do repositório:

```powershell
scripts\setup_inference_env.ps1
```

- Isto cria `.venv_infer` (Python ≥ 3.11) e instala as dependências fixas para carregar `model.joblib` com `scikit-learn==1.8.0`. A instalação do PyRadiomics é opcional; se falhar, o modo CSV permanece funcional.

### Por que não usar `.venv39`?
- O `.venv39` usa `scikit-learn 1.4.x` e não é compatível com o `model.joblib` salvo na versão 1.8.0. Use o ambiente `.venv_infer` para inferência.

## Inferência via CSV (Modo A)
- CSV com as features na ordem definida em `inference/models/v1_prostatex/meta.json` (campo `features`).
- Exemplo de execução:

```powershell
scripts\run_infer_features.ps1 -FeaturesCsv "caminho\para\features.csv" -RowIndex 0 -ModelDir "inference\models\v1_prostatex" -OutJson "pred.json"
```

- Saída (`pred.json`):
```json
{
  "model": "v1_prostatex",
  "prob_pos": 0.123,
  "thr_cv": 0.6162,
  "pred_label": 0,
  "features_used": { ... },
  "timestamp": "..."
}
```

## Inferência com DICOM+Mask (Modo B, opcional)
- Requer PyRadiomics. Se não estiver instalado, use o modo CSV.
- Execução (exemplo):

```powershell
.\.venv_infer\Scripts\python -m inference.infer_cli --dicom_dir "C:\pasta\serie" --mask "C:\pasta\mask_L1.nii.gz" --model_dir "inference\models\v1_prostatex"
```
