param(
  [Parameter(Mandatory=$true)][string]$FeaturesCsv,
  [int]$RowIndex = 0,
  [string]$ModelDir = "inference\models\v1_prostatex",
  [string]$OutJson = "inference_output.json"
)

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo
Set-Location ..

if (-not (Test-Path ".\.venv_infer\Scripts\python.exe")) {
  Write-Host "Ambiente .venv_infer não encontrado. Execute scripts\setup_inference_env.ps1"
  exit 1
}

.\.venv_infer\Scripts\python -m inference.infer_cli --features_csv "$FeaturesCsv" --row_index $RowIndex --model_dir "$ModelDir" --out_json "$OutJson"

Write-Host "[OK] JSON gerado em $OutJson"
