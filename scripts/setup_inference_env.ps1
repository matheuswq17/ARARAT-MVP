$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location ..

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
  Write-Host "Launcher 'py' não encontrado. Instale Python 3.12+ com o launcher ou ajuste o script."
  exit 1
}

Write-Host "[1/4] Criando .venv_infer com py -3.12"
py -3.12 -m venv ".venv_infer"

Write-Host "[2/4] Atualizando pip"
.\.venv_infer\Scripts\python -m pip install -U pip

Write-Host "[3/4] Instalando requirements/inference.txt"
.\.venv_infer\Scripts\python -m pip install -r requirements\inference.txt

Write-Host "[4/4] Instalando PyRadiomics (opcional)"
try {
  .\.venv_infer\Scripts\python -m pip install pyradiomics
  Write-Host "PyRadiomics instalado."
} catch {
  Write-Host "PyRadiomics não instalado. Modo A (CSV) seguirá funcionando."
}

Write-Host "[OK] Ambiente de inferência pronto em .venv_infer"
