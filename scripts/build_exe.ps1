param(
    [switch]$Clean,
    [switch]$InstallDeps,
    [switch]$UsePy311
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

Write-Host "[ARARAT] Root: $root"

$work = Join-Path $root ("build_pyinstaller_" + (Get-Date -Format "yyyyMMdd_HHmmss"))

$preferPy311 = $UsePy311.IsPresent

function Resolve-BuildPython {
    param([switch]$PreferPy311)

    if ($PreferPy311 -and (Get-Command py -ErrorAction SilentlyContinue)) {
        try {
            py -3.11 -V | Out-Null
            return @{
                Exe = "py"
                Prefix = @("-3.11")
            }
        } catch {
        }
    }
    return @{
        Exe = "python"
        Prefix = @()
    }
}

$buildPy = Resolve-BuildPython -PreferPy311:$preferPy311
$buildExe = $buildPy.Exe
$buildPrefix = $buildPy.Prefix
$venvDir = Join-Path $root ".venv_build"
$venvPy = Join-Path $venvDir "Scripts\\python.exe"

function Invoke-Native {
    param(
        [string]$Exe,
        [string[]]$ArgList
    )
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Exe @ArgList 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($code -ne 0) {
        throw "Falha ao executar: $Exe $($ArgList -join ' ') (exit=$code)"
    }
}

if ($preferPy311) {
    if (-not (Test-Path $venvPy)) {
        Write-Host "[ARARAT] Criando venv build em .venv_build (Python 3.11)..."
        Invoke-Native -Exe $buildExe -ArgList ($buildPrefix + @("-m", "venv", $venvDir))
    }
    $buildExe = $venvPy
    $buildPrefix = @()
}

if ($Clean) {
    if (Test-Path ".\build") {
        try { Remove-Item ".\build" -Recurse -Force -ErrorAction Stop } catch { Write-Host "[ARARAT] Aviso: falha ao limpar build. Seguindo..." }
    }
    if (Test-Path ".\dist") {
        try { Remove-Item ".\dist" -Recurse -Force -ErrorAction Stop } catch { Write-Host "[ARARAT] Aviso: falha ao limpar dist. Seguindo..." }
    }
}

if ($InstallDeps) {
    Invoke-Native -Exe $buildExe -ArgList ($buildPrefix + @("-m", "pip", "install", "-r", ".\\requirements_freeze.txt"))
    Invoke-Native -Exe $buildExe -ArgList ($buildPrefix + @("-m", "pip", "install", "pyinstaller"))
}

Write-Host "[ARARAT] Building onefile EXE..."
Invoke-Native -Exe $buildExe -ArgList ($buildPrefix + @("-m", "PyInstaller.__main__", "--noconfirm", "--clean", "--log-level=WARN", "--workpath", $work, "--distpath", ".\\dist", ".\\ARARAT_Viewer.spec"))

if (-not (Test-Path ".\dist\ARARAT_Viewer.exe")) {
    throw "Build falhou: dist\ARARAT_Viewer.exe não foi gerado."
}

Write-Host "[ARARAT] OK: dist\ARARAT_Viewer.exe"
