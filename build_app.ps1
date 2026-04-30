$ErrorActionPreference = "Stop"

$python = ".\SPISv2_env\Scripts\python.exe"
$distPath = "dist_app"

if (-not (Test-Path $python)) {
    $python = "python"
}

& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name "HPSS Studio" `
    --distpath $distPath `
    --collect-all customtkinter `
    --collect-submodules librosa `
    --collect-submodules soundfile `
    --hidden-import matplotlib.backends.backend_tkagg `
    --hidden-import scipy.special._cdflib `
    app.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller failed. Close any running HPSS Studio windows and retry."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Build complete: $distPath\HPSS Studio\HPSS Studio.exe"
