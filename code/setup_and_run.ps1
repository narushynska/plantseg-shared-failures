# One-time setup + run. Open PowerShell in this folder and run:
#   powershell -ExecutionPolicy Bypass -File .\setup_and_run.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path .venv)) {
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv .venv } else { python -m venv .venv }
    .\.venv\Scripts\python -m pip install --upgrade pip
    .\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    .\.venv\Scripts\pip install segmentation-models-pytorch albumentations opencv-python pandas
}
.\.venv\Scripts\python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
.\.venv\Scripts\python run_all.py --data plantseg --smoke
.\.venv\Scripts\python run_all.py --data plantseg
