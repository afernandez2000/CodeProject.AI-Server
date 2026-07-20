:: ============================================================================
::
:: Object Detection module install script for Windows
::
:: This script is called from ..\..\src\setup.bat
::
:: ============================================================================

@if "%1" NEQ "install" (
    echo This script is only called from ..\..\src\setup.bat
    @pause
    @goto:eof
)

:: Download the generic YOLO26 weight and the IPcam custom model set
"%venvPythonCmdPath%" "%moduleDirPath%\download_models.py" --dest "%moduleDirPath%\assets" --custom-dest "%moduleDirPath%\custom-models"

:: Fail the install if the primary CCTV model is missing after download
if not exist "%moduleDirPath%\custom-models\ipcam-combined.pt" set moduleInstallErrors=Failed to download CCTV models
