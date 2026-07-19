:: ============================================================================
::
:: Improved Face Processing module install script for Windows
::
:: This script is called from ..\..\src\setup.bat
::
:: ============================================================================

@if "%1" NEQ "install" (
    echo This script is only called from ..\..\src\setup.bat
    @pause
    @goto:eof
)

:: Download the SCRFD (detector) + AdaFace (recognizer) model weights for both
:: tiers into the assets folder. download_models.py verifies SHA-256 and skips
:: files that already exist.
"%venvPythonCmdPath%" "%moduleDirPath%\intelligencelayer\download_models.py" --dest "%moduleDirPath%\assets"

:: Fail the install if the primary weights are missing after download
if not exist "%moduleDirPath%\assets\scrfd_10g.onnx"  set moduleInstallErrors=Failed to download face detection model
if not exist "%moduleDirPath%\assets\adaface_ir101.pt" set moduleInstallErrors=Failed to download face recognition model

:: Create the datastore folder for the face database
if not exist "%moduleDirPath%\datastore\" mkdir "%moduleDirPath%\datastore"
