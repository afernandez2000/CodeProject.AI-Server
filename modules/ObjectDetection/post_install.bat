:: ============================================================================
::
:: Object Detection module post-install script for Windows
::
:: This script is called from ..\..\src\setup.bat after the Python packages
:: have been installed, but before the module self-test.
::
:: Mirrors post_install.sh. NOTE: developed/tested on Linux; the Windows path is
:: best-effort and should be verified on the target machine.
::
:: ============================================================================

@if "%1" NEQ "post-install" (
    echo This script is only called from ..\..\src\setup.bat
    @pause
    @goto:eof
)

:: (1) The yolov5 package imports pkg_resources, which setuptools >= 81 removed.
::     setup.bat upgrades setuptools to latest, so pin it back below 81 to
::     restore pkg_resources. (Needed for CPU and GPU.)
echo Pinning setuptools ^< 81 (restores pkg_resources for yolov5)
"%venvPythonCmdPath%" -m pip install "setuptools<81"

:: (2) GPU only: torch 2.13.0+cu130 bundles a broken cuDNN 9.20 for RTX 50-series
::     (sm_120) - every convolution fails with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH.
::     cuDNN 9.24 fixes it.
if /i "%installGPU%"=="true" (
    if /i "%hasCUDA%"=="true" (
        echo Forcing cuDNN 9.24 ^(fixes RTX 50-series convolutions on cu130^)
        "%venvPythonCmdPath%" -m pip install --upgrade "nvidia-cudnn-cu13==9.24.0.43"
    )
)
