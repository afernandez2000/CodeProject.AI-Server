#!/bin/bash

# Development mode setup script ::::::::::::::::::::::::::::::::::::::::::::::
#
#                        ImprovedFaceProcessing
#
# This script is called from the ImprovedFaceProcessing directory using:
#
#    bash ../../src/setup.sh
#
# The setup.sh script will find this install.sh file and execute it.
#
# For help with install scripts, notes on variables and methods available, tips,
# and explanations, see /src/modules/install_script_help.md

if [ "$1" != "install" ]; then
    read -t 3 -p "This script is only called from: bash ../../src/setup.sh"
    echo
    exit 1
fi

# Download model weights into assets (both tiers so the module can switch at runtime)
if [ -f "${moduleDirPath}/intelligencelayer/download_models.py" ]; then
    writeLine "Downloading ImprovedFaceProcessing model weights..." $color_info
    "${venvPythonCmdPath}" "${moduleDirPath}/intelligencelayer/download_models.py" \
        --dest "${moduleDirPath}/assets" || \
        moduleInstallErrors="Failed to download face models"
fi

# Verify that all four canonical model files were actually downloaded and are non-empty.
_required_assets=(
    "${moduleDirPath}/assets/scrfd_10g.onnx"
    "${moduleDirPath}/assets/scrfd_2.5g.onnx"
    "${moduleDirPath}/assets/adaface_ir101.pt"
    "${moduleDirPath}/assets/adaface_ir50.pt"
)
for _asset in "${_required_assets[@]}"; do
    if [ ! -s "$_asset" ]; then
        moduleInstallErrors="Required model asset missing or empty: $_asset"
        break
    fi
done
