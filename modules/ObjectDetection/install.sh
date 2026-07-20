#!/bin/bash

# Module Installation script :::::::::::::::::::::::::::::::::::::::::::::::::::
#
#                         Object Detection
#
# This script is called from the ObjectDetection directory using:
#
#    bash ../../src/setup.sh
#
# The setup.sh script will find this install.sh file and execute it.
#
# For help with install scripts, notes on variables and methods available, tips,
# and explanations, see /modules/install_script_help.md

if [ "$1" != "install" ]; then
    read -t 3 -p "This script is only called from: bash ../../src/setup.sh"
    echo
    exit 1
fi

# Download the generic YOLO26 weight and the IPcam custom models
if [ "$moduleInstallErrors" = "" ]; then
    "${venvPythonCmdPath}" "${moduleDirPath}/download_models.py" \
        --dest "${moduleDirPath}/assets" \
        --custom-dest "${moduleDirPath}/custom-models"
fi

# Fail the install if the primary CCTV model is missing
if [ ! -f "${moduleDirPath}/custom-models/ipcam-combined.pt" ]; then
    moduleInstallErrors="Failed to download CCTV models"
fi
