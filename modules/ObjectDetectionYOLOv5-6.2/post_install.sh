#!/bin/bash

# Post Installation script ::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#                      Object Detection (YOLOv5 6.2)
#
# The setup.sh file will find this post_install.sh file and execute it.

if [ "$1" != "post-install" ]; then
    read -t 3 -p "This script is only called from: bash ../../src/setup.sh"
    echo
    exit 1
fi

# Both fixes below can't live in requirements.*.txt: the package installer skips
# any package already present by name (so version pins for torch's pre-pulled
# deps are ignored). We enforce the correct versions here, after pip install but
# before the self-test.

# (1) setuptools: the yolov5 package's code imports pkg_resources, which
#     setuptools >= 81 removed. setup.sh upgrades setuptools to latest, so pin
#     it back below 81 to restore pkg_resources. (Needed for CPU and GPU.)
writeLine "Pinning setuptools < 81 (restores pkg_resources for yolov5)" $color_info
"${venvPythonCmdPath}" -m pip install "setuptools<81"
if [ $? -gt 0 ]; then moduleInstallErrors="Failed to pin setuptools<81"; fi

# (2) cuDNN: torch 2.13.0+cu130 bundles cuDNN 9.20.0.48, which is broken on
#     RTX 50-series (Blackwell / sm_120) - every convolution fails with
#     CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH. cuDNN 9.24 fixes it. (GPU only.)
if [ "${installGPU}" = "true" ] && [ "${hasCUDA}" = "true" ]; then
    writeLine "Forcing cuDNN 9.24 (fixes RTX 50-series convolutions on cu130)" $color_info
    "${venvPythonCmdPath}" -m pip install --upgrade "nvidia-cudnn-cu13==9.24.0.43"
    if [ $? -gt 0 ]; then moduleInstallErrors="Failed to install cuDNN 9.24.0.43"; fi
fi
