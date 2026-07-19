#!/bin/bash

# Post Installation script ::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#                      Improved Face Processing
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

# (1) setuptools: insightface and torch tooling import pkg_resources, which
#     setuptools >= 81 removed. setup.sh upgrades setuptools to latest, so pin
#     it back below 81 to restore pkg_resources. (Needed for CPU and GPU.)
writeLine "Pinning setuptools < 81 (restores pkg_resources for insightface/torch)" $color_info
"${venvPythonCmdPath}" -m pip install "setuptools<81"
if [ $? -gt 0 ]; then moduleInstallErrors="Failed to pin setuptools<81"; fi

# (2) onnxruntime fix: insightface pulls the CPU onnxruntime package which
#     creates files in the shared onnxruntime namespace and shadows the CUDA
#     provider. On GPU installs we:
#       a) uninstall the CPU onnxruntime (removes the shadowing namespace files)
#       b) force-reinstall onnxruntime-gpu so its namespace files are restored
#          (uninstalling onnxruntime removes shared namespace entries; without
#          reinstalling onnxruntime-gpu, `import onnxruntime` resolves to an
#          empty/broken namespace)
#       c) upgrade to cuDNN 9.24 (fixes RTX 50-series / Blackwell sm_120;
#          torch 2.13+cu130 bundles the broken cuDNN 9.20).
if [ "${installGPU}" = "true" ] && [ "${hasCUDA}" = "true" ]; then
    writeLine "Removing CPU onnxruntime (shadows GPU CUDA provider)" $color_info
    "${venvPythonCmdPath}" -m pip uninstall -y onnxruntime >/dev/null 2>&1

    writeLine "Reinstalling onnxruntime-gpu (restores namespace after CPU uninstall)" $color_info
    "${venvPythonCmdPath}" -m pip install --force-reinstall onnxruntime-gpu
    if [ $? -gt 0 ]; then moduleInstallErrors="Failed to reinstall onnxruntime-gpu"; fi

    writeLine "Forcing cuDNN 9.24 (fixes RTX 50-series convolutions on cu130)" $color_info
    "${venvPythonCmdPath}" -m pip install --upgrade "nvidia-cudnn-cu13==9.24.0.43"
    if [ $? -gt 0 ]; then moduleInstallErrors="Failed to install cuDNN 9.24.0.43"; fi
fi
