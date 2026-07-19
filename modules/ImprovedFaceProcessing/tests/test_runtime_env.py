import os, glob, importlib
VENV = "runtimes/bin/ubuntu/python311/venv"

def _add_cu13_libpath():
    libdir = os.path.abspath(os.path.join(VENV, "lib/python3.11/site-packages/nvidia/cu13/lib"))
    if os.path.isdir(libdir):
        os.environ["LD_LIBRARY_PATH"] = libdir + ":" + os.environ.get("LD_LIBRARY_PATH", "")

def test_onnxruntime_and_insightface_importable():
    _add_cu13_libpath()
    import onnxruntime as ort, insightface  # noqa
    provs = ort.get_available_providers()
    # On a CUDA box, CUDA provider must be present; on CPU-only it may be CPU-only.
    import torch
    if torch.cuda.is_available():
        assert "CUDAExecutionProvider" in provs, provs
