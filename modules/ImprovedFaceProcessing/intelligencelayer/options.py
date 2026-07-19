import os
from codeproject_ai_sdk import ModuleOptions

VRAM_CUTOFF_BYTES = 6 * 10**9  # ~6 GB → accurate tier

def select_tier(has_cuda, vram_bytes, override):
    override = (override or "auto").lower()
    if override in ("accurate", "fast"):
        return override
    if has_cuda and vram_bytes >= VRAM_CUTOFF_BYTES:
        return "accurate"
    return "fast"

def onnx_providers(has_cuda):
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if has_cuda else ["CPUExecutionProvider"]

# Per-tier model filenames + recognition threshold on the remapped (cos+1)/2 scale [0,1].
# Raw cosine 0.28 → remapped (0.28+1)/2 = 0.64.
TIER = {
    "accurate": {"detector": "scrfd_10g.onnx",  "recognizer": "adaface_ir101.pt", "det_size": 640, "threshold": 0.64},
    "fast":     {"detector": "scrfd_2.5g.onnx", "recognizer": "adaface_ir50.pt",  "det_size": 640, "threshold": 0.64},
}

class Options:
    def __init__(self):
        self.enable_gpu  = ModuleOptions.enable_GPU
        self.app_dir     = os.path.normpath(ModuleOptions.getEnvVariable(
                               "APPDIR", os.path.join(os.getcwd())))
        self.models_dir  = os.path.normpath(ModuleOptions.getEnvVariable(
                               "MODELS_DIR", os.path.join(self.app_dir, "assets")))
        self.data_dir    = os.path.normpath(ModuleOptions.getEnvVariable(
                               "DATA_DIR", "/etc/codeproject/ai"))
        override         = ModuleOptions.getEnvVariable("MODEL_TIER", "auto")

        self.use_cuda, vram = False, 0
        if self.enable_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.use_cuda = True
                    vram = torch.cuda.get_device_properties(0).total_memory
            except Exception:
                self.use_cuda = False

        self.tier           = select_tier(self.use_cuda, vram, override)
        cfg                 = TIER[self.tier]
        self.det_size       = cfg["det_size"]
        self.threshold      = cfg["threshold"]
        self.providers      = onnx_providers(self.use_cuda)
        self.detector_path  = os.path.join(self.models_dir, cfg["detector"])
        self.recognizer_path= os.path.join(self.models_dir, cfg["recognizer"])
        self.db_path        = os.path.join(self.data_dir, "improved_faceembedding.db")
