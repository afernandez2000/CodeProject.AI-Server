import os
from codeproject_ai_sdk import ModuleOptions

# Threshold for the "accurate" tier (yolo26x, the extra-large model). Expressed
# in BINARY GiB: an "8 GiB" card reports ~8*1024**3 = 8.59e9 bytes, so a decimal
# 8e9 cutoff wrongly promoted 8 GiB cards (e.g. RTX 3070) to yolo26x, which is
# too heavy for them. Require ~12 GiB (11.5 with headroom) so 8-11 GiB cards get
# the lighter "balanced" tier (yolo26m) and only 12 GiB+ cards run yolo26x.
VRAM_CUTOFF_BYTES = int(11.5 * 1024**3)  # >= ~12 GiB -> accurate

# Generic YOLO26 family weight per tier (weight id may be adjusted per SPIKE.md).
GENERIC = {"accurate": "yolo26x.pt", "balanced": "yolo26m.pt", "fast": "yolo26n.pt"}
# CCTV family (yolov5 engine). Variants per tier; "" means no CCTV variant -> fall back.
CCTV = {
    "ipcam-combined": {"accurate": "ipcam-combined", "balanced": "ipcam-combined", "fast": ""},
}

def select_tier(has_cuda, vram_bytes, override):
    override = (override or "auto").lower()
    if override in ("accurate", "balanced", "fast"):
        return override
    if not has_cuda:
        return "fast"
    return "accurate" if vram_bytes >= VRAM_CUTOFF_BYTES else "balanced"

def resolve_default(default_model, tier):
    """Return (engine, weight_id) for the built-in detect route."""
    if default_model in CCTV:
        variant = CCTV[default_model].get(tier, "")
        if variant:
            return "yolov5", variant
        return "ultralytics", GENERIC["fast"]      # CPU/fast fallback to nano generic
    return "ultralytics", GENERIC.get(tier, GENERIC["fast"])

class Options:
    def __init__(self):
        self.app_dir           = os.path.normpath(ModuleOptions.getEnvVariable("APPDIR", os.getcwd()))
        self.models_dir        = os.path.normpath(ModuleOptions.getEnvVariable("MODELS_DIR", f"{self.app_dir}/assets"))
        self.custom_models_dir = os.path.normpath(ModuleOptions.getEnvVariable("CUSTOM_MODELS_DIR", f"{self.app_dir}/custom-models"))
        self.default_model     = ModuleOptions.getEnvVariable("DEFAULT_MODEL", "ipcam-combined")
        cf                     = ModuleOptions.getEnvVariable("CLASS_FILTER", "")
        self.class_filter      = [c.strip().lower() for c in cf.split(",") if c.strip()]
        self.min_confidence_default = 0.4
        override               = ModuleOptions.getEnvVariable("MODEL_TIER", "auto")

        self.use_CUDA, vram = False, 0
        if ModuleOptions.enable_GPU:
            try:
                import torch
                if torch.cuda.is_available():
                    self.use_CUDA = True
                    vram = torch.cuda.get_device_properties(0).total_memory
            except Exception:
                self.use_CUDA = False

        self.tier           = select_tier(self.use_CUDA, vram, override)
        self.default_engine, self.default_weight = resolve_default(self.default_model, self.tier)
