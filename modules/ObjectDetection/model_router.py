"""
Model router: routes a .pt checkpoint to the correct engine.

Routing logic
-------------
1. Pre-check: if the file exists locally, peek at the checkpoint and inspect
   the inner model's module namespace. A checkpoint whose inner model lives in
   ``models.yolo`` (the legacy YOLOv5 package namespace) is unambiguously a
   legacy weight and is sent directly to Yolov5Engine without ever touching
   Ultralytics, regardless of any monkeypatching in the environment.

2. Fallback: attempt UltralyticsEngine. If it raises and the exception text
   contains a known legacy marker (models.yolo, weights_only, Can't get
   attribute, No module named 'models', forwards compatible, ultralytics/yolov5)
   fall back to Yolov5Engine.

3. Cache by path: second call with the same path returns the same instance.

Why we need the pre-check
--------------------------
When the yolov5 package is already imported, its models.yolo namespace is
available in sys.modules. Ultralytics will therefore NOT raise a TypeError for
legacy YOLOv5 .pt files – it loads the model "successfully" using the yolov5
classes. The pre-check detects this before Ultralytics is involved.
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from engine_ultralytics import UltralyticsEngine
from engine_yolov5 import Yolov5Engine

_LEGACY_MARKERS = (
    "models.yolo",
    "weights_only",
    "Can't get attribute",
    "No module named 'models'",
    "forwards compatible",      # Ultralytics TypeError when yolov5 pkg not loaded
    "ultralytics/yolov5",       # same error message
)

# Module namespace prefix that marks a YOLOv5-trained checkpoint
_LEGACY_MODULE_PREFIX = "models."   # e.g. models.yolo, models.common


def _is_legacy_checkpoint(model_path: str) -> bool:
    """Return True if model_path is a local file containing a legacy YOLOv5 checkpoint."""
    if not os.path.isfile(model_path):
        return False
    try:
        ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
    except Exception:
        return False
    # ckpt is usually a dict with a 'model' key
    if isinstance(ckpt, dict):
        inner = ckpt.get("model", None)
    else:
        inner = ckpt
    if inner is None:
        return False
    mod = type(inner).__module__ or ""
    return mod.startswith(_LEGACY_MODULE_PREFIX)


class ModelRouter:
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        self._cache = {}

    def get(self, model_path):
        if model_path in self._cache:
            return self._cache[model_path]

        # Pre-check: inspect the checkpoint namespace before loading via Ultralytics.
        # This is necessary because yolov5's import registers models.yolo in
        # sys.modules, causing Ultralytics to silently accept legacy weights.
        if _is_legacy_checkpoint(model_path):
            eng = Yolov5Engine(model_path, self.use_cuda)
            self._cache[model_path] = eng
            return eng

        # Try Ultralytics; fall back for known legacy-format exceptions
        try:
            eng = UltralyticsEngine(model_path, self.use_cuda)
        except Exception as ex:
            if any(m in str(ex) for m in _LEGACY_MARKERS):
                eng = Yolov5Engine(model_path, self.use_cuda)
            else:
                raise

        self._cache[model_path] = eng
        return eng
