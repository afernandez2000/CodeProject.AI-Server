"""
detect_adapter.py — ObjectDetection module adapter (Blue Iris parity).

Response shape matches modules/ObjectDetectionYOLOv5-6.2 byte-for-byte:
  detect/custom: {success, count, predictions:[{confidence,label,x_min,y_min,x_max,y_max}],
                  inferenceMs, processMs, message}
  list-custom:   {success, models:[...]}
"""

import os
import sys
import time
from time import perf_counter

# Ensure local package imports work regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2

from engine_ultralytics import UltralyticsEngine
from engine_yolov5 import Yolov5Engine
from model_router import ModelRouter


# ---------------------------------------------------------------------------
# Detector — the testable business-logic layer
# ---------------------------------------------------------------------------

class Detector:
    def __init__(self, opts):
        self.opts = opts
        self.router = ModelRouter(opts.use_CUDA)
        self._default_engine = self._build_default_engine(opts.use_CUDA)
        self._model_names: list = []
        self._models_last_checked = None

    # ------------------------------------------------------------------
    # Engine construction with GPU-OOM → CPU fallback (mirrors FaceProcessing)
    # ------------------------------------------------------------------

    def _build_default_engine(self, use_cuda):
        try:
            return self._make_engine(use_cuda)
        except Exception as ex:
            if "out of memory" in str(ex).lower():
                self.opts.use_CUDA = False
                self.router = ModelRouter(False)
                return self._make_engine(False)
            raise

    def _make_engine(self, use_cuda):
        opts = self.opts
        if opts.default_engine == "yolov5":
            # Legacy CCTV family: weight lives in custom-models directory
            path = os.path.join(opts.custom_models_dir, opts.default_weight)
            if not path.endswith(".pt"):
                path += ".pt"
            return Yolov5Engine(path, use_cuda)
        else:
            # Ultralytics family: weight id (auto-download) or local .pt
            return UltralyticsEngine(opts.default_weight, use_cuda)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format(self, preds, t0, infer_ms):
        """Apply class_filter and build the Blue-Iris-shaped response dict."""
        cf = self.opts.class_filter
        if cf:
            preds = [p for p in preds if p["label"].lower() in cf]
        count = len(preds)
        label_str = ", ".join(p["label"] for p in preds) if preds else "nothing"
        return {
            "success": True,
            "count": count,
            "predictions": preds,
            "inferenceMs": int(infer_ms),
            "processMs": int((perf_counter() - t0) * 1000),
            "message": f"Found {count} object{'s' if count != 1 else ''}: {label_str}",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, bgr, conf):
        """Run the default engine on a BGR ndarray."""
        t0 = perf_counter()
        t_infer = perf_counter()
        preds = self._default_engine.detect(bgr, conf)
        infer_ms = (perf_counter() - t_infer) * 1000
        return self._format(preds, t0, infer_ms)

    def custom(self, model_name, bgr, conf):
        """Run a custom .pt model (looked up from custom_models_dir)."""
        # Map 'general' → 'ipcam-general' (same as YOLOv5-6.2 adapter)
        if model_name == "general":
            model_name = "ipcam-general"

        path = os.path.join(self.opts.custom_models_dir, model_name + ".pt")
        if not os.path.isfile(path):
            return {"success": False, "error": f"model {model_name} not found"}

        t0 = perf_counter()
        t_infer = perf_counter()
        preds = self.router.get(path).detect(bgr, conf)
        infer_ms = (perf_counter() - t_infer) * 1000
        return self._format(preds, t0, infer_ms)

    def list_models(self):
        """Scan custom_models_dir for *.pt files (excluding yolov5* weights).

        Result is cached for at least 60 seconds to avoid repeated directory scans.
        """
        if self._models_last_checked is None or (time.time() - self._models_last_checked) >= 60:
            models_dir = self.opts.custom_models_dir
            try:
                self._model_names = [
                    entry.name[:-3]
                    for entry in os.scandir(models_dir)
                    if entry.is_file()
                    and entry.name.endswith(".pt")
                    and not entry.name.startswith("yolov5")
                ]
            except FileNotFoundError:
                self._model_names = []
            self._models_last_checked = time.time()
        return {"success": True, "models": self._model_names}


# ---------------------------------------------------------------------------
# ModuleRunner adapter
# ---------------------------------------------------------------------------

try:
    from codeproject_ai_sdk import JSON, ModuleRunner, LogMethod, LogVerbosity, RequestData
    from PIL import Image
    from options import Options

    _SDK_AVAILABLE = True

    class ObjectDetection_adapter(ModuleRunner):

        def __init__(self):
            super().__init__()
            self._num_items_found = 0
            self._histogram: dict = {}

        def initialise(self):
            self.opts = Options()

            # When launched standalone (not by server) adopt the shared queue name
            if not self.launched_by_server:
                self.queue_name = "objectdetection_queue"

            # Honour server GPU capability flags
            if self.opts.use_CUDA:
                self.opts.use_CUDA = self.system_info.hasTorchCuda

            self._detector = Detector(self.opts)

            # Report GPU status to the server dashboard (mirrors ObjectDetectionYOLOv5-6.2)
            self.can_use_GPU = self.system_info.hasTorchCuda or self.system_info.hasTorchMPS
            if self.opts.use_CUDA:
                self.inference_device  = "GPU"
                self.inference_library = "CUDA"

            self._num_items_found = 0
            self._histogram: dict = {}

        def process(self, data: RequestData) -> JSON:
            response = None

            if data.command == "list-custom":
                response = self._detector.list_models()

            elif data.command == "detect":
                threshold = float(data.get_value("min_confidence", str(self.opts.min_confidence_default)))
                bgr = self._pil_to_bgr(data.get_image(0))
                response = self._detector.detect(bgr, threshold)

            elif data.command == "custom":
                threshold = float(data.get_value("min_confidence", str(self.opts.min_confidence_default)))
                bgr = self._pil_to_bgr(data.get_image(0))
                model_name = "general"
                if data.segments and data.segments[0]:
                    model_name = data.segments[0]
                response = self._detector.custom(model_name, bgr, threshold)

            else:
                response = {"success": False, "error": f"unsupported command: {data.command}"}
                self.report_error(None, __file__, f"Unknown command {data.command}")

            return response

        def status(self) -> JSON:
            status_data = super().status()
            status_data["numItemsFound"] = self._num_items_found
            status_data["histogram"] = self._histogram
            return status_data

        def update_statistics(self, response):
            super().update_statistics(response)
            if response.get("success") and "predictions" in response:
                predictions = response["predictions"]
                self._num_items_found += len(predictions)
                for prediction in predictions:
                    label = prediction["label"]
                    self._histogram[label] = self._histogram.get(label, 0) + 1

        def selftest(self) -> JSON:
            file_name = os.path.join("test", "objects.jpg")
            request_data = RequestData()
            request_data.queue = self.queue_name
            request_data.command = "detect"
            request_data.add_file(file_name)
            request_data.add_value("min_confidence", self.opts.min_confidence_default)
            result = self.process(request_data)
            print(f"Info: Self-test for {self.module_id}. Success: {result['success']}")
            return {"success": result["success"], "message": "Object detection test successful"}

        # ------------------------------------------------------------------
        # Helper: PIL Image → BGR ndarray
        # ------------------------------------------------------------------

        @staticmethod
        def _pil_to_bgr(img: Image.Image) -> np.ndarray:
            """Convert a PIL Image (any mode) to a BGR uint8 ndarray for the engines."""
            rgb = np.array(img.convert("RGB"), dtype=np.uint8)
            return rgb[:, :, ::-1].copy()  # RGB → BGR

except ImportError:
    # SDK not available — e.g. when running unit tests without the full server env.
    # Detector is still importable; ObjectDetection_adapter will simply be absent.
    pass


if __name__ == "__main__":
    ObjectDetection_adapter().start_loop()
