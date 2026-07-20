"""ImprovedFaceProcessing – module adapter + six command handlers.

Exposes:
  * Pipeline          – composable class (test-friendly, no SDK dependency)
  * ImprovedFace_adapter(ModuleRunner) – production adapter
"""

# ---------------------------------------------------------------------------
# CUDA library path shim (must run before any import that loads onnxruntime).
# Cross-platform (Linux re-exec / Windows add_dll_directory) — see _cuda_libpath.
# ---------------------------------------------------------------------------
import os, sys  # noqa: E401 – intentional early double-import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cuda_libpath import ensure_cuda_libpath
ensure_cuda_libpath()

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import time
import threading
import traceback

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the intelligencelayer directory is on sys.path so that importlib-
# loaded sub-modules (detector, recognizer, …) can import each other.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# Local sub-modules (tasks 3-7)
# ---------------------------------------------------------------------------
from detector   import ScrfdDetector
from align      import align_face
from recognizer import AdaFaceRecognizer
from gallery    import Gallery


# ===========================================================================
# In-memory gallery (used when db_path == ":memory:")
# ===========================================================================

class _MemoryGallery:
    """Thread-safe in-process gallery that never touches disk."""

    def __init__(self):
        self._lock = threading.Lock()
        self._store: dict[str, np.ndarray] = {}   # userid → (512,) float32

    # ---- Gallery-compatible API ----

    def add(self, userid: str, embedding: np.ndarray) -> None:
        with self._lock:
            self._store[userid] = embedding.astype("float32")

    def delete(self, userid: str) -> int:
        with self._lock:
            existed = userid in self._store
            self._store.pop(userid, None)
        return 1 if existed else 0

    def list_ids(self) -> list:
        with self._lock:
            return list(self._store.keys())

    def load(self) -> None:
        pass   # nothing to reload; state lives in self._store

    def match(self, embedding: np.ndarray, threshold: float):
        with self._lock:
            ids = list(self._store.keys())
            if not ids:
                return "unknown", 0.0
            mat = np.stack([self._store[uid] for uid in ids])  # (N,512)
        emb = embedding.astype("float32")
        raw_cos = mat @ emb
        i = int(raw_cos.argmax())
        sim_remapped = float((raw_cos[i] + 1.0) / 2.0)
        return (ids[i], sim_remapped) if sim_remapped >= threshold else ("unknown", sim_remapped)


# ===========================================================================
# Pipeline
# ===========================================================================

class Pipeline:
    """Composable face-processing pipeline (no ModuleRunner dependency).

    Parameters
    ----------
    detector_path   : path to SCRFD ONNX weight file
    recognizer_path : path to AdaFace .pt checkpoint
    arch            : AdaFace architecture string – 'ir_50' or 'ir_101'
    use_cuda        : whether to attempt GPU execution
    db_path         : SQLite path or ':memory:'
    threshold       : default cosine-similarity threshold for recognition
    """

    def __init__(self,
                 detector_path: str,
                 recognizer_path: str,
                 arch: str,
                 use_cuda: bool,
                 db_path: str,
                 threshold: float,
                 gpu_mem_limit_mb: int = 0):

        self.threshold = threshold

        # ---- build ONNX providers -----------------------------------------
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if use_cuda else ["CPUExecutionProvider"])

        # Optional GPU-memory cap for the SCRFD onnxruntime arena. When the GPU
        # is shared with another module, an uncapped arena (kNextPowerOfTwo +
        # EXHAUSTIVE cudnn search) grabs multi-GB sticky blocks and starves the
        # neighbour. gpu_mem_limit bounds it; kSameAsRequested + HEURISTIC stop
        # it over-allocating. None = default (unbounded) behaviour.
        provider_options = None
        if use_cuda and gpu_mem_limit_mb and gpu_mem_limit_mb > 0:
            provider_options = [
                {"gpu_mem_limit": int(gpu_mem_limit_mb) * 1024 * 1024,
                 "arena_extend_strategy": "kSameAsRequested",
                 "cudnn_conv_algo_search": "HEURISTIC"},
                {},  # CPUExecutionProvider — no options
            ]

        # ---- detector (ONNX / SCRFD) ---------------------------------------
        self.detector = ScrfdDetector(detector_path, providers, det_size=640,
                                      provider_options=provider_options)

        # ---- recognizer (PyTorch / AdaFace) --------------------------------
        # GPU-OOM → CPU fallback
        try:
            self.recognizer = AdaFaceRecognizer(recognizer_path, arch, use_cuda)
        except Exception as ex:
            if use_cuda and "out of memory" in str(ex).lower():
                self.recognizer = AdaFaceRecognizer(recognizer_path, arch, False)
            else:
                raise

        # ---- gallery -------------------------------------------------------
        if db_path == ":memory:":
            self.gallery = _MemoryGallery()
        else:
            # Gallery.__init__ calls os.makedirs(dirname(db_path)) which fails
            # if the parent directory doesn't exist yet – create it first.
            parent = os.path.dirname(db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self.gallery = Gallery(db_path)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def detect(self, bgr, threshold: float = 0.4) -> dict:
        """Detect faces in a BGR NumPy image.

        Parameters
        ----------
        bgr       : BGR NumPy image
        threshold : minimum SCRFD detection score (0–1) to keep a face.
                    Distinct from the cosine-similarity threshold used by
                    ``recognize``/``match``.

        Returns a FaceProcessing-shaped dict::

            {success, count, predictions:[{confidence,x_min,y_min,x_max,y_max}],
             inferenceMs}
        """
        if bgr is None:
            return {"success": False, "error": "No image supplied", "predictions": [], "count": 0, "inferenceMs": 0}
        try:
            t0 = time.perf_counter()
            faces = self.detector.detect(bgr)
            inference_ms = int((time.perf_counter() - t0) * 1000)

            # Apply detection-score threshold (SCRFD det_score, ~0.5–0.9 scale)
            faces = [f for f in faces if f.score >= threshold]

            predictions = [
                {
                    "confidence": f.score,
                    "x_min": f.bbox[0],
                    "y_min": f.bbox[1],
                    "x_max": f.bbox[2],
                    "y_max": f.bbox[3],
                }
                for f in faces
            ]

            n = len(predictions)
            return {
                "success": True,
                "predictions": predictions,
                "count": n,
                "message": "Found 1 face" if n == 1 else f"Found {n} faces",
                "inferenceMs": inference_ms,
            }
        except Exception as ex:
            trace = "".join(traceback.TracebackException.from_exception(ex).format())
            return {
                "success": False,
                "error": "An Error occurred during processing",
                "err_trace": trace,
                "predictions": [],
                "count": 0,
                "inferenceMs": 0,
            }

    def register(self, bgr, userid: str) -> dict:
        """Register a face for *userid*.

        Requires exactly one detectable face in the image.

        Returns a FaceProcessing-shaped dict::

            {success, message, inferenceMs}
        """
        if bgr is None:
            return {"success": False, "error": "No image supplied", "inferenceMs": 0}
        try:
            t0 = time.perf_counter()
            faces = self.detector.detect(bgr)
            if not faces:
                return {"success": False, "error": "No face detected", "inferenceMs": int((time.perf_counter() - t0)*1000)}

            # Use the highest-scoring face
            face = max(faces, key=lambda f: f.score)
            if face.kps is None:
                # Fall back to simple crop + resize when keypoints unavailable
                x1, y1, x2, y2 = face.bbox
                crop = bgr[max(0, y1):y2, max(0, x1):x2]
                crop = cv2.resize(crop, (112, 112))
            else:
                crop = align_face(bgr, face.kps, image_size=112)

            embedding = self.recognizer.embed(crop)
            is_update = userid in self.gallery.list_ids()
            self.gallery.add(userid, embedding)
            self.gallery.load()  # refresh in-memory index

            inference_ms = int((time.perf_counter() - t0) * 1000)
            return {
                "success": True,
                "message": "face updated" if is_update else "face added",
                "inferenceMs": inference_ms,
            }
        except Exception as ex:
            trace = "".join(traceback.TracebackException.from_exception(ex).format())
            return {
                "success": False,
                "error": "An Error occurred during processing",
                "err_trace": trace,
                "inferenceMs": 0,
            }

    def recognize(self, bgr, threshold: float | None = None) -> dict:
        """Recognise all faces in the image.

        Returns a FaceProcessing-shaped dict::

            {success, count, predictions:[{confidence,userid,x_min,y_min,x_max,y_max}],
             message, inferenceMs}
        """
        if threshold is None:
            threshold = self.threshold

        if bgr is None:
            return {"success": False, "error": "No image supplied", "predictions": [], "count": 0, "inferenceMs": 0}
        try:
            t0 = time.perf_counter()
            faces = self.detector.detect(bgr)
            inference_ms = int((time.perf_counter() - t0) * 1000)

            if not faces:
                return {
                    "success": False,
                    "error": "No face found in image",
                    "predictions": [],
                    "count": 0,
                    "inferenceMs": inference_ms,
                }

            predictions = []
            found_known = False

            for face in faces:
                if face.kps is None:
                    x1, y1, x2, y2 = face.bbox
                    crop = bgr[max(0, y1):y2, max(0, x1):x2]
                    crop = cv2.resize(crop, (112, 112))
                else:
                    crop = align_face(bgr, face.kps, image_size=112)

                t1 = time.perf_counter()
                embedding = self.recognizer.embed(crop)
                inference_ms += int((time.perf_counter() - t1) * 1000)

                userid, similarity = self.gallery.match(embedding, threshold)

                if userid != "unknown":
                    found_known = True
                    confidence = float(similarity)
                else:
                    confidence = 0.0

                x1, y1, x2, y2 = face.bbox
                predictions.append({
                    "confidence": confidence,
                    "userid": userid,
                    "x_min": max(0, x1),
                    "y_min": max(0, y1),
                    "x_max": max(0, x2),
                    "y_max": max(0, y2),
                })

            message = "A face was recognised" if found_known else "No known faces"
            return {
                "success": True,
                "predictions": predictions,
                "count": len(predictions),
                "message": message,
                "inferenceMs": inference_ms,
            }
        except Exception as ex:
            trace = "".join(traceback.TracebackException.from_exception(ex).format())
            return {
                "success": False,
                "error": "An Error occurred during processing",
                "err_trace": trace,
                "predictions": [],
                "count": 0,
                "inferenceMs": 0,
            }

    def match(self, bgr1, bgr2) -> dict:
        """Compare the top face in each image via cosine similarity.

        Returns a FaceProcessing-shaped dict::

            {success, similarity, inferenceMs}
        """
        if bgr1 is None or bgr2 is None:
            return {"success": False, "error": "One or both images are missing", "inferenceMs": 0}
        try:
            t0 = time.perf_counter()

            faces1 = self.detector.detect(bgr1)
            faces2 = self.detector.detect(bgr2)

            inference_ms = int((time.perf_counter() - t0) * 1000)

            if not faces1 or not faces2:
                return {
                    "success": False,
                    "error": "No face found in one or both images",
                    "inferenceMs": inference_ms,
                }

            def _best_crop(bgr, faces):
                face = max(faces, key=lambda f: f.score)
                if face.kps is None:
                    x1, y1, x2, y2 = face.bbox
                    crop = bgr[max(0, y1):y2, max(0, x1):x2]
                    return cv2.resize(crop, (112, 112))
                return align_face(bgr, face.kps, image_size=112)

            crop1 = _best_crop(bgr1, faces1)
            crop2 = _best_crop(bgr2, faces2)

            t1 = time.perf_counter()
            emb1 = self.recognizer.embed(crop1)
            emb2 = self.recognizer.embed(crop2)
            inference_ms += int((time.perf_counter() - t1) * 1000)

            # Both embeddings are L2-normalised → dot product == cosine similarity
            # Map from [-1, 1] to [0, 1] to match FaceProcessing convention
            raw_cos = float(np.dot(emb1, emb2))
            similarity = (raw_cos + 1.0) / 2.0

            return {
                "success": True,
                "similarity": similarity,
                "inferenceMs": inference_ms,
            }
        except Exception as ex:
            trace = "".join(traceback.TracebackException.from_exception(ex).format())
            return {
                "success": False,
                "error": "An Error occurred during processing",
                "err_trace": trace,
                "inferenceMs": 0,
            }


# ===========================================================================
# ModuleRunner adapter
# ===========================================================================

try:
    from codeproject_ai_sdk import RequestData, ModuleRunner, LogMethod, JSON
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False


def _pil_to_bgr(pil_img) -> np.ndarray:
    """Convert a PIL image (RGB or RGBA) to a BGR numpy array."""
    if pil_img is None:
        return None
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


if _SDK_AVAILABLE:

    class ImprovedFace_adapter(ModuleRunner):

        def __init__(self):
            super().__init__()
            self._pipeline: Pipeline | None = None
            self._update_faces_active = False
            self._num_items_found = 0

        # ------------------------------------------------------------------ #
        # Lifecycle
        # ------------------------------------------------------------------ #

        def initialise(self) -> None:
            from options import Options
            opts = Options()

            self.can_use_GPU = self.system_info.hasTorchCuda

            # GPU-OOM fallback is handled inside Pipeline.__init__
            try:
                self._pipeline = Pipeline(
                    detector_path    = opts.detector_path,
                    recognizer_path  = opts.recognizer_path,
                    arch             = "ir_101" if opts.tier == "accurate" else "ir_50",
                    use_cuda         = opts.use_cuda,
                    db_path          = opts.db_path,
                    threshold        = opts.threshold,
                    gpu_mem_limit_mb = opts.gpu_mem_limit_mb,
                )
            except Exception as ex:
                # If GPU OOM at pipeline level, retry CPU
                if opts.use_cuda and "out of memory" in str(ex).lower():
                    opts.use_cuda = False
                    self._pipeline = Pipeline(
                        detector_path    = opts.detector_path,
                        recognizer_path  = opts.recognizer_path,
                        arch             = "ir_101" if opts.tier == "accurate" else "ir_50",
                        use_cuda         = False,
                        db_path          = opts.db_path,
                        threshold        = opts.threshold,
                        gpu_mem_limit_mb = 0,   # CPU retry: no GPU cap needed
                    )
                else:
                    raise

            if self._pipeline is not None:
                self.inference_device  = "GPU" if opts.use_cuda else "CPU"
                self.inference_library = "CUDA" if opts.use_cuda else ""

                # Surface the resolved tier / models / device so it is visible in
                # the server log and the dashboard module status.
                self._active_tier       = opts.tier
                self._active_detector   = os.path.basename(opts.detector_path)
                self._active_recognizer = "ir_101" if opts.tier == "accurate" else "ir_50"
                self._active_device     = "GPU" if opts.use_cuda else "CPU"
                # print() reaches the server log verbatim (module stdout is captured).
                print(f"Improved Face Processing ACTIVE: tier={self._active_tier} "
                      f"detector={self._active_detector} "
                      f"recognizer=AdaFace-{self._active_recognizer} "
                      f"device={self._active_device}", flush=True)
                self.log(LogMethod.Info | LogMethod.Server, {
                    "filename": __file__,
                    "method":   "initialise",
                    "loglevel": "information",
                    "message":  f"Improved Face Processing active: tier={self._active_tier}, "
                                f"detector={self._active_detector}, "
                                f"recognizer=AdaFace-{self._active_recognizer}, "
                                f"device={self._active_device}",
                })

            # Load existing gallery into memory
            if isinstance(self._pipeline.gallery, Gallery):
                self._pipeline.gallery.load()

            # Refresh gallery every 5 seconds (mirrors FaceProcessing pattern)
            self._update_faces_active = True
            t = threading.Thread(target=self._refresh_gallery, args=(5,), daemon=True)
            t.start()

            self._num_items_found = 0

        def cleanup(self) -> None:
            self._update_faces_active = False

        def _refresh_gallery(self, delay: int) -> None:
            while self._update_faces_active:
                try:
                    if self._pipeline and isinstance(self._pipeline.gallery, Gallery):
                        self._pipeline.gallery.load()
                except Exception:
                    pass
                time.sleep(delay)

        # ------------------------------------------------------------------ #
        # Request routing
        # ------------------------------------------------------------------ #

        def process(self, data: RequestData) -> JSON:
            command = data.command
            response = {"success": False, "error": "Unknown command"}

            start_time = time.perf_counter()

            if command == "detect":
                response = self._do_detect(data)
            elif command == "register":
                response = self._do_register(data)
            elif command == "list":
                response = self._do_list(data)
            elif command == "delete":
                response = self._do_delete(data)
            elif command == "recognize":
                response = self._do_recognize(data)
            elif command == "match":
                response = self._do_match(data)

            if response.get("success"):
                response["processMs"] = int((time.perf_counter() - start_time) * 1000)
            else:
                message = response.get("error", "Error occurred")
                if response.get("err_trace", ""):
                    message += ": " + response["err_trace"]
                self.log(LogMethod.Error | LogMethod.Server, {
                    "filename": __file__,
                    "method": command,
                    "message": message,
                    "loglevel": "error",
                })

            return response

        # ------------------------------------------------------------------ #
        # Status / statistics
        # ------------------------------------------------------------------ #

        def status(self) -> JSON:
            data = super().status()
            data["numItemsFound"] = self._num_items_found
            data["tier"]            = getattr(self, "_active_tier", None)
            data["activeDetector"]  = getattr(self, "_active_detector", None)
            data["activeRecognizer"] = getattr(self, "_active_recognizer", None)
            data["device"]          = getattr(self, "_active_device", None)
            return data

        def update_statistics(self, response):
            super().update_statistics(response)
            if response.get("success") and "predictions" in response:
                self._num_items_found += len(response["predictions"])

        # ------------------------------------------------------------------ #
        # Self-test
        # ------------------------------------------------------------------ #

        def selftest(self) -> JSON:
            file_name = os.path.join("test", "person.jpg")

            request_data = RequestData()
            request_data.queue   = self.queue_name
            request_data.command = "detect"
            request_data.add_file(file_name)
            request_data.add_value("min_confidence", 0.4)

            result = self.process(request_data)
            print(f"Info: Self-test for {self.module_id}. Success: {result['success']}")

            return {"success": result["success"], "message": "Face detection test successful"}

        # ------------------------------------------------------------------ #
        # Command implementations (bridge from RequestData → Pipeline)
        # ------------------------------------------------------------------ #

        def _do_detect(self, data: RequestData) -> JSON:
            try:
                threshold = float(data.get_value("min_confidence", "0.4"))
                bgr = _pil_to_bgr(data.get_image(0))
                return self._pipeline.detect(bgr, threshold)
            except Exception as ex:
                trace = "".join(traceback.TracebackException.from_exception(ex).format())
                return {"success": False, "error": "An Error occurred during processing", "err_trace": trace}

        def _do_register(self, data: RequestData) -> JSON:
            try:
                userid = data.get_value("userid")
                if not userid:
                    return {"success": False, "error": "userid is required"}

                num_files = len(data.files) if data.files else 0
                if num_files == 0:
                    return {"success": False, "error": "At least one image is required"}

                # Register all provided images; accumulate embeddings then average
                # For simplicity, register each image independently (last one wins
                # in gallery.add, which matches FaceProcessing behaviour of using
                # the mean of all provided images).
                embeddings = []
                inference_ms = 0

                for i in range(num_files):
                    bgr = _pil_to_bgr(data.get_image(i))
                    if bgr is None:
                        continue
                    t0 = time.perf_counter()
                    faces = self._pipeline.detector.detect(bgr)
                    inference_ms += int((time.perf_counter() - t0) * 1000)
                    if not faces:
                        continue
                    face = max(faces, key=lambda f: f.score)
                    if face.kps is None:
                        x1, y1, x2, y2 = face.bbox
                        crop = bgr[max(0, y1):y2, max(0, x1):x2]
                        crop = cv2.resize(crop, (112, 112))
                    else:
                        crop = align_face(bgr, face.kps, image_size=112)
                    t1 = time.perf_counter()
                    emb = self._pipeline.recognizer.embed(crop)
                    inference_ms += int((time.perf_counter() - t1) * 1000)
                    embeddings.append(emb)

                if not embeddings:
                    return {"success": False, "error": "No face detected", "inferenceMs": inference_ms}

                mean_emb = np.mean(np.stack(embeddings), axis=0).astype("float32")
                norm = np.linalg.norm(mean_emb)
                if norm > 0:
                    mean_emb /= norm
                is_update = userid in self._pipeline.gallery.list_ids()
                self._pipeline.gallery.add(userid, mean_emb)
                self._pipeline.gallery.load()

                msg = "face updated" if is_update else "face added"
                return {"success": True, "message": msg, "inferenceMs": inference_ms}

            except Exception as ex:
                trace = "".join(traceback.TracebackException.from_exception(ex).format())
                return {"success": False, "error": "An Error occurred during processing", "err_trace": trace}

        def _do_list(self, data: RequestData) -> JSON:
            try:
                faces = self._pipeline.gallery.list_ids()
                return {"success": True, "faces": faces}
            except Exception as ex:
                trace = "".join(traceback.TracebackException.from_exception(ex).format())
                return {"success": False, "error": "An Error occurred during processing", "err_trace": trace}

        def _do_delete(self, data: RequestData) -> JSON:
            try:
                userid = data.get_value("userid")
                if not userid:
                    return {"success": False, "error": "userid is required"}
                self._pipeline.gallery.delete(userid)
                self._pipeline.gallery.load()
                return {"success": True}
            except Exception as ex:
                trace = "".join(traceback.TracebackException.from_exception(ex).format())
                return {"success": False, "error": "An Error occurred during processing", "err_trace": trace}

        def _do_recognize(self, data: RequestData) -> JSON:
            try:
                threshold = float(data.get_value("min_confidence", str(self._pipeline.threshold)))
                bgr = _pil_to_bgr(data.get_image(0))
                return self._pipeline.recognize(bgr, threshold)
            except Exception as ex:
                trace = "".join(traceback.TracebackException.from_exception(ex).format())
                return {"success": False, "error": "An Error occurred during processing", "err_trace": trace}

        def _do_match(self, data: RequestData) -> JSON:
            try:
                bgr1 = _pil_to_bgr(data.get_image(0))
                bgr2 = _pil_to_bgr(data.get_image(1))
                return self._pipeline.match(bgr1, bgr2)
            except Exception as ex:
                trace = "".join(traceback.TracebackException.from_exception(ex).format())
                return {"success": False, "error": "An Error occurred during processing", "err_trace": trace}


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    if _SDK_AVAILABLE:
        ImprovedFace_adapter().start_loop()
    else:
        raise RuntimeError("codeproject_ai_sdk is not installed; cannot start the module loop.")
