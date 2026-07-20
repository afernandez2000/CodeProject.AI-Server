# Object Detection Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new `ObjectDetection` module (v1.0.0) for Blue Iris 6: a CCTV-domain model default (ipcam-combined) via a yolov5 engine, YOLO26 as a switchable modern generic detector via Ultralytics, full legacy + modern custom-model support, hardware auto-tiering, cross-platform.

**Architecture:** A stateless CodeProject.AI module in `modules/ObjectDetection/`. A `ModuleRunner` adapter dispatches the three Blue-Iris commands (`detect`, `custom`, `list-custom`). Two torch-based inference engines share one venv: **yolov5==6.2.3** (default CCTV model + legacy custom `.pt`) and **Ultralytics** (YOLO26 generic default + modern custom `.pt`). `options.py` selects the default model family and the size/variant by detected GPU/VRAM. Custom requests auto-route to the engine that can load the file.

**Tech Stack:** Python 3.11, torch 2.13.0+cu130 / +cpu, ultralytics, yolov5==6.2.3, opencv, numpy, CodeProject.AI Python SDK, pytest.

## Global Constraints

- Module ID `ObjectDetection`; Name "Object Detection"; Version `1.0.0`; Runtime `python3.11`, RuntimeLocation `Shared`.
- Queue `objectdetection_queue`. Routes: `vision/detection` (`detect`), `vision/custom/<model>` (`custom`), `vision/custom/list` (`list-custom`).
- Response JSON byte-compatible with `modules/ObjectDetectionYOLOv5-6.2`: `{ success, count, predictions:[{confidence,label,x_min,y_min,x_max,y_max}], inferenceMs, processMs, message | error }`.
- `DEFAULT_MODEL` setting: `ipcam-combined` (default) or `yolo26`. `MODEL_TIER`: `auto|accurate|balanced|fast`.
- Tiering: accurate=GPU≥~8GB, balanced=GPU<~8GB, fast=CPU. CCTV family tiers to available IPcam variants; where a light CCTV variant is missing on the fast/CPU tier, fall back to YOLO26-N.
- GPU requirements: `torch==2.13.0+cu130`, `torchvision==0.28.0+cu130` (index cu130), `ultralytics`, `yolov5==6.2.3`. CPU: `torch==2.13.0+cpu`, `torchvision==0.28.0+cpu` (index cpu), same packages.
- `post_install.{sh,bat}` MUST: `pip install "setuptools<81"`; on GPU `pip install --upgrade "nvidia-cudnn-cu13==9.24.0.43"`. NO onnxruntime; NO `_cuda_libpath` shim.
- Successor to `ObjectDetectionYOLOv5-6.2` (only one enabled at a time; server route-ownership fix already in place).
- Weights are NOT committed (`.gitignore`). Test interpreter: `runtimes/bin/ubuntu/python311/venv/bin/python`. Reference images: `src/demos/TestData/Objects/*`.

---

### Task 1: Spike (validate assumptions + resolve model sources) and scaffold settings

**Files:**
- Create: `modules/ObjectDetection/modulesettings.json`
- Create: `modules/ObjectDetection/modulesettings.windows.json`
- Create: `modules/ObjectDetection/.gitignore`
- Create: `modules/ObjectDetection/SPIKE.md` (findings; not shipped logic)
- Test: `modules/ObjectDetection/tests/test_modulesettings.py`

**Interfaces:**
- Produces: `modulesettings.json` with the three RouteMaps, queue `objectdetection_queue`, EnvironmentVariables (`DEFAULT_MODEL`,`MODEL_TIER`,`CLASS_FILTER`,`MODELS_DIR`,`CUSTOM_MODELS_DIR`), and `ModuleReleases` 1.0.0.
- Produces (`SPIKE.md`): resolved facts later tasks depend on — the exact YOLO26 model id string that Ultralytics accepts (e.g. `yolo26n.pt`), whether `yolov5` + `ultralytics` import together under torch 2.13, whether a legacy IPcam `.pt` loads under `yolov5==6.2.3`, and the exact CDN/URL + filenames for the IPcam custom models and their size variants.

- [ ] **Step 1: Spike — engine coexistence + YOLO26 availability**

Run (record output in SPIKE.md):
```bash
V=runtimes/bin/ubuntu/python311/venv/bin/python
$V -m pip install -q ultralytics
$V - <<'PY'
import torch, ultralytics, yolov5
print("torch", torch.__version__, "ultralytics", ultralytics.__version__)
from ultralytics import YOLO
# find the YOLO26 id Ultralytics accepts (try in order; the first that downloads wins)
for name in ("yolo26n.pt","yolo11n.pt","yolov8n.pt"):
    try:
        m = YOLO(name); print("OK default-generic id:", name); break
    except Exception as e:
        print("no:", name, str(e)[:80])
PY
```
Record: the working generic model id (YOLO26 if available, else the fallback per Global Constraints), and that `import yolov5` and `import ultralytics` coexist.

- [ ] **Step 2: Spike — legacy IPcam model loads under yolov5, and resolve its source**

Use an existing legacy model to confirm the yolov5 path works, and find the IPcam asset URL. The `ObjectDetectionYOLOv5-6.2` module already downloads `custom-models-yolo5-pt.zip` in its `install.sh` (`getFromServer "models/" "custom-models-yolo5-pt.zip" ...`). Record that asset name + the `assetStorageUrl` from `server/appsettings.json` in SPIKE.md, and confirm a legacy `.pt` (e.g. copy one from `modules/ObjectDetectionYOLOv5-6.2/custom-models/` if present, else download the zip) loads:
```bash
V=runtimes/bin/ubuntu/python311/venv/bin/python
$V - <<'PY'
import functools, torch
torch.load = functools.partial(torch.load, weights_only=False)   # legacy trusted weights
from yolov5.models.common import DetectMultiBackend
m = DetectMultiBackend("<path-to-legacy-ipcam-combined.pt>", device=torch.device("cpu"))
print("legacy IPcam loaded OK, names:", list(m.names.values())[:8])
PY
```
Record in SPIKE.md: the exact IPcam filenames (ipcam-combined, ipcam-general, ipcam-animal, ipcam-dark, license-plate, delivery), any size variants, and the download URL(s).

- [ ] **Step 3: Write the failing test**

```python
# modules/ObjectDetection/tests/test_modulesettings.py
import json, re, os
BASE = os.path.dirname(os.path.dirname(__file__))
def _load():
    raw = re.sub(r"(?<!:)//.*", "", open(os.path.join(BASE,"modulesettings.json"),encoding="utf-8").read())
    return json.loads(raw)
def test_identity_routes():
    m = _load()["Modules"]["ObjectDetection"]
    assert m["Version"] == "1.0.0"
    assert m["LaunchSettings"]["Queue"] == "objectdetection_queue"
    assert m["LaunchSettings"]["Runtime"] == "python3.11"
    routes = {r["Route"] for r in m["RouteMaps"]}
    assert routes == {"vision/detection", "vision/custom", "vision/custom/list"}
    cmds = {r["Command"] for r in m["RouteMaps"]}
    assert cmds == {"detect", "custom", "list-custom"}
    ev = m["EnvironmentVariables"]
    assert "DEFAULT_MODEL" in ev and "MODEL_TIER" in ev
```

- [ ] **Step 4: Run test to verify it fails**

Run: `runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ObjectDetection/tests/test_modulesettings.py -v`
Expected: FAIL (files missing).

- [ ] **Step 5: Write `modulesettings.json`, `modulesettings.windows.json`, `.gitignore`**

Base `modulesettings.json` on `modules/ObjectDetectionYOLOv5-6.2/modulesettings.json` (read it): keep the three RouteMaps (Routes/Methods/Commands/Inputs/Outputs) IDENTICAL for Blue Iris parity; set module id/key `ObjectDetection`, Name "Object Detection", Version 1.0.0, Queue `objectdetection_queue`, FilePath `detect_adapter.py`, Runtime `python3.11`. Set `EnvironmentVariables`: `DEFAULT_MODEL`=`ipcam-combined`, `MODEL_TIER`=`auto`, `CLASS_FILTER`=`""`, `MODELS_DIR`, `CUSTOM_MODELS_DIR`, `DATA_DIR`=`%DATA_DIR%`. Add `ModuleReleases` 1.0.0. `modulesettings.windows.json`: `{ "Modules": { "ObjectDetection": { "LaunchSettings": { "Runtime": "python3.11" } } } }`. `.gitignore`: `assets/`, `custom-models/`, `__pycache__/`, `*.pyc`, `datastore/`.

- [ ] **Step 6: Run test to verify it passes**

Run: `runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ObjectDetection/tests/test_modulesettings.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add modules/ObjectDetection/modulesettings.json modules/ObjectDetection/modulesettings.windows.json modules/ObjectDetection/.gitignore modules/ObjectDetection/SPIKE.md modules/ObjectDetection/tests/test_modulesettings.py
git commit -m "feat(object-detection): scaffold settings + spike (engine coexistence, model sources)"
```

If the spike shows YOLO26 is not yet in the installed Ultralytics, record the chosen generic id (yolo11n etc.) in SPIKE.md and use it as the `yolo26` family id in Task 2 (the family name stays `yolo26`, the underlying weight id is whatever the spike resolved).

---

### Task 2: `options.py` — tiering + model registry

**Files:**
- Create: `modules/ObjectDetection/options.py`
- Test: `modules/ObjectDetection/tests/test_options.py`

**Interfaces:**
- Produces: `select_tier(has_cuda, vram_bytes, override) -> "accurate"|"balanced"|"fast"`; `resolve_default(default_model, tier) -> (engine, weight_id)` where `engine in {"yolov5","ultralytics"}`; a `Options` object exposing `use_CUDA`, `tier`, `default_model`, `default_engine`, `default_weight`, `models_dir`, `custom_models_dir`, `class_filter` (list[str]), `min_confidence_default` (0.4).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_options.py
import importlib.util, os
BASE = os.path.dirname(os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location("options", os.path.join(BASE,"options.py"))
o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)

def test_tiers():
    assert o.select_tier(True, 32*10**9, "auto") == "accurate"
    assert o.select_tier(True,  6*10**9, "auto") == "balanced"
    assert o.select_tier(False, 0,       "auto") == "fast"
    assert o.select_tier(True,  6*10**9, "accurate") == "accurate"

def test_resolve_default_cctv():
    # CCTV family uses yolov5 engine on GPU tiers
    eng, w = o.resolve_default("ipcam-combined", "accurate")
    assert eng == "yolov5" and "ipcam" in w
    # CPU/fast tier for CCTV falls back to the generic nano (ultralytics)
    eng2, w2 = o.resolve_default("ipcam-combined", "fast")
    assert eng2 == "ultralytics"

def test_resolve_default_generic():
    eng, w = o.resolve_default("yolo26", "accurate")
    assert eng == "ultralytics"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ObjectDetection/tests/test_options.py -v`
Expected: FAIL.

- [ ] **Step 3: Write `options.py`**

```python
import os
from codeproject_ai_sdk import ModuleOptions

VRAM_CUTOFF_BYTES = 8 * 10**9  # >= 8GB -> accurate

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/ObjectDetection/options.py modules/ObjectDetection/tests/test_options.py
git commit -m "feat(object-detection): tiering + default-model registry"
```

---

### Task 3: Ultralytics engine wrapper (`engine_ultralytics.py`)

**Files:**
- Create: `modules/ObjectDetection/engine_ultralytics.py`
- Test: `modules/ObjectDetection/tests/test_engine_ultralytics.py`

**Interfaces:**
- Produces: `UltralyticsEngine(model_path_or_id, use_cuda).detect(bgr_or_pil, min_confidence) -> list[dict]` where each dict is `{confidence: float, label: str, x_min: int, y_min: int, x_max: int, y_max: int}`.

- [ ] **Step 1: Write the failing test** (integration — needs a generic model; uses the id resolved in SPIKE.md, default `yolo26n.pt`, which Ultralytics auto-downloads)

```python
# tests/test_engine_ultralytics.py
import importlib.util, os, cv2, pytest
BASE = os.path.dirname(os.path.dirname(__file__))
def _load():
    spec = importlib.util.spec_from_file_location("engine_ultralytics", os.path.join(BASE,"engine_ultralytics.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
IMG = "src/demos/TestData/Objects/street-at-night.jpg"

def test_detect_returns_predictions():
    E = _load().UltralyticsEngine("yolo26n.pt", use_cuda=False)
    preds = E.detect(cv2.imread(IMG), 0.4)
    assert isinstance(preds, list) and len(preds) >= 1
    p = preds[0]
    assert {"confidence","label","x_min","y_min","x_max","y_max"} <= set(p)
    assert isinstance(p["label"], str) and 0.0 <= p["confidence"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ObjectDetection/tests/test_engine_ultralytics.py -v`
Expected: FAIL.

- [ ] **Step 3: Write `engine_ultralytics.py`**

```python
import functools, numpy as np, torch
# Trusted local model files (Ultralytics/community .pt): restore torch<2.6 default.
_orig = torch.load
torch.load = functools.partial(_orig, weights_only=False)
from ultralytics import YOLO

class UltralyticsEngine:
    def __init__(self, model_path_or_id, use_cuda):
        self.device = 0 if use_cuda else "cpu"
        self.model = YOLO(model_path_or_id)   # auto-downloads known ids; loads local .pt

    def detect(self, image, min_confidence):
        r = self.model.predict(source=image, conf=float(min_confidence),
                               device=self.device, verbose=False)[0]
        names = r.names
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            out.append({
                "confidence": float(b.conf[0]),
                "label": str(names[int(b.cls[0])]),
                "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
            })
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS (≥1 prediction, correct keys).

- [ ] **Step 5: Commit**

```bash
git add modules/ObjectDetection/engine_ultralytics.py modules/ObjectDetection/tests/test_engine_ultralytics.py
git commit -m "feat(object-detection): Ultralytics (YOLO26) engine wrapper"
```

---

### Task 4: yolov5 legacy engine wrapper (`engine_yolov5.py`)

**Files:**
- Create: `modules/ObjectDetection/engine_yolov5.py`
- Test: `modules/ObjectDetection/tests/test_engine_yolov5.py`

**Interfaces:**
- Produces: `Yolov5Engine(model_path, use_cuda).detect(bgr, min_confidence) -> list[dict]` (SAME dict shape as Task 3). This mirrors `modules/ObjectDetectionYOLOv5-6.2/detect.py`'s loading (DetectMultiBackend + AutoShape + `weights_only=False`).

- [ ] **Step 1: Write the failing test** (needs a legacy `.pt`; use the IPcam model path resolved in SPIKE.md; skip if absent)

```python
# tests/test_engine_yolov5.py
import importlib.util, os, cv2, pytest
BASE = os.path.dirname(os.path.dirname(__file__))
MODEL = os.path.join(BASE, "custom-models", "ipcam-combined.pt")
def _load():
    spec = importlib.util.spec_from_file_location("engine_yolov5", os.path.join(BASE,"engine_yolov5.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

@pytest.mark.skipif(not os.path.exists(MODEL), reason="legacy IPcam model not downloaded")
def test_legacy_detect():
    E = _load().Yolov5Engine(MODEL, use_cuda=False)
    preds = E.detect(cv2.imread("src/demos/TestData/Objects/study-group.jpg"), 0.4)
    assert isinstance(preds, list)
    if preds:
        assert {"confidence","label","x_min","y_min","x_max","y_max"} <= set(preds[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ObjectDetection/tests/test_engine_yolov5.py -v`
Expected: FAIL (module missing) — or SKIP if the legacy model isn't present yet; download it per SPIKE.md first so the test actually runs.

- [ ] **Step 3: Write `engine_yolov5.py`**

```python
import functools, numpy as np, torch
_orig = torch.load
torch.load = functools.partial(_orig, weights_only=False)   # legacy trusted weights (torch 2.13)
from yolov5.models.common import DetectMultiBackend, AutoShape

class Yolov5Engine:
    def __init__(self, model_path, use_cuda):
        dev = torch.device("cuda:0" if use_cuda else "cpu")
        self.model = AutoShape(DetectMultiBackend(model_path, device=dev, fp16=False))

    def detect(self, bgr, min_confidence):
        self.model.conf = float(min_confidence)
        r = self.model(bgr[:, :, ::-1], size=640)   # AutoShape expects RGB
        names = self.model.names
        out = []
        for *xyxy, conf, cls in r.xyxy[0].tolist():
            out.append({
                "confidence": float(conf),
                "label": str(names[int(cls)]),
                "x_min": int(xyxy[0]), "y_min": int(xyxy[1]),
                "x_max": int(xyxy[2]), "y_max": int(xyxy[3]),
            })
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS (loads a legacy IPcam model, returns valid predictions).

- [ ] **Step 5: Commit**

```bash
git add modules/ObjectDetection/engine_yolov5.py modules/ObjectDetection/tests/test_engine_yolov5.py
git commit -m "feat(object-detection): yolov5 legacy engine wrapper"
```

---

### Task 5: Custom-model router (`model_router.py`)

**Files:**
- Create: `modules/ObjectDetection/model_router.py`
- Test: `modules/ObjectDetection/tests/test_model_router.py`

**Interfaces:**
- Consumes: `UltralyticsEngine`, `Yolov5Engine`.
- Produces: `ModelRouter(use_cuda)` with `get(model_path) -> engine` (an object exposing `.detect(image, conf)`), caching by path. Loads with Ultralytics first; on a legacy-pickle failure (exception text contains `models.yolo` or `weights_only` or `Can't get attribute`), falls back to `Yolov5Engine`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_router.py
import importlib.util, os, pytest
BASE = os.path.dirname(os.path.dirname(__file__))
def _load():
    spec = importlib.util.spec_from_file_location("model_router", os.path.join(BASE,"model_router.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
LEGACY = os.path.join(BASE, "custom-models", "ipcam-combined.pt")

@pytest.mark.skipif(not os.path.exists(LEGACY), reason="legacy model not downloaded")
def test_legacy_routes_to_yolov5():
    R = _load().ModelRouter(use_cuda=False)
    eng = R.get(LEGACY)
    assert type(eng).__name__ == "Yolov5Engine"
    # cached: same instance on second call
    assert R.get(LEGACY) is eng

def test_modern_routes_to_ultralytics(tmp_path):
    # yolo26n.pt is a modern Ultralytics weight
    R = _load().ModelRouter(use_cuda=False)
    eng = R.get("yolo26n.pt")
    assert type(eng).__name__ == "UltralyticsEngine"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ObjectDetection/tests/test_model_router.py -v`
Expected: FAIL.

- [ ] **Step 3: Write `model_router.py`**

```python
from engine_ultralytics import UltralyticsEngine
from engine_yolov5 import Yolov5Engine

_LEGACY_MARKERS = ("models.yolo", "weights_only", "Can't get attribute", "No module named 'models'")

class ModelRouter:
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        self._cache = {}

    def get(self, model_path):
        if model_path in self._cache:
            return self._cache[model_path]
        try:
            eng = UltralyticsEngine(model_path, self.use_cuda)
        except Exception as ex:
            if any(m in str(ex) for m in _LEGACY_MARKERS):
                eng = Yolov5Engine(model_path, self.use_cuda)   # legacy fallback
            else:
                raise
        self._cache[model_path] = eng
        return eng
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS (legacy→Yolov5Engine cached; modern→UltralyticsEngine).

- [ ] **Step 5: Commit**

```bash
git add modules/ObjectDetection/model_router.py modules/ObjectDetection/tests/test_model_router.py
git commit -m "feat(object-detection): custom-model engine router (ultralytics first, yolov5 fallback)"
```

---

### Task 6: Adapter + command handlers (`detect_adapter.py`)

**Files:**
- Create: `modules/ObjectDetection/detect_adapter.py`
- Test: `modules/ObjectDetection/tests/test_adapter.py`

**Interfaces:**
- Consumes: `options.Options`, `model_router.ModelRouter`, `engine_ultralytics.UltralyticsEngine`, `engine_yolov5.Yolov5Engine`.
- Produces: a testable `Detector(opts)` class with `detect(bgr, min_confidence) -> JSON`, `custom(model_name, bgr, min_confidence) -> JSON`, `list_models() -> JSON` (Blue-Iris-shaped: `{success,count,predictions,inferenceMs,processMs,message}` / `{success,models,message}`); and `ObjectDetection_adapter(ModuleRunner)` with `initialise/process/selftest`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapter.py
import importlib.util, os, cv2, pytest
BASE = os.path.dirname(os.path.dirname(__file__))
def _mod():
    spec = importlib.util.spec_from_file_location("detect_adapter", os.path.join(BASE,"detect_adapter.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

class _Opts:  # minimal stand-in
    use_CUDA=False; tier="fast"; default_model="yolo26"; default_engine="ultralytics"
    default_weight="yolo26n.pt"; models_dir=os.path.join(BASE,"assets")
    custom_models_dir=os.path.join(BASE,"custom-models"); class_filter=[]; min_confidence_default=0.4

def test_detect_shape():
    D = _mod().Detector(_Opts())
    out = D.detect(cv2.imread("src/demos/TestData/Objects/street-at-night.jpg"), 0.4)
    assert out["success"] is True
    assert {"count","predictions","inferenceMs","processMs"} <= set(out)
    if out["predictions"]:
        assert {"confidence","label","x_min","y_min","x_max","y_max"} <= set(out["predictions"][0])

def test_class_filter():
    o = _Opts(); o.class_filter=["car"]
    D = _mod().Detector(o)
    out = D.detect(cv2.imread("src/demos/TestData/Objects/traffic.jpg"), 0.3)
    assert all(p["label"].lower()=="car" for p in out["predictions"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ObjectDetection/tests/test_adapter.py -v`
Expected: FAIL.

- [ ] **Step 3: Write `detect_adapter.py`**

Implement `Detector(opts)`:
- `__init__`: build the default engine — if `opts.default_engine=="yolov5"`, `Yolov5Engine(os.path.join(opts.custom_models_dir, opts.default_weight + ".pt"), opts.use_CUDA)`; else `UltralyticsEngine(opts.default_weight, opts.use_CUDA)`. Also create `ModelRouter(opts.use_CUDA)` for custom. Wrap default-engine construction in the GPU-OOM→CPU fallback (`try/except`, on `"out of memory"` set use_CUDA False and rebuild), matching FaceProcessing.
- `_format(preds, t0)`: apply `opts.class_filter` (keep predictions whose `label.lower()` is in the filter, if non-empty); build `{ success:True, count:len, predictions, inferenceMs, processMs:int((perf_counter()-t0)*1000), message: "Found "+... }`.
- `detect(bgr, conf)`: time, run default engine, `_format`.
- `custom(model_name, bgr, conf)`: map `model_name=="general"` → `"ipcam-general"`; resolve `<custom_models_dir>/<model_name>.pt` (return `{success:False,error:"model <name> not found"}` if missing); `self.router.get(path).detect(bgr, conf)`; `_format`.
- `list_models()`: scan `custom_models_dir` for `*.pt` → `{ success:True, models:[names], message }`.

Then `ObjectDetection_adapter(ModuleRunner)` mirroring `modules/ObjectDetectionYOLOv5-6.2/detect_adapter.py`: `initialise()` builds `Options()`+`Detector`; `process(data)` dispatches `detect`/`custom` (model from `data.segments[0]`)/`list-custom`, decoding the image via `data.get_image(0)` → BGR ndarray; `selftest()` runs `detect` on `test/objects.jpg`; `if __name__=="__main__": ObjectDetection_adapter().start_loop()`. Keep response keys byte-identical to the existing module.

- [ ] **Step 4: Run tests to verify they pass**

Run: same as Step 2. Expected: PASS (detect shape correct; class filter keeps only `car`).

- [ ] **Step 5: Commit**

```bash
git add modules/ObjectDetection/detect_adapter.py modules/ObjectDetection/tests/test_adapter.py
git commit -m "feat(object-detection): adapter + detect/custom/list handlers (Blue Iris parity)"
```

---

### Task 7: install/requirements/post_install + Windows + self-test

**Files:**
- Create: `modules/ObjectDetection/requirements.linux.cuda12.txt`, `requirements.linux.cuda.txt`, `requirements.windows.cuda.txt`, `requirements.txt` (CPU), `requirements.windows.txt` (CPU)
- Create: `modules/ObjectDetection/install.sh`, `install.bat`
- Create: `modules/ObjectDetection/post_install.sh`, `post_install.bat`
- Create: `modules/ObjectDetection/download_models.py`
- Create: `modules/ObjectDetection/explore.html`, `modules/ObjectDetection/test/objects.jpg`

**Interfaces:**
- Produces: a module that `bash ../../src/setup.sh` installs end-to-end (venv deps + YOLO26 + IPcam custom models) with a passing GPU self-test.

- [ ] **Step 1: Write requirements**

GPU (`requirements.linux.cuda12.txt` = `requirements.linux.cuda.txt` = `requirements.windows.cuda.txt`):
```
#! Python3.11
numpy
opencv-python
Pillow
--extra-index-url https://download.pytorch.org/whl/cu130
torch==2.13.0+cu130
--extra-index-url https://download.pytorch.org/whl/cu130
torchvision==0.28.0+cu130
ultralytics
yolov5==6.2.3
CodeProject-AI-SDK
```
CPU (`requirements.txt` = `requirements.windows.txt`): same but `--extra-index-url .../cpu` and `torch==2.13.0+cpu` / `torchvision==0.28.0+cpu`.

- [ ] **Step 2: Write `download_models.py`**

An argparse script (`--dest`, `--custom-dest`) that: (a) triggers Ultralytics to fetch the tier's YOLO26 weight (`from ultralytics import YOLO; YOLO(weight_id)`), (b) downloads the IPcam custom-model set + default CCTV model to `--custom-dest` using the URL(s) recorded in `SPIKE.md` (verify via file existence; use `urllib` with an HTTPRedirectHandler). Skip files that already exist.

- [ ] **Step 3: Write `install.sh` + `install.bat`**

`install.sh` (guard `[ "$1" = "install" ]`), then:
```bash
"${venvPythonCmdPath}" "${moduleDirPath}/download_models.py" --dest "${moduleDirPath}/assets" --custom-dest "${moduleDirPath}/custom-models"
if [ ! -f "${moduleDirPath}/custom-models/ipcam-combined.pt" ]; then moduleInstallErrors="Failed to download CCTV models"; fi
```
`install.bat`: the Windows equivalent using `%venvPythonCmdPath%` / `%moduleDirPath%` (mirror `modules/ImprovedFaceProcessing/install.bat`).

- [ ] **Step 4: Write `post_install.sh` + `post_install.bat`**

Mirror `modules/ObjectDetectionYOLOv5-6.2/post_install.sh` (which you can read): `pip install "setuptools<81"`; on GPU (`installGPU`=`true` && `hasCUDA`=`true`) `pip install --upgrade "nvidia-cudnn-cu13==9.24.0.43"`. `.bat` mirror per `modules/ImprovedFaceProcessing/post_install.bat` (minus the onnxruntime lines — this module has none).

- [ ] **Step 5: explore.html + test image**

```bash
cp modules/ObjectDetectionYOLOv5-6.2/explore.html modules/ObjectDetection/explore.html
mkdir -p modules/ObjectDetection/test
cp src/demos/TestData/Objects/street-at-night.jpg modules/ObjectDetection/test/objects.jpg
```
Update explore.html title to "Object Detection".

- [ ] **Step 6: Run full module setup + self-test**

```bash
cd modules/ObjectDetection && bash ../../src/setup.sh --verbosity info 2>&1 | tee /tmp/od_setup.log; cd -
grep -E "Self-test|Setup complete|SETUP FAILED" /tmp/od_setup.log
```
Expected: `Self-test passed`, `Setup complete`, exit 0. Fix any failure before committing.

- [ ] **Step 7: Commit**

```bash
git add modules/ObjectDetection/requirements*.txt modules/ObjectDetection/install.sh modules/ObjectDetection/install.bat modules/ObjectDetection/post_install.sh modules/ObjectDetection/post_install.bat modules/ObjectDetection/download_models.py modules/ObjectDetection/explore.html modules/ObjectDetection/test/objects.jpg
git commit -m "feat(object-detection): install/requirements/post_install (+Windows) + self-test passing"
```

---

### Task 8: Server integration, Blue Iris parity, A/B + latency

**Files:**
- Modify: `modules/ObjectDetectionYOLOv5-6.2/modulesettings.json` (`AutoStart` → `false`)
- Create: `docs/superpowers/plans/object-detection-benchmark.md`

- [ ] **Step 1: Disable the predecessor**

Set `LaunchSettings.AutoStart: false` in `modules/ObjectDetectionYOLOv5-6.2/modulesettings.json` so `ObjectDetection` owns the routes (the server's route-ownership logic makes the enabled module win).

- [ ] **Step 2: Restart server, confirm ObjectDetection Started**

```bash
kill $(cat /tmp/cpai_server.pid) 2>/dev/null; sleep 3
cd src/server && ASPNETCORE_ENVIRONMENT=Development dotnet bin/Debug/net9.0/CodeProject.AI.Server.dll >/tmp/cpai_server.log 2>&1 & echo $! > /tmp/cpai_server.pid; cd -
```
Wait for ping; confirm `ObjectDetection` reaches `Started` and reports GPU.

- [ ] **Step 3: Blue Iris parity smoke tests**

```bash
# default detection
curl -s -X POST http://127.0.0.1:32168/v1/vision/detection -F image=@src/demos/TestData/Objects/traffic.jpg | python3 -m json.tool
# custom (legacy IPcam) — proves the yolov5 path via the server
curl -s -X POST http://127.0.0.1:32168/v1/vision/custom/ipcam-combined -F image=@src/demos/TestData/Objects/study-group.jpg | python3 -m json.tool
# list
curl -s -X POST http://127.0.0.1:32168/v1/vision/custom/list | python3 -m json.tool
```
Expect: `success:true`, `predictions` with `label/confidence/x_min/y_min/x_max/y_max`, `moduleId: "ObjectDetection"`, and custom/list returns the IPcam model names.

- [ ] **Step 4: A/B + measured latency**

Throwaway script (in `/tmp`, not committed): run default detect + the `ipcam-combined` custom model on all `src/demos/TestData/Objects/*.jpg`; record per-image object counts (CCTV-default vs yolo26) and time 50 `/v1/vision/detection` calls for median GPU latency; repeat with `MODEL_TIER=fast` (CPU torch) for CPU-tier latency. Save to `docs/superpowers/plans/object-detection-benchmark.md`.

- [ ] **Step 5: Commit**

```bash
git add modules/ObjectDetectionYOLOv5-6.2/modulesettings.json docs/superpowers/plans/object-detection-benchmark.md
git commit -m "feat(object-detection): make successor default; A/B + latency benchmarks"
```

---

## Notes for the implementer

- The `weights_only=False` monkeypatch is required in both engine wrappers (torch 2.6+ default; trusted local weights) — same fix used across the cu130 modules.
- Keep every response JSON key byte-identical to `modules/ObjectDetectionYOLOv5-6.2` — Blue Iris parses these; the shape test in Task 6/8 guards it.
- `data.segments[0]` carries the `<model>` from `/v1/vision/custom/<model>` (see the existing adapter). `general` maps to `ipcam-general`.
- Weights are large and gitignored; the legacy IPcam models and YOLO26 weights must be downloaded (Task 1 spike resolves the URLs; Task 7 automates it) before Tasks 3–8 integration tests pass.
- If SPIKE.md finds YOLO26 isn't in the installed Ultralytics yet, use the resolved fallback id (yolo11x/m/n) as the `GENERIC` weights in `options.py` — the `yolo26` family name and the design are unchanged.
