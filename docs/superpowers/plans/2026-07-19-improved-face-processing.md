# Improved Face Processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new CodeProject.AI module `ImprovedFaceProcessing` (v1.0.0) that replaces FaceProcessing with an SCRFD detector + AdaFace recognizer pipeline, with hardware auto-tiering and full face-API parity.

**Architecture:** A Python module in `modules/ImprovedFaceProcessing/` following the FaceProcessing layout: an SDK `ModuleRunner` adapter dispatches the six face commands; SCRFD (onnxruntime) does detection+landmarks, `norm_crop` aligns to 112×112, AdaFace (PyTorch) produces 512-d embeddings, and a SQLite-backed in-memory gallery does cosine matching. Device/VRAM detection at startup selects an "accurate" (SCRFD-10G + AdaFace IR-101) or "fast" (SCRFD-2.5G + AdaFace IR-50) tier.

**Tech Stack:** Python 3.11, torch 2.13.0+cu130 (GPU) / torch 2.13.0+cpu (CPU), onnxruntime-gpu / onnxruntime, insightface (SCRFD + `face_align.norm_crop`), AdaFace net (vendored), sqlite3, CodeProject.AI Python SDK, pytest.

## Global Constraints

- Module ID `ImprovedFaceProcessing`; Name "Improved Face Processing"; Version `1.0.0`.
- Runtime `python3.11`, RuntimeLocation `Shared` (shares the venv at `runtimes/bin/ubuntu/python311/venv`).
- GPU requirements: `torch==2.13.0+cu130`, `torchvision==0.28.0+cu130` (index `https://download.pytorch.org/whl/cu130`), `onnxruntime-gpu`. CPU requirements: `torch==2.13.0+cpu`, `torchvision==0.28.0+cpu` (index `https://download.pytorch.org/whl/cpu`), `onnxruntime`.
- `post_install.sh` MUST: `pip install "setuptools<81"`; on GPU `pip install --upgrade "nvidia-cudnn-cu13==9.24.0.43"`; uninstall CPU `onnxruntime` if present alongside `onnxruntime-gpu`.
- onnxruntime GPU needs `libcudart.so.13` etc. from torch's cu13 wheels at `<venv>/lib/python3.11/site-packages/nvidia/cu13/lib`; that dir MUST be on `LD_LIBRARY_PATH` before `import onnxruntime` (spike-proven).
- Routes: `vision/face`, `vision/face/recognize`, `vision/face/register`, `vision/face/match`, `vision/face/list`, `vision/face/delete`. Queue `improvedfaceprocessing_queue`.
- Own DB: `/etc/codeproject/ai/improved_faceembedding.db`. Schema: `TB_EMBEDDINGS(userid TEXT PRIMARY KEY, embedding TEXT NOT NULL)`.
- Only one of FaceProcessing / ImprovedFaceProcessing may be enabled (same routes).
- Response JSON shapes MUST match FaceProcessing so existing clients work.
- Weights are research-oriented (attributed); no commercial-license guarantee.
- Test interpreter: `runtimes/bin/ubuntu/python311/venv/bin/python`. Reference test images: `src/demos/TestData/Faces/*`.

---

### Task 1: Module scaffolding, settings, and a model-fetch helper

**Files:**
- Create: `modules/ImprovedFaceProcessing/modulesettings.json`
- Create: `modules/ImprovedFaceProcessing/intelligencelayer/download_models.py`
- Create: `modules/ImprovedFaceProcessing/test/README.md` (placeholder for test assets)
- Test: `modules/ImprovedFaceProcessing/tests/test_modulesettings.py`

**Interfaces:**
- Produces: `download_models.py` exposes `TIER_MODELS: dict` (keys `"accurate"`,`"fast"`) each `{ "detector": {"file","url","sha256"}, "recognizer": {"file","url","sha256"} }`, and `download_tier(tier: str, dest_dir: str) -> list[str]` (returns downloaded file paths).
- Produces: `modulesettings.json` with module id `ImprovedFaceProcessing`, queue `improvedfaceprocessing_queue`, the six RouteMaps, and `EnvironmentVariables` incl. `MODEL_TIER` and an `LD_LIBRARY_PATH` addition.

- [ ] **Step 1: Write the failing test**

```python
# modules/ImprovedFaceProcessing/tests/test_modulesettings.py
import json, re, os
BASE = os.path.dirname(os.path.dirname(__file__))

def _load_settings():
    raw = open(os.path.join(BASE, "modulesettings.json"), encoding="utf-8").read()
    raw = re.sub(r"//.*", "", raw)               # strip // comments
    return json.loads(raw)

def test_module_identity_and_routes():
    d = _load_settings()
    m = d["Modules"]["ImprovedFaceProcessing"]
    assert m["Version"] == "1.0.0"
    assert m["LaunchSettings"]["Queue"] == "improvedfaceprocessing_queue"
    assert m["LaunchSettings"]["Runtime"] == "python3.11"
    routes = {r["Route"] for r in m["RouteMaps"]}
    assert routes == {"vision/face", "vision/face/recognize", "vision/face/register",
                      "vision/face/match", "vision/face/list", "vision/face/delete"}

def test_download_tiers_defined():
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "download_models", os.path.join(BASE, "intelligencelayer", "download_models.py"))
    dm = importlib.util.module_from_spec(spec); spec.loader.exec_module(dm)
    for tier in ("accurate", "fast"):
        t = dm.TIER_MODELS[tier]
        assert t["detector"]["file"].endswith(".onnx")
        assert t["recognizer"]["file"].endswith(".pt")
        assert t["detector"]["url"].startswith("http")
        assert t["recognizer"]["url"].startswith("http")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ImprovedFaceProcessing/tests/test_modulesettings.py -v`
Expected: FAIL (files do not exist).

- [ ] **Step 3: Create `modulesettings.json`**

Copy `modules/FaceProcessing/modulesettings.json` as a base and change: top-level key and `Name`→"Improved Face Processing", `Version`→"1.0.0"; `LaunchSettings.Queue`→`improvedfaceprocessing_queue`; `LaunchSettings.FilePath`→`improved_face.py`; keep the six `RouteMaps` Routes/Commands/Inputs/Outputs identical (client parity). Add to `EnvironmentVariables`:

```jsonc
"MODEL_TIER":       "auto",   // auto | accurate | fast
"DATA_DIR":         "/etc/codeproject/ai",
"LD_LIBRARY_PATH":  "%CURRENT_MODULE_PATH%/../../runtimes/bin/ubuntu/python311/venv/lib/python3.11/site-packages/nvidia/cu13/lib"
```

Add a `ModuleReleases` entry: `{ "ModuleVersion": "1.0.0", "ServerVersionRange": [ "2.8.0", "" ], "ReleaseDate": "2026-07-19", "ReleaseNotes": "SCRFD + AdaFace accuracy-first pipeline", "Importance": "Major" }`.

- [ ] **Step 4: Create `download_models.py`**

```python
# modules/ImprovedFaceProcessing/intelligencelayer/download_models.py
import os, hashlib, urllib.request

# NOTE: fill exact URLs/sha256 during Task 1 implementation. Detector = SCRFD .onnx
# from the InsightFace model packs (buffalo_l -> det_10g.onnx, buffalo_m -> det_2.5g.onnx).
# Recognizer = AdaFace .pt from the AdaFace/CVLface HuggingFace repos (IR-101 WebFace12M, IR-50).
TIER_MODELS = {
    "accurate": {
        "detector":   {"file": "scrfd_10g.onnx",  "url": "<INSIGHTFACE_SCRFD_10G_ONNX_URL>",  "sha256": "<sha>"},
        "recognizer": {"file": "adaface_ir101.pt", "url": "<HF_ADAFACE_IR101_WEBFACE12M_URL>", "sha256": "<sha>"},
    },
    "fast": {
        "detector":   {"file": "scrfd_2.5g.onnx", "url": "<INSIGHTFACE_SCRFD_2.5G_ONNX_URL>", "sha256": "<sha>"},
        "recognizer": {"file": "adaface_ir50.pt",  "url": "<HF_ADAFACE_IR50_URL>",             "sha256": "<sha>"},
    },
}

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def download_tier(tier, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    out = []
    for role in ("detector", "recognizer"):
        spec = TIER_MODELS[tier][role]
        dest = os.path.join(dest_dir, spec["file"])
        if not os.path.exists(dest):
            urllib.request.urlretrieve(spec["url"], dest)
        if spec.get("sha256") and not spec["sha256"].startswith("<"):
            assert _sha256(dest) == spec["sha256"], f"checksum mismatch for {spec['file']}"
        out.append(dest)
    return out
```

During implementation, resolve the real URLs (InsightFace model-zoo release assets for SCRFD onnx; HuggingFace `resolve/main` URLs for AdaFace `.pt`) and fill `sha256`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ImprovedFaceProcessing/tests/test_modulesettings.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Fetch the weights (both tiers) for later tasks**

Run: `runtimes/bin/ubuntu/python311/venv/bin/python -c "import sys; sys.path.insert(0,'modules/ImprovedFaceProcessing/intelligencelayer'); import download_models as d; d.download_tier('accurate','modules/ImprovedFaceProcessing/assets'); d.download_tier('fast','modules/ImprovedFaceProcessing/assets')"`
Expected: four files in `modules/ImprovedFaceProcessing/assets/`.

- [ ] **Step 7: Commit**

```bash
git add modules/ImprovedFaceProcessing/modulesettings.json modules/ImprovedFaceProcessing/intelligencelayer/download_models.py modules/ImprovedFaceProcessing/tests/test_modulesettings.py modules/ImprovedFaceProcessing/test/README.md
git commit -m "feat(improved-face): scaffold module settings + model download helper"
```

---

### Task 2: Install onnxruntime + insightface into the shared venv (dev environment)

**Files:**
- Modify: (venv only — no repo files) — this task prepares the test environment; the production equivalent is codified in Task 8.
- Test: `modules/ImprovedFaceProcessing/tests/test_runtime_env.py`

**Interfaces:**
- Produces: a shared venv where `import onnxruntime` exposes `CUDAExecutionProvider` (on the GPU box) and `insightface` imports.

- [ ] **Step 1: Write the failing test**

```python
# modules/ImprovedFaceProcessing/tests/test_runtime_env.py
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
```

Note: because `LD_LIBRARY_PATH` must be set *before* the process starts for the C++ loader, run this test with the env var already exported (Step 3).

- [ ] **Step 2: Run test to verify it fails**

Run: `runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ImprovedFaceProcessing/tests/test_runtime_env.py -v`
Expected: FAIL (`onnxruntime`/`insightface` not installed).

- [ ] **Step 3: Install deps + verify provider**

```bash
VENV=runtimes/bin/ubuntu/python311/venv
$VENV/bin/python -m pip install onnxruntime-gpu insightface
# insightface pulls CPU onnxruntime — remove it so it doesn't shadow the GPU provider
$VENV/bin/python -m pip uninstall -y onnxruntime
NVLIB=$(pwd)/$VENV/lib/python3.11/site-packages/nvidia/cu13/lib
LD_LIBRARY_PATH="$NVLIB" $VENV/bin/python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
Expected: providers include `CUDAExecutionProvider` on the 5090.

- [ ] **Step 4: Run test to verify it passes**

Run: `LD_LIBRARY_PATH="$(pwd)/runtimes/bin/ubuntu/python311/venv/lib/python3.11/site-packages/nvidia/cu13/lib" runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ImprovedFaceProcessing/tests/test_runtime_env.py -v`
Expected: PASS.

- [ ] **Step 5: Commit** (test file only)

```bash
git add modules/ImprovedFaceProcessing/tests/test_runtime_env.py
git commit -m "test(improved-face): runtime env check for onnxruntime GPU provider"
```

---

### Task 3: Device/tier selection (`options.py`)

**Files:**
- Create: `modules/ImprovedFaceProcessing/intelligencelayer/options.py`
- Test: `modules/ImprovedFaceProcessing/tests/test_options.py`

**Interfaces:**
- Produces: `select_tier(has_cuda: bool, vram_bytes: int, override: str) -> str` returning `"accurate"|"fast"`; `onnx_providers(has_cuda: bool) -> list[str]`; a `Options` object exposing `tier`, `use_cuda`, `providers`, `data_dir`, `models_dir`, `detector_path`, `recognizer_path`, `threshold` (float default), `db_path`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_options.py
import importlib.util, os
BASE = os.path.dirname(os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location("options", os.path.join(BASE,"intelligencelayer","options.py"))
opt = importlib.util.module_from_spec(spec); spec.loader.exec_module(opt)

def test_tier_selection():
    assert opt.select_tier(True, 32*10**9, "auto") == "accurate"   # 5090
    assert opt.select_tier(True,  8*10**9, "auto") == "accurate"   # 3070 (>=6GB)
    assert opt.select_tier(True,  4*10**9, "auto") == "fast"       # low-VRAM GPU
    assert opt.select_tier(False, 0,       "auto") == "fast"       # CPU
    assert opt.select_tier(True,  4*10**9, "accurate") == "accurate"  # override wins
    assert opt.select_tier(True, 32*10**9, "fast") == "fast"

def test_onnx_providers():
    assert opt.onnx_providers(True)[0] == "CUDAExecutionProvider"
    assert opt.onnx_providers(True)[-1] == "CPUExecutionProvider"
    assert opt.onnx_providers(False) == ["CPUExecutionProvider"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ImprovedFaceProcessing/tests/test_options.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write `options.py`**

```python
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

# Per-tier model filenames + a starting cosine threshold (recalibrated in Task 9).
TIER = {
    "accurate": {"detector": "scrfd_10g.onnx",  "recognizer": "adaface_ir101.pt", "det_size": 640, "threshold": 0.28},
    "fast":     {"detector": "scrfd_2.5g.onnx", "recognizer": "adaface_ir50.pt",  "det_size": 640, "threshold": 0.28},
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ImprovedFaceProcessing/tests/test_options.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/ImprovedFaceProcessing/intelligencelayer/options.py modules/ImprovedFaceProcessing/tests/test_options.py
git commit -m "feat(improved-face): device/VRAM tier selection + onnx providers"
```

---

### Task 4: SCRFD detector wrapper (`detector.py`)

**Files:**
- Create: `modules/ImprovedFaceProcessing/intelligencelayer/detector.py`
- Test: `modules/ImprovedFaceProcessing/tests/test_detector.py`

**Interfaces:**
- Consumes: `options.Options` (`detector_path`, `providers`, `det_size`).
- Produces: `ScrfdDetector(model_path, providers, det_size).detect(bgr_img) -> list[Face]` where `Face` has `.bbox (x1,y1,x2,y2 ints)`, `.score float`, `.kps (5,2) np.ndarray`.

- [ ] **Step 1: Write the failing test** (integration — needs weights from Task 1 Step 6)

```python
# tests/test_detector.py
import importlib.util, os, cv2, pytest
BASE = os.path.dirname(os.path.dirname(__file__))
DET  = os.path.join(BASE, "assets", "scrfd_10g.onnx")
IMG  = "src/demos/TestData/Faces/family-on-couch.jpg"

def _load():
    spec = importlib.util.spec_from_file_location("detector", os.path.join(BASE,"intelligencelayer","detector.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

@pytest.mark.skipif(not os.path.exists(DET), reason="weights not downloaded")
def test_detect_finds_faces_with_landmarks():
    det = _load().ScrfdDetector(DET, ["CPUExecutionProvider"], 640)
    faces = det.detect(cv2.imread(IMG))
    assert len(faces) >= 3
    f = faces[0]
    assert f.kps.shape == (5, 2)
    assert len(f.bbox) == 4 and f.score > 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `LD_LIBRARY_PATH="$(pwd)/runtimes/bin/ubuntu/python311/venv/lib/python3.11/site-packages/nvidia/cu13/lib" runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ImprovedFaceProcessing/tests/test_detector.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write `detector.py`**

```python
import os
# Ensure onnxruntime can find torch's cu13 CUDA libs BEFORE it is imported.
def _add_cu13_libpath():
    here = os.path.dirname(os.path.realpath(__file__))
    venv = os.path.normpath(os.path.join(here, "..", "..", "..",
             "runtimes", "bin", "ubuntu", "python311", "venv",
             "lib", "python3.11", "site-packages", "nvidia", "cu13", "lib"))
    if os.path.isdir(venv):
        os.environ["LD_LIBRARY_PATH"] = venv + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
_add_cu13_libpath()

from collections import namedtuple
from insightface.model_zoo import model_zoo

Face = namedtuple("Face", ["bbox", "score", "kps"])

class ScrfdDetector:
    def __init__(self, model_path, providers, det_size=640):
        self.model = model_zoo.get_model(model_path, providers=providers)
        ctx_id = 0 if any("CUDA" in p for p in providers) else -1
        self.model.prepare(ctx_id=ctx_id, input_size=(det_size, det_size))

    def detect(self, bgr_img):
        bboxes, kpss = self.model.detect(bgr_img, max_num=0, metric="default")
        faces = []
        for i, b in enumerate(bboxes):
            x1, y1, x2, y2, score = b
            kps = kpss[i] if kpss is not None else None
            faces.append(Face((int(x1), int(y1), int(x2), int(y2)), float(score), kps))
        return faces
```

- [ ] **Step 4: Run test to verify it passes**

Run: `LD_LIBRARY_PATH="$(pwd)/runtimes/bin/ubuntu/python311/venv/lib/python3.11/site-packages/nvidia/cu13/lib" runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ImprovedFaceProcessing/tests/test_detector.py -v`
Expected: PASS (≥3 faces, 5-pt landmarks).

- [ ] **Step 5: Commit**

```bash
git add modules/ImprovedFaceProcessing/intelligencelayer/detector.py modules/ImprovedFaceProcessing/tests/test_detector.py
git commit -m "feat(improved-face): SCRFD detector wrapper (onnxruntime)"
```

---

### Task 5: Alignment (`align.py`)

**Files:**
- Create: `modules/ImprovedFaceProcessing/intelligencelayer/align.py`
- Test: `modules/ImprovedFaceProcessing/tests/test_align.py`

**Interfaces:**
- Consumes: a `Face.kps` `(5,2)` array + the BGR image.
- Produces: `align_face(bgr_img, kps) -> np.ndarray` shape `(112,112,3)` BGR.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_align.py
import importlib.util, os, cv2, numpy as np, pytest
BASE = os.path.dirname(os.path.dirname(__file__))
DET  = os.path.join(BASE, "assets", "scrfd_10g.onnx")
IMG  = "src/demos/TestData/Faces/family-on-couch.jpg"

def _mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE,"intelligencelayer",name+".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

@pytest.mark.skipif(not os.path.exists(DET), reason="weights not downloaded")
def test_align_returns_112():
    det = _mod("detector").ScrfdDetector(DET, ["CPUExecutionProvider"], 640)
    face = det.detect(cv2.imread(IMG))[0]
    crop = _mod("align").align_face(cv2.imread(IMG), face.kps)
    assert crop.shape == (112, 112, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `LD_LIBRARY_PATH="$(pwd)/runtimes/bin/ubuntu/python311/venv/lib/python3.11/site-packages/nvidia/cu13/lib" runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ImprovedFaceProcessing/tests/test_align.py -v`
Expected: FAIL.

- [ ] **Step 3: Write `align.py`**

```python
from insightface.utils import face_align

def align_face(bgr_img, kps, image_size=112):
    return face_align.norm_crop(bgr_img, kps, image_size=image_size)
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/ImprovedFaceProcessing/intelligencelayer/align.py modules/ImprovedFaceProcessing/tests/test_align.py
git commit -m "feat(improved-face): 5-point norm_crop alignment"
```

---

### Task 6: AdaFace recognizer (`net.py` vendored + `recognizer.py`)

**Files:**
- Create: `modules/ImprovedFaceProcessing/intelligencelayer/adaface_net.py` (vendored from the AdaFace repo `net.py`)
- Create: `modules/ImprovedFaceProcessing/intelligencelayer/recognizer.py`
- Test: `modules/ImprovedFaceProcessing/tests/test_recognizer.py`

**Interfaces:**
- Consumes: a `(112,112,3)` BGR aligned crop; `options` (`recognizer_path`, `use_cuda`, `tier`).
- Produces: `AdaFaceRecognizer(model_path, arch, use_cuda).embed(bgr_crop) -> np.ndarray (512,) L2-normalized`; and `embed_batch(list_of_crops) -> np.ndarray (N,512)`.

- [ ] **Step 1: Vendor the AdaFace network definition**

Copy `net.py` from the AdaFace repo (mk-minchul/AdaFace, MIT) to `adaface_net.py`. It exposes `build_model(arch)` where `arch` ∈ {`"ir_50"`,`"ir_101"`}. Keep attribution header. (This mirrors how FaceProcessing vendors `recognition/networks.py`.)

- [ ] **Step 2: Write the failing test**

```python
# tests/test_recognizer.py
import importlib.util, os, cv2, numpy as np, pytest
BASE = os.path.dirname(os.path.dirname(__file__))
DET  = os.path.join(BASE, "assets", "scrfd_10g.onnx")
REC  = os.path.join(BASE, "assets", "adaface_ir101.pt")

def _mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE,"intelligencelayer",name+".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

@pytest.mark.skipif(not (os.path.exists(DET) and os.path.exists(REC)), reason="weights not downloaded")
def test_embedding_is_normalized_and_discriminative():
    det   = _mod("detector").ScrfdDetector(DET, ["CPUExecutionProvider"], 640)
    align = _mod("align").align_face
    rec   = _mod("recognizer").AdaFaceRecognizer(REC, "ir_101", use_cuda=False)

    def emb(path):
        img = cv2.imread(path); f = det.detect(img)[0]
        return rec.embed(align(img, f.kps))

    e1 = emb("src/demos/TestData/Faces/Chris-Hemsworth-2.jpg")
    e2 = emb("src/demos/TestData/Faces/chris-hemsworth-1.jpg")
    e3 = emb("src/demos/TestData/Faces/scarlett-johanson-1.jpg")
    assert e1.shape == (512,)
    assert abs(np.linalg.norm(e1) - 1.0) < 1e-3          # L2-normalized
    same = float(e1 @ e2); diff = float(e1 @ e3)
    assert same > diff and same > 0.3                     # same person more similar
```

- [ ] **Step 3: Run test to verify it fails**

Run: `runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ImprovedFaceProcessing/tests/test_recognizer.py -v`
Expected: FAIL (`recognizer.py` missing).

- [ ] **Step 4: Write `recognizer.py`**

```python
import numpy as np, torch
from adaface_net import build_model

class AdaFaceRecognizer:
    def __init__(self, model_path, arch, use_cuda):
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.model = build_model(arch)
        # AdaFace checkpoints store weights under "state_dict" with a "model." prefix.
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        sd = { k[6:]: v for k, v in sd.items() if k.startswith("model.") } or sd
        self.model.load_state_dict(sd)
        self.model.to(self.device).eval()

    @staticmethod
    def _to_input(bgr_crop):
        # AdaFace preprocessing: BGR, scaled to [-1,1], CHW.
        t = ((bgr_crop[:, :, ::-1].astype("float32") / 255.0) - 0.5) / 0.5
        return torch.from_numpy(t.transpose(2, 0, 1).copy()).unsqueeze(0)

    @torch.no_grad()
    def embed_batch(self, crops):
        batch = torch.cat([self._to_input(c) for c in crops]).to(self.device)
        feats, _ = self.model(batch)          # AdaFace returns (embedding, norm)
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy()

    def embed(self, bgr_crop):
        return self.embed_batch([bgr_crop])[0]
```

If the vendored `build_model` returns only the embedding (not a tuple), adjust the `feats, _ =` line accordingly during implementation and re-run the test.

- [ ] **Step 5: Run test to verify it passes**

Run: same as Step 3. Expected: PASS (normalized + same>diff).

- [ ] **Step 6: Commit**

```bash
git add modules/ImprovedFaceProcessing/intelligencelayer/adaface_net.py modules/ImprovedFaceProcessing/intelligencelayer/recognizer.py modules/ImprovedFaceProcessing/tests/test_recognizer.py
git commit -m "feat(improved-face): AdaFace recognizer (vendored net + embedding)"
```

---

### Task 7: Embedding gallery / SQLite store (`gallery.py`)

**Files:**
- Create: `modules/ImprovedFaceProcessing/intelligencelayer/gallery.py`
- Test: `modules/ImprovedFaceProcessing/tests/test_gallery.py`

**Interfaces:**
- Produces: `Gallery(db_path)` with `add(userid, embedding: np.ndarray)`, `delete(userid) -> int`, `list_ids() -> list[str]`, `load() -> None` (into memory), and `match(embedding, threshold) -> (userid, similarity)` (returns `("unknown", best_sim)` below threshold). Embeddings stored as text (repr of list), matching FaceProcessing's format.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gallery.py
import importlib.util, os, numpy as np, tempfile
BASE = os.path.dirname(os.path.dirname(__file__))
def _load():
    spec = importlib.util.spec_from_file_location("gallery", os.path.join(BASE,"intelligencelayer","gallery.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def test_add_list_match_delete():
    G = _load().Gallery(os.path.join(tempfile.mkdtemp(), "t.db"))
    a = np.ones(512, dtype="float32");  a /= np.linalg.norm(a)
    b = np.arange(512, dtype="float32"); b /= np.linalg.norm(b)
    G.add("alice", a); G.add("bob", b); G.load()
    assert set(G.list_ids()) == {"alice", "bob"}
    uid, sim = G.match(a, threshold=0.5)
    assert uid == "alice" and sim > 0.99
    uid2, _ = G.match(np.zeros(512, dtype="float32"), threshold=0.5)
    assert uid2 == "unknown"
    assert G.delete("alice") == 1
    G.load(); assert G.list_ids() == ["bob"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ImprovedFaceProcessing/tests/test_gallery.py -v`
Expected: FAIL.

- [ ] **Step 3: Write `gallery.py`**

```python
import os, sqlite3, threading, numpy as np

CREATE = "CREATE TABLE IF NOT EXISTS TB_EMBEDDINGS(userid TEXT PRIMARY KEY, embedding TEXT NOT NULL)"

class Gallery:
    def __init__(self, db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._lock = threading.Lock()
        self._ids = []
        self._mat = None                      # (N,512) float32
        with sqlite3.connect(self.db_path) as c:
            c.execute(CREATE)

    def add(self, userid, embedding):
        text = repr(embedding.astype("float32").tolist())
        with sqlite3.connect(self.db_path) as c:
            c.execute("INSERT INTO TB_EMBEDDINGS(userid,embedding) VALUES(?,?) "
                      "ON CONFLICT(userid) DO UPDATE SET embedding=excluded.embedding", (userid, text))

    def delete(self, userid):
        with sqlite3.connect(self.db_path) as c:
            cur = c.execute("DELETE FROM TB_EMBEDDINGS WHERE userid=?", (userid,))
            return cur.rowcount

    def list_ids(self):
        with sqlite3.connect(self.db_path) as c:
            return [r[0] for r in c.execute("SELECT userid FROM TB_EMBEDDINGS")]

    def load(self):
        ids, vecs = [], []
        with sqlite3.connect(self.db_path) as c:
            for uid, text in c.execute("SELECT userid, embedding FROM TB_EMBEDDINGS"):
                ids.append(uid); vecs.append(np.asarray(eval(text), dtype="float32"))
        with self._lock:
            self._ids = ids
            self._mat = np.stack(vecs) if vecs else None

    def match(self, embedding, threshold):
        with self._lock:
            ids, mat = self._ids, self._mat
        if mat is None or len(ids) == 0:
            return "unknown", 0.0
        sims = mat @ embedding.astype("float32")
        i = int(sims.argmax()); best = float(sims[i])
        return (ids[i], best) if best >= threshold else ("unknown", best)
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/ImprovedFaceProcessing/intelligencelayer/gallery.py modules/ImprovedFaceProcessing/tests/test_gallery.py
git commit -m "feat(improved-face): SQLite embedding gallery + cosine match"
```

---

### Task 8: Module adapter + command handlers (`improved_face.py`)

**Files:**
- Create: `modules/ImprovedFaceProcessing/intelligencelayer/improved_face.py`
- Test: `modules/ImprovedFaceProcessing/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `options.Options`, `detector.ScrfdDetector`, `align.align_face`, `recognizer.AdaFaceRecognizer`, `gallery.Gallery`.
- Produces: `ImprovedFace_adapter(ModuleRunner)` with `initialise()`, `process(data)`, `selftest()`, and a testable `Pipeline` class exposing `detect(bgr) -> list[dict]`, `register(bgr, userid) -> dict`, `recognize(bgr, threshold) -> dict`, `match(bgr1, bgr2) -> dict` (JSON shapes identical to FaceProcessing).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import importlib.util, os, cv2, pytest
BASE = os.path.dirname(os.path.dirname(__file__))
DET  = os.path.join(BASE, "assets", "scrfd_10g.onnx")
REC  = os.path.join(BASE, "assets", "adaface_ir101.pt")

def _mod():
    spec = importlib.util.spec_from_file_location("improved_face", os.path.join(BASE,"intelligencelayer","improved_face.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

@pytest.mark.skipif(not (os.path.exists(DET) and os.path.exists(REC)), reason="weights not downloaded")
def test_register_then_recognize(tmp_path):
    P = _mod().Pipeline(DET, REC, "ir_101", use_cuda=False, db_path=str(tmp_path/"g.db"), threshold=0.2)
    ref = cv2.imread("src/demos/TestData/Faces/Chris-Hemsworth-2.jpg")
    assert P.register(ref, "chris")["success"] is True
    out = P.recognize(cv2.imread("src/demos/TestData/Faces/chris-hemsworth-1.jpg"), threshold=0.2)
    assert out["success"] is True
    ids = [p["userid"] for p in out["predictions"]]
    assert "chris" in ids

@pytest.mark.skipif(not os.path.exists(DET), reason="weights not downloaded")
def test_detect_shape():
    P = _mod().Pipeline(DET, REC, "ir_101", use_cuda=False, db_path=":memory:", threshold=0.2)
    out = P.detect(cv2.imread("src/demos/TestData/Faces/family-on-couch.jpg"))
    assert out["success"] is True and out["count"] >= 3
    assert {"confidence","x_min","y_min","x_max","y_max"} <= set(out["predictions"][0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `LD_LIBRARY_PATH="$(pwd)/runtimes/bin/ubuntu/python311/venv/lib/python3.11/site-packages/nvidia/cu13/lib" runtimes/bin/ubuntu/python311/venv/bin/python -m pytest modules/ImprovedFaceProcessing/tests/test_pipeline.py -v`
Expected: FAIL.

- [ ] **Step 3: Write `improved_face.py`**

Implement a `Pipeline` class composing detector+align+recognizer+gallery with methods returning FaceProcessing-shaped JSON:
- `detect(bgr)` → `{success, count, predictions:[{confidence,x_min,y_min,x_max,y_max}], inferenceMs}`
- `register(bgr, userid)` → detect (require exactly one face) → align → embed → `gallery.add` → `{success, message}`
- `recognize(bgr, threshold)` → detect → align+embed each → `gallery.match` → predictions `[{confidence,userid,x_min..}]`
- `match(bgr1,bgr2)` → embed the top face in each → cosine → `{success, similarity}`
Then an `ImprovedFace_adapter(ModuleRunner)` mirroring `Face_adapter`: `initialise()` builds `Options`, the `Pipeline`, loads the gallery, starts the 5-second refresh thread; `process(data)` dispatches the six commands using `data.get_image_from_request`/`data.get_value`; `selftest()` runs `detect` on `test/person.jpg`. Reuse FaceProcessing's GPU-OOM→CPU fallback (`try/except` around model init that flips `use_cuda=False` and rebuilds). Entry: `if __name__ == "__main__": ImprovedFace_adapter().start_loop()`.

Full handler code is derived directly from `modules/FaceProcessing/intelligencelayer/face.py` (same command semantics); keep the response keys identical.

- [ ] **Step 4: Run tests to verify they pass**

Run: same as Step 2. Expected: PASS (register→recognize finds "chris"; detect ≥3).

- [ ] **Step 5: Commit**

```bash
git add modules/ImprovedFaceProcessing/intelligencelayer/improved_face.py modules/ImprovedFaceProcessing/tests/test_pipeline.py
git commit -m "feat(improved-face): module adapter + six command handlers"
```

---

### Task 9: install.sh, requirements, post_install.sh, explore.html

**Files:**
- Create: `modules/ImprovedFaceProcessing/install.sh`
- Create: `modules/ImprovedFaceProcessing/post_install.sh`
- Create: `modules/ImprovedFaceProcessing/requirements.linux.cuda12.txt`
- Create: `modules/ImprovedFaceProcessing/requirements.linux.cuda.txt`
- Create: `modules/ImprovedFaceProcessing/requirements.txt`
- Create: `modules/ImprovedFaceProcessing/explore.html` (copy/adapt FaceProcessing's)
- Create: `modules/ImprovedFaceProcessing/test/person.jpg` (copy from FaceProcessing/test)

**Interfaces:**
- Produces: a module that `bash ../../src/setup.sh` installs end-to-end (venv deps + weights) and whose self-test passes on the 5090.

- [ ] **Step 1: Write requirements files**

`requirements.linux.cuda12.txt` and `requirements.linux.cuda.txt` (identical, GPU):
```
#! Python3.11
numpy
opencv-python
Pillow
scikit-image
--extra-index-url https://download.pytorch.org/whl/cu130
torch==2.13.0+cu130
--extra-index-url https://download.pytorch.org/whl/cu130
torchvision==0.28.0+cu130
# NOTE: cuDNN 9.24 + onnxruntime cleanup handled in post_install.sh
onnxruntime-gpu
insightface
CodeProject-AI-SDK
```
`requirements.txt` (CPU):
```
#! Python3.11
numpy
opencv-python
Pillow
scikit-image
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.13.0+cpu
--extra-index-url https://download.pytorch.org/whl/cpu
torchvision==0.28.0+cpu
onnxruntime
insightface
CodeProject-AI-SDK
```

- [ ] **Step 2: Write `install.sh`**

Mirror FaceProcessing's `install.sh` guard, then download weights for the tier(s) via `download_models.py`:
```bash
if [ "$1" != "install" ]; then
    read -t 3 -p "This script is only called from: bash ../../src/setup.sh"; echo; exit 1
fi
# Download model weights into assets (both tiers so the module can switch at runtime)
"${venvPythonCmdPath}" "${moduleDirPath}/intelligencelayer/download_models.py" --dest "${moduleDirPath}/assets" || \
    moduleInstallErrors="Failed to download face models"
```
(Add an `if __name__=="__main__"` argparse block to `download_models.py` that downloads both tiers to `--dest`.)

- [ ] **Step 3: Write `post_install.sh`**

```bash
if [ "$1" != "post-install" ]; then
    read -t 3 -p "This script is only called from: bash ../../src/setup.sh"; echo; exit 1
fi
# Restore pkg_resources for insightface/torch tooling
"${venvPythonCmdPath}" -m pip install "setuptools<81"
# insightface pulls CPU onnxruntime which shadows the GPU CUDA provider — remove it
if [ "${installGPU}" = "true" ] && [ "${hasCUDA}" = "true" ]; then
    "${venvPythonCmdPath}" -m pip uninstall -y onnxruntime >/dev/null 2>&1
    "${venvPythonCmdPath}" -m pip install --upgrade "nvidia-cudnn-cu13==9.24.0.43"
    if [ $? -gt 0 ]; then moduleInstallErrors="Failed to install cuDNN 9.24.0.43"; fi
fi
```

- [ ] **Step 4: Copy explore.html + test image**

```bash
cp modules/FaceProcessing/explore.html modules/ImprovedFaceProcessing/explore.html
mkdir -p modules/ImprovedFaceProcessing/test
cp modules/FaceProcessing/test/person.jpg modules/ImprovedFaceProcessing/test/person.jpg
```
Update the title/text in `explore.html` to "Improved Face Processing".

- [ ] **Step 5: Run full module setup**

```bash
cd modules/ImprovedFaceProcessing && bash ../../src/setup.sh --verbosity info 2>&1 | tee /tmp/ifp_setup.log; cd -
grep -E "Self-test|Setup complete|SETUP FAILED" /tmp/ifp_setup.log
```
Expected: `Self-test passed`, `Setup complete`, exit 0. If the self-test fails, inspect and fix before committing.

- [ ] **Step 6: Commit**

```bash
git add modules/ImprovedFaceProcessing/install.sh modules/ImprovedFaceProcessing/post_install.sh modules/ImprovedFaceProcessing/requirements*.txt modules/ImprovedFaceProcessing/explore.html modules/ImprovedFaceProcessing/test/person.jpg
git commit -m "feat(improved-face): install/requirements/post_install + self-test passing"
```

---

### Task 10: Server integration, A/B validation, and measured latency

**Files:**
- Modify: `modules/FaceProcessing/modulesettings.json` (set `LaunchSettings.AutoStart` to `false` — disable to free the routes)
- Create: `docs/superpowers/plans/improved-face-benchmark.md` (results record)

**Interfaces:**
- Consumes: the running server.

- [ ] **Step 1: Disable FaceProcessing to free the `/v1/vision/face` routes**

Set `AutoStart: false` in `modules/FaceProcessing/modulesettings.json` so only ImprovedFaceProcessing serves the routes.

- [ ] **Step 2: Restart server and confirm ImprovedFaceProcessing starts**

```bash
kill $(cat /tmp/cpai_server.pid) 2>/dev/null; sleep 3
cd src/server && ASPNETCORE_ENVIRONMENT=Development dotnet bin/Debug/net9.0/CodeProject.AI.Server.dll >/tmp/cpai_server.log 2>&1 & echo $! > /tmp/cpai_server.pid; cd -
# wait for Started
```
Verify `ImprovedFaceProcessing` reaches status `Started` and reports `GPU in use`.

- [ ] **Step 3: Live face API smoke test**

```bash
curl -s -X POST http://127.0.0.1:32168/v1/vision/face -F image=@src/demos/TestData/Faces/family-on-couch.jpg | python3 -m json.tool
```
Expected: `success: true`, `count >= 3`, `inferenceDevice: GPU`.

- [ ] **Step 4: A/B recall + measured latency**

Write a throwaway script comparing detect counts on all `src/demos/TestData/Faces/*.jpg` between the two modules (temporarily re-enabling FaceProcessing on a scratch run if needed), and time 50 recognize calls to record median latency on GPU. Repeat with `MODEL_TIER=fast` / CPU torch to record the CPU-tier latency. Save numbers to `docs/superpowers/plans/improved-face-benchmark.md`.

- [ ] **Step 5: Commit**

```bash
git add modules/FaceProcessing/modulesettings.json docs/superpowers/plans/improved-face-benchmark.md
git commit -m "feat(improved-face): make successor default; record A/B + latency benchmarks"
```

---

## Notes for the implementer

- Always export `LD_LIBRARY_PATH=<venv>/lib/python3.11/site-packages/nvidia/cu13/lib` before running any GPU onnxruntime test, or import `detector` first (its `_add_cu13_libpath()` sets it for the process — but the C++ loader reads it at process start, so the env-var export in the test command is the reliable path).
- The `weights_only=False` in `recognizer.py` mirrors the fix already used across the cu130 modules (trusted local weights).
- Keep every response JSON shape byte-for-byte compatible with FaceProcessing; the `explore.html` and existing clients depend on it.
- Model weights are large; Task 1 Step 6 must complete before Tasks 4–8 tests can pass (they `skipif` when weights are absent).
