# Improved Face Processing — Design Spec (v1.0.0)

**Date:** 2026-07-19
**Status:** Approved design (pre-implementation)
**Module ID:** `ImprovedFaceProcessing` · **Name:** "Improved Face Processing" · **Version:** 1.0.0

## 1. Purpose & background

A new CodeProject.AI Python analysis module that replaces the existing `FaceProcessing`
module with an accuracy-first pipeline, based on the 2026 research findings:

- **Detection:** YOLOv5 (`face.pt`) → **SCRFD** (better hard-case recall; found 4 faces vs
  YOLOv5's 3 on the test image, with 5-point landmarks).
- **Recognition:** IR-SE ResNet-50 ArcFace → **AdaFace** (quality-adaptive margin; stronger on
  low-quality/mixed faces), 512-d embeddings.
- **Alignment:** add proper 5-point similarity-transform alignment (`norm_crop` → 112×112)
  between detect and recognize — the current module's main accuracy gap.

Prioritize **accuracy**; support a range of hardware (high-end GPU, low GPU, CPU-only).

## 2. Scope

**In scope (v1.0.0):** full API parity with FaceProcessing — commands `detect`, `recognize`,
`register`, `match`, `list`, `delete` on the existing `/v1/vision/face*` routes; auto-tiered
models by hardware; GPU and CPU installs; own face database; module self-test.

**Out of scope (future):** FAISS/ANN vector index (SQLite cosine scan is sufficient at v1.0.0
gallery sizes); migrating existing registered faces (impossible across embedding spaces —
users re-register); commercial-license-clean weights (uses research-oriented weights, same
posture as the existing module).

## 3. Identity & coexistence

- New folder: `modules/ImprovedFaceProcessing/`.
- Own queue: `improvedfaceprocessing_queue`.
- Own database: `/etc/codeproject/ai/improved_faceembedding.db` (schema identical to
  FaceProcessing's `TB_EMBEDDINGS(userid TEXT PRIMARY KEY, embedding TEXT NOT NULL)`).
- **Successor** to FaceProcessing: it claims the same `/v1/vision/face*` routes. Two modules
  cannot own the same route, so **only one of `FaceProcessing` / `ImprovedFaceProcessing`
  may be enabled at a time.** Enabling this module implies disabling the old one; documented
  in the module README/description.
- **Face re-registration required:** AdaFace embeddings live in a different vector space than
  the old IR-SE-ArcFace ones. The old DB is not reused. Only embeddings (not source images)
  are stored, so there is nothing to auto-recompute — users re-register faces.

## 4. Models & auto-tiering

Tier is chosen at startup from `torch.cuda.is_available()` and, when a GPU is present,
`torch.cuda.get_device_properties(0).total_memory`. A manual override setting
(`CPAI_MODULE_..._MODEL_TIER` / a modulesettings `EnvironmentVariable`, values `accurate|fast|auto`)
takes precedence.

| Tier | Selected when | Detector | Recognizer |
|---|---|---|---|
| **accurate** | CUDA GPU with ≥ ~6 GB VRAM | SCRFD-10G | AdaFace IR-101 |
| **fast** | CPU-only, or GPU < ~6 GB VRAM | SCRFD-2.5G (fallback 500M) | AdaFace IR-50 |

- **SCRFD** weights: the detection `.onnx` from InsightFace model packs (buffalo_l → SCRFD-10G,
  buffalo_m → SCRFD-2.5G, buffalo_s → SCRFD-500M). Only the detection model is used (not the
  full FaceAnalysis pack's recognition/genderage models).
- **AdaFace** weights: PyTorch `.pt` from the AdaFace / CVLface HuggingFace repos
  (IR-101 WebFace12M; IR-50). MIT code; research-oriented weights, attributed.
- Embedding: 512-d, L2-normalized, cosine similarity.

## 5. Data flow

```
image bytes
  → SCRFD detect            (onnxruntime; CUDA provider on GPU, CPU provider otherwise)
      → boxes + det_score + 5-point landmarks
  → norm_crop align         (insightface.utils.face_align → 112×112 per face)
  → AdaFace embed           (PyTorch; .cuda() on GPU, CPU otherwise) → 512-d, L2-normalized
  → cosine compare vs gallery (in-memory tensor of registered embeddings)
      → best match; if max_similarity < threshold → "unknown"
```

Command handlers mirror FaceProcessing's structure (`detect`, `register`, `list`, `delete`,
`recognize`, `match`), returning the same JSON response shapes so existing clients keep working.

## 6. Matching

- SQLite + in-memory cosine scan, with the same 5-second background refresh thread as
  FaceProcessing (`_update_faces`), against the module's own DB.
- **Threshold recalibration:** AdaFace cosine similarity operates at a different scale than the
  current 0.67. Pick a validated default for the chosen recognizer and expose it as the
  `min_confidence` request parameter (same knob clients already use).
- FAISS/HNSW noted as a future upgrade for large galleries; not in v1.0.0.

## 7. Runtime, venv & install (multi-hardware)

### 7.1 GPU install
- `requirements.linux.cuda12.txt` (selected by CodeProject.AI when CUDA is detected):
  torch `2.13.0+cu130` + torchvision `0.28.0+cu130` (`--extra-index-url .../cu130`) +
  **`onnxruntime-gpu`** + insightface + opencv + numpy + SDK.
- Shares the python3.11 **Shared** venv with the other cu130 modules.
- **`post_install.sh`** (runs after pip install, before self-test):
  1. `pip install "setuptools<81"` (restore `pkg_resources`).
  2. `pip install --upgrade "nvidia-cudnn-cu13==9.24.0.43"` (Blackwell conv fix).
  3. Ensure **only `onnxruntime-gpu`** is installed — if the CPU `onnxruntime` package is
     present (pulled transitively by insightface), uninstall it so it does not shadow the
     CUDA provider.
- **onnxruntime CUDA-lib discovery:** onnxruntime-gpu 1.27 is a CUDA-13 build and needs
  `libcudart.so.13` etc. These are provided by torch's cu13 wheels at
  `<venv>/lib/python3.11/site-packages/nvidia/cu13/lib`. The module must add that directory to
  `LD_LIBRARY_PATH` at launch (via modulesettings `EnvironmentVariables` and/or an in-process
  `os.environ`/`add_dll_directory` shim before importing onnxruntime). **Spike-verified:** with
  this path set, providers become `[TensorRT, CUDA, CPU]` and SCRFD runs on the RTX 5090 at
  ~38 ms (vs ~1.3 s on CPU).
- **Driver floor:** cu130 / onnxruntime-CUDA-13 requires a recent NVIDIA driver (CUDA 13,
  ~2025+). Supports Ampere (RTX 3070, sm_86) and Blackwell (RTX 5090, sm_120) — both are in
  the torch 2.13+cu130 arch list. Documented as a requirement.

### 7.2 CPU install
- `requirements.txt`: torch `2.13.0+cpu` + torchvision `0.28.0+cpu` (`--extra-index-url .../cpu`)
  + **`onnxruntime`** (CPU) + insightface + opencv + numpy + SDK. No cuDNN pin needed (still
  pin `setuptools<81`). Uses the **fast** tier.

### 7.3 Runtime device handling
- At startup: detect device (`torch.cuda.is_available()`), select tier, set onnxruntime
  providers (`CUDAExecutionProvider` then `CPUExecutionProvider`, or CPU only), verify the
  active provider and log it.
- Keep the existing GPU-OOM → CPU fallback (`_init_models` re-entry) from FaceProcessing.

### 7.4 Weight download
- `install.sh` downloads the tier's SCRFD `.onnx` (InsightFace) and AdaFace `.pt`
  (HuggingFace) into `assets/`. Not via `getFromServer` (that is CodeProject-CDN-only).
  Use direct download (curl/wget or `huggingface_hub`), with checksums where available.

## 8. Module scaffolding (follow existing conventions)

- `modulesettings.json`: `LaunchSettings` (Runtime `python3.11`, RuntimeLocation `Shared`,
  FilePath the adapter entry, Queue `improvedfaceprocessing_queue`), `RouteMaps` for the six
  `vision/face*` routes (Route/Method/Command/Inputs/Outputs), `EnvironmentVariables`
  (MODELS_DIR, model tier, `LD_LIBRARY_PATH` addition), `ModelRequirements`, `ModuleReleases`
  with a 1.0.0 entry, plus platform overrides as needed.
- Python entry: an adapter class subclassing the SDK `ModuleRunner`, calling `.start_loop()`
  (same pattern as `FaceProcessing`'s `Face_adapter`).
- Code layout mirrors FaceProcessing: entry adapter + a detector wrapper (SCRFD/onnxruntime) +
  a recognizer wrapper (AdaFace/PyTorch) + alignment + DB/embedding-store + shared options.
- `install.sh`, `post_install.sh`, `requirements*.txt`, `explore.html`, `test/` assets.

## 9. Testing

- **Self-test:** run `detect` on a bundled face image; assert success, faces found, and the
  expected device/provider.
- **A/B validation:** compare against the running FaceProcessing on the existing
  `src/demos/TestData/Faces/*` images — detection recall and recognition behavior.
- **Latency:** record measured per-tier latency on the RTX 5090 (GPU) and on CPU — fills the
  gap the research could not cite from public sources.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| onnxruntime can't find CUDA-13 libs | Add `<venv>/.../nvidia/cu13/lib` to `LD_LIBRARY_PATH` (spike-proven) |
| CPU `onnxruntime` shadows GPU build | Install only `onnxruntime-gpu`; `post_install` removes CPU pkg; verify provider at startup |
| GPU driver too old for CUDA 13 | Document recent-driver requirement; CPU path always works |
| AdaFace threshold ≠ 0.67 | Recalibrate default; expose `min_confidence` |
| Model too heavy for CPU/low GPU | Auto-tier to SCRFD-2.5G/500M + AdaFace IR-50 |
| Weight licensing | Research-oriented weights, same posture as existing module, attributed |
| Registered faces don't carry over | Documented; users re-register (different embedding space) |

## 11. Open items to resolve during implementation

- Exact AdaFace weight files/URLs per tier and their default cosine thresholds (validate on
  `TestData/Faces`).
- Whether onnxruntime GPU lib discovery is best done via modulesettings `LD_LIBRARY_PATH` or an
  in-process shim before `import onnxruntime` (test both; prefer the more robust on Windows too).
- VRAM cutoff for the tier switch (~6 GB is a starting estimate; confirm AdaFace IR-101 +
  SCRFD-10G peak VRAM).
