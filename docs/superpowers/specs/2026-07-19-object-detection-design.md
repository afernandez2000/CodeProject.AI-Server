# Object Detection module — Design Spec (v1.0.0)

**Date:** 2026-07-19
**Status:** Approved design (pre-implementation)
**Module ID:** `ObjectDetection` · **Name:** "Object Detection" · **Version:** 1.0.0

## 1. Purpose & background

A new CodeProject.AI object-detection module, the successor to
`ObjectDetectionYOLOv5-6.2`, built on 2026 best-practice models. Primary consumer is
**Blue Iris 6** (CCTV/NVR) sending camera snapshots to detect people, vehicles, animals,
packages, etc. It must be **API-compatible with what Blue Iris already calls**, run
**cross-platform (Windows + Linux)** with **hardware auto-tiering** (high-end GPU / low-end
GPU / CPU-only), and **keep loading the existing community YOLO custom models** that Blue
Iris users depend on.

Research findings that shaped this design (2026):
- Best YOLO ladder: **YOLO26** (Ultralytics, ~Jan 2026) n→x = 40.9→57.5 COCO mAP; nano is
  ~43% faster on CPU (ONNX) than YOLO11n. Alternative: YOLOv12 (40.6→55.2). All **AGPL-3.0**.
- Best accuracy overall: **RF-DETR** (Apache), but transformer + unproven CPU tier — not chosen.
- **Critical:** legacy **yolov5-6.2-era community `.pt` models (MikeLud IPcam-combined,
  IPcam-dark, delivery, license-plate, etc.) do NOT load in modern Ultralytics** (they pickle
  a `models.yolo` dependency and error). They require the original **yolov5** runtime.
- No verified YOLOv13. Community recommends the **Small** model as the fast tier.

## 2. Scope

**In scope (v1.0.0):** the three Blue-Iris routes (`detect`, `custom`, `list-custom`) with
byte-compatible JSON; a dual inference engine (Ultralytics YOLO26 default + yolov5 for legacy
custom models); hardware auto-tiering for the default model; GPU + CPU installs on Windows
and Linux; bundled default legacy IPcam custom models; module self-test.

**Out of scope (future):** ONNX/OpenVINO CPU acceleration (v1.0 uses PyTorch on all tiers —
noted as a future ~3× CPU speedup); RF-DETR / transformer detectors; SAHI slicing; training/
fine-tuning; per-class server-side filtering beyond `min_confidence`.

## 3. Identity & coexistence

- New folder `modules/ObjectDetection/`. Queue `objectdetection_queue`.
- **Successor** to `ObjectDetectionYOLOv5-6.2`: it claims the same `/v1/vision/detection` and
  `/v1/vision/custom/*` routes. Only one of the two modules may be enabled at a time; enabling
  this one implies disabling the old (handled cleanly by the server's deterministic
  route-ownership logic already in place — enabled module wins the shared route).
- **Stateless**: no database. Detection returns results per request.

## 4. Dual inference engine

1. **Ultralytics** (`ultralytics` package) running **YOLO26** — the built-in
   `/v1/vision/detection` default detector, plus modern custom models (v8 / v11 / v5u `.pt`).
2. **yolov5** (`yolov5==6.2.3`, the same runtime + fixes used by `ObjectDetectionYOLOv5-6.2`:
   `weights_only=False` load, torch-2.13 compat) — for **legacy** custom `.pt` models.

No `onnxruntime` in v1.0, so **no `_cuda_libpath` shim is required** (torch loads its own CUDA
libraries). This module is simpler on the plumbing side than ImprovedFaceProcessing.

## 5. Routes & Blue Iris parity

| Route | Method | Command | Engine |
|---|---|---|---|
| `vision/detection` | POST | `detect` | YOLO26 (tier-selected) |
| `vision/custom/<model>` | POST | `custom` | auto-routed (Ultralytics ↔ yolov5) |
| `vision/custom/list` | POST | `list-custom` | — |

Inputs: `image` (File), `min_confidence` (Float, default 0.4). Outputs mirror
`ObjectDetectionYOLOv5-6.2` exactly: `success`, `message`, `error`, `count`, `inferenceMs`,
`processMs`, and `predictions[]` each `{ label, confidence, x_min, y_min, x_max, y_max }`.
Response shapes must stay byte-compatible so existing Blue Iris configs work unchanged.

## 6. Models & auto-tiering

Tier chosen at startup from `torch.cuda.is_available()` + `torch.cuda.get_device_properties(0)
.total_memory`, with a manual override setting (`MODEL_TIER` = `auto|accurate|balanced|fast`).

| Tier | Selected when | Default model |
|---|---|---|
| **accurate** | CUDA GPU, VRAM ≥ ~8 GB | YOLO26-X (or -L) |
| **balanced** | CUDA GPU, VRAM < ~8 GB | YOLO26-M (or -S) |
| **fast** | CPU-only | YOLO26-N |

The **model family is a single config constant** (`options.py`): swapping `YOLO26` →
`YOLOv12` / `YOLO11` is a one-line change per the "easy switch later" decision. YOLO26 weights
auto-download via Ultralytics on first use (also pre-fetched by `install.sh`).

## 7. Custom-model routing

On `/v1/vision/custom/<model>`:
1. Resolve `<model>.pt` in the custom-models dir (return a clear error if missing).
2. Load with **Ultralytics** first; on the legacy-pickle failure (`models.yolo` /
   `weights_only` error), fall back to the **yolov5** runtime.
3. Cache the loaded model **and which engine** by name (refreshed like the existing module).
4. `list-custom` enumerates `*.pt` in the custom-models dir.

`install.sh`/`install.bat` ships a **default set of legacy IPcam custom models**
(ipcam-combined, ipcam-general, ipcam-animal, ipcam-dark, license-plate, delivery) from the
CodeProject CDN so existing Blue Iris configs work out of the box.

## 8. Data flow

```
default detect:
  image → YOLO26 (Ultralytics, tier model, conf=min_confidence)
        → predictions [{label, confidence, x_min, y_min, x_max, y_max}] → JSON

custom detect:
  image → resolve <model>.pt → engine(Ultralytics | yolov5) → predictions → JSON (same shape)
```

## 9. Runtime, venv & install (cross-platform)

### 9.1 GPU install (Linux + Windows)
- `requirements.linux.cuda12.txt` / `requirements.windows.cuda.txt`:
  `torch==2.13.0+cu130`, `torchvision==0.28.0+cu130` (index cu130), `ultralytics`,
  `yolov5==6.2.3`, opencv, numpy, SDK.
- Shares the python3.11 venv.
- `post_install.sh` / `post_install.bat`:
  1. `pip install "setuptools<81"` (restore `pkg_resources` for yolov5).
  2. `pip install --upgrade "nvidia-cudnn-cu13==9.24.0.43"` (torch 2.13's bundled cuDNN 9.20 is
     broken on RTX 50-series / sm_120 — every convolution fails without this).
  (No onnxruntime handling needed.)
- **Driver floor:** cu130 needs a recent (CUDA-13-capable) NVIDIA driver. Supports Ampere
  (3070, sm_86) through Blackwell (5090, sm_120).

### 9.2 CPU install
- `requirements.txt` / `requirements.windows.txt`: `torch==2.13.0+cpu`,
  `torchvision==0.28.0+cpu` (index cpu), `ultralytics`, `yolov5==6.2.3`, opencv, numpy, SDK.
  Uses the **fast** tier (YOLO26-N). PyTorch inference (ONNX/OpenVINO acceleration deferred).

### 9.3 Weight download
- `install.sh`/`install.bat` pre-download the tier's YOLO26 weights (via Ultralytics) and the
  default legacy IPcam custom-model set (CodeProject CDN). `.gitignore` keeps weights out of git.

### 9.4 Module scaffolding (follow ImprovedFaceProcessing / ObjectDetectionYOLOv5-6.2 patterns)
- `modulesettings.json` (+ `modulesettings.windows.json`): LaunchSettings (Runtime
  `python3.11`, Queue `objectdetection_queue`), the three RouteMaps, EnvironmentVariables
  (MODEL_TIER, MODELS_DIR, CUSTOM_MODELS_DIR), ModelRequirements, ModuleReleases (1.0.0).
- Python: an SDK `ModuleRunner` adapter (`detect_adapter.py`) + a detector wrapper per engine +
  `options.py` tiering. Mirror the existing object-detection module's structure where it fits.
- `install.sh` + `install.bat`, `post_install.sh` + `post_install.bat`, `requirements*.txt`,
  `explore.html`, `test/` sample image, `.gitignore`.

## 10. Testing

- **Self-test:** `detect` on a bundled image; assert success, objects found, and device/tier.
- **Custom-model tests:** load one **legacy** IPcam model (yolov5 path) and one **modern** `.pt`
  (Ultralytics path); assert both return valid predictions — this is the dual-engine guarantee.
- **Tier test:** tier selection by (has_cuda, vram, override).
- **Blue Iris shape test:** response JSON keys match `ObjectDetectionYOLOv5-6.2` exactly.
- **A/B + latency:** compare detections vs the old module on `TestData/Objects/*`; record
  measured RTX 5090 and CPU latency per tier.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| YOLO26 is very new (churn) | Model family is a one-line config swap → YOLOv12 / YOLO11 |
| Legacy custom `.pt` won't load in Ultralytics | yolov5 fallback engine (core of the design) |
| AGPL-3.0 (YOLO26 default) | Documented per weight; RF-DETR (Apache) noted as alt path |
| CPU too slow for CCTV | YOLO26-N fast tier; ONNX/OpenVINO ~3× boost noted as future |
| Driver too old for cu130 | Documented; CPU path always works |
| Windows untested from Linux dev box | Reuse verified cross-platform patterns; validate on the box |
| Blue Iris response drift | Response-shape test pinned against the existing module |

## 12. Open items to resolve during implementation

- Exact YOLO26 size per tier (X vs L for accurate; M vs S for balanced) — confirm VRAM peak.
- Exact source URLs for the bundled legacy IPcam custom models (CodeProject CDN asset names).
- Whether Ultralytics reliably auto-downloads YOLO26 weights offline vs needing an explicit URL.
- Confirm yolov5 + ultralytics coexist cleanly in one venv on torch 2.13 (spike during Task 1).
