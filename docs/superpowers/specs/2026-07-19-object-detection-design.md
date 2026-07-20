# Object Detection module — Design Spec (v1.0.0)

**Date:** 2026-07-19
**Status:** Approved design (pre-implementation)
**Module ID:** `ObjectDetection` · **Name:** "Object Detection" · **Version:** 1.0.0

## 1. Purpose & background

A new CodeProject.AI object-detection module, the successor to
`ObjectDetectionYOLOv5-6.2`, tuned for its primary consumer: **Blue Iris 6** (CCTV/NVR)
sending camera snapshots to detect people, vehicles, animals, packages, etc. It must be
**API-compatible with what Blue Iris already calls**, run **cross-platform (Windows + Linux)**
with **hardware auto-tiering** (high-end GPU / low-end GPU / CPU-only), and **keep loading the
existing community YOLO custom models** Blue Iris users depend on.

**Design principle (decided during brainstorming):** for CCTV, a **domain-trained** model
(trained on real surveillance imagery — CCTV angles, small/distant subjects, night IR) gives
**fewer false alarms** than a newer, generic COCO-trained model, even when the generic model has
a higher COCO mAP. **COCO mAP does not predict CCTV false-alarm rate.** Therefore the module's
**default detector is a CCTV-domain model**, with the newest generic model (YOLO26) offered as
a switchable alternative rather than the default.

Research findings (2026) that informed this:
- Best generic YOLO ladder: **YOLO26** (Ultralytics, ~Jan 2026) n→x = 40.9→57.5 COCO mAP; nano
  ~43% faster on CPU (ONNX) than YOLO11n. Alternative: YOLOv12. All **AGPL-3.0**.
- Best generic accuracy: **RF-DETR** (Apache), transformer, CPU tier unproven — not chosen.
- The valuable **CCTV custom models are MikeLud's IPcam set** (ipcam-combined, ipcam-general,
  ipcam-animal, ipcam-dark, license-plate, delivery), which are **legacy YOLOv5-6.2** and do
  **NOT load in modern Ultralytics** — they require the original **yolov5** runtime.
- The theoretical best ("modern backbone + CCTV data") needs **retraining** the IPcam classes
  on YOLO26/v11 — a separate project (dataset + GPU training), noted as future work.

## 2. Scope

**In scope (v1.0.0):** the three Blue-Iris routes (`detect`, `custom`, `list-custom`) with
byte-compatible JSON; a dual inference engine; a **configurable default detector shipped as a
CCTV-domain model** with hardware tiering; **YOLO26** as a switchable modern general-purpose
alternative; full legacy + modern custom-model loading; GPU + CPU installs on Windows and
Linux; bundled default CCTV + custom models; CCTV-sensible defaults (class filtering,
confidence); module self-test.

**Out of scope (future):** retraining IPcam classes on a modern backbone (the real "best
results" path); ONNX/OpenVINO CPU acceleration (v1.0 uses PyTorch on all tiers); RF-DETR /
transformer detectors; SAHI slicing; training/fine-tuning; auto low-light model switching.

## 3. Identity & coexistence

- New folder `modules/ObjectDetection/`. Queue `objectdetection_queue`.
- **Successor** to `ObjectDetectionYOLOv5-6.2`: it claims the same `/v1/vision/detection` and
  `/v1/vision/custom/*` routes. Only one of the two modules may be enabled at a time; enabling
  this one implies disabling the old (handled by the server's deterministic route-ownership
  logic already in place — enabled module wins the shared route).
- **Stateless**: no database.

## 4. Dual inference engine

1. **yolov5** (`yolov5==6.2.3`, the proven runtime + fixes from `ObjectDetectionYOLOv5-6.2`:
   `weights_only=False` load, torch-2.13 compat) — runs the **default CCTV-domain model** and
   all **legacy** custom `.pt` models. This is the primary path for Blue Iris users.
2. **Ultralytics** (`ultralytics` package) running **YOLO26** — the switchable modern
   general-purpose detector, plus modern custom models (v8 / v11 / v5u `.pt`).

No `onnxruntime` in v1.0, so **no `_cuda_libpath` shim is required** (torch loads its own CUDA
libraries). Both engines are torch-based and coexist in one venv (verified in Task 1 spike).

## 5. Routes & Blue Iris parity

| Route | Method | Command | Engine |
|---|---|---|---|
| `vision/detection` | POST | `detect` | default model (CCTV via yolov5, or YOLO26 via Ultralytics per `DEFAULT_MODEL`) |
| `vision/custom/<model>` | POST | `custom` | auto-routed (Ultralytics ↔ yolov5) |
| `vision/custom/list` | POST | `list-custom` | — |

Inputs: `image` (File), `min_confidence` (Float, default 0.4). Outputs mirror
`ObjectDetectionYOLOv5-6.2` exactly: `success`, `message`, `error`, `count`, `inferenceMs`,
`processMs`, and `predictions[]` each `{ label, confidence, x_min, y_min, x_max, y_max }`.
Response shapes stay byte-compatible so existing Blue Iris configs work unchanged.

## 6. Default model, tiering & CCTV tuning

The default detector is chosen by a setting `DEFAULT_MODEL` (values: a CCTV family id, e.g.
`ipcam-combined`, or `yolo26`). **Default value: `ipcam-combined`** (CCTV-domain, best
real-world results). YOLO26 is a one-setting switch for users who want the modern generic model.

**Hardware auto-tiering** (`options.py`, from `torch.cuda.is_available()` +
`get_device_properties(0).total_memory`, with `MODEL_TIER = auto|accurate|balanced|fast`
override) selects the *size/variant within the chosen family*:

| Tier | Selected when | CCTV family (`ipcam-combined`) | Generic family (`yolo26`) |
|---|---|---|---|
| **accurate** | GPU ≥ ~8 GB | largest available IPcam variant | YOLO26-X / -L |
| **balanced** | GPU < ~8 GB | mid IPcam variant | YOLO26-M / -S |
| **fast** | CPU-only | lightest IPcam variant, **else fall back to YOLO26-N** | YOLO26-N |

The IPcam models lack a clean n/s/m/l/x ladder (they are MikeLud-trained at a few sizes), so the
CCTV family tiers to whatever variants exist; where a light CCTV variant is missing for the CPU
tier, the module falls back to **YOLO26-N** (fast, generic) so CPU-only installs stay
responsive. (Blue Iris snapshot detection is not hard-real-time, so a larger model on CPU is
tolerable but the nano fallback keeps it snappy.)

**CCTV tuning defaults:** optionally class-filter results to CCTV-relevant classes
(person / vehicle group / animal group) and use a sensible default `min_confidence`, to reduce
false alarms. Class filtering is a configurable setting (default: no filter, i.e. return all —
Blue Iris does its own class selection; the setting lets users opt in).

## 7. Custom-model routing

On `/v1/vision/custom/<model>`:
1. Resolve `<model>.pt` in the custom-models dir (clear error if missing).
2. Load with **Ultralytics** first; on the legacy-pickle failure (`models.yolo` /
   `weights_only`), fall back to the **yolov5** runtime.
3. Cache the loaded model **and which engine** by name (refreshed like the existing module).
4. `list-custom` enumerates `*.pt` in the custom-models dir.

`install.sh`/`install.bat` ship the **default set of legacy IPcam custom models**
(ipcam-combined, ipcam-general, ipcam-animal, ipcam-dark, license-plate, delivery) from the
CodeProject CDN, so existing Blue Iris configs work out of the box.

## 8. Data flow

```
default detect (DEFAULT_MODEL = ipcam-combined):
  image → yolov5 CCTV model (tier variant, conf=min_confidence)
        → predictions [{label, confidence, x_min, y_min, x_max, y_max}] → JSON

default detect (DEFAULT_MODEL = yolo26):
  image → Ultralytics YOLO26 (tier size) → predictions → JSON (same shape)

custom detect:
  image → resolve <model>.pt → engine(Ultralytics | yolov5) → predictions → JSON (same shape)
```

## 9. Runtime, venv & install (cross-platform)

### 9.1 GPU install (Linux + Windows)
- `requirements.linux.cuda12.txt` / `requirements.windows.cuda.txt`:
  `torch==2.13.0+cu130`, `torchvision==0.28.0+cu130` (index cu130), `ultralytics`,
  `yolov5==6.2.3`, opencv, numpy, SDK. Shares the python3.11 venv.
- `post_install.sh` / `post_install.bat`:
  1. `pip install "setuptools<81"` (restore `pkg_resources` for yolov5).
  2. `pip install --upgrade "nvidia-cudnn-cu13==9.24.0.43"` (torch 2.13's bundled cuDNN 9.20 is
     broken on RTX 50-series / sm_120 — convolutions fail without this).
- **Driver floor:** cu130 needs a recent (CUDA-13-capable) NVIDIA driver. Supports Ampere
  (3070, sm_86) through Blackwell (5090, sm_120).

### 9.2 CPU install
- `requirements.txt` / `requirements.windows.txt`: `torch==2.13.0+cpu`,
  `torchvision==0.28.0+cpu` (index cpu), `ultralytics`, `yolov5==6.2.3`, opencv, numpy, SDK.
  Uses the **fast** tier. PyTorch inference (ONNX/OpenVINO acceleration deferred).

### 9.3 Weight download
- `install.sh`/`install.bat` pre-download: the **default CCTV models** (IPcam set, incl. the
  configured default `ipcam-combined` + its tier variants), the **YOLO26** weights for the
  generic family (via Ultralytics), and the default legacy custom-model set. `.gitignore` keeps
  weights out of git.

### 9.4 Module scaffolding (follow ImprovedFaceProcessing / ObjectDetectionYOLOv5-6.2 patterns)
- `modulesettings.json` (+ `modulesettings.windows.json`): LaunchSettings (Runtime
  `python3.11`, Queue `objectdetection_queue`), the three RouteMaps, EnvironmentVariables
  (`DEFAULT_MODEL`, `MODEL_TIER`, `CLASS_FILTER`, MODELS_DIR, CUSTOM_MODELS_DIR),
  ModelRequirements, ModuleReleases (1.0.0).
- Python: an SDK `ModuleRunner` adapter (`detect_adapter.py`) + a detector wrapper per engine +
  `options.py` (device/VRAM tiering + default-model + class-filter config). Mirror the existing
  object-detection module where it fits.
- `install.sh` + `install.bat`, `post_install.sh` + `post_install.bat`, `requirements*.txt`,
  `explore.html`, `test/` sample image, `.gitignore`.

## 10. Testing

- **Self-test:** `detect` (default `ipcam-combined`) on a bundled CCTV-style image; assert
  success, objects found, device/tier.
- **Dual-engine custom tests:** load one **legacy** IPcam model (yolov5 path) and one **modern**
  `.pt` (Ultralytics path); both return valid predictions.
- **Default-switch test:** `DEFAULT_MODEL=yolo26` runs the generic path and returns predictions.
- **Tier test:** tier selection by (has_cuda, vram, override), incl. the CPU YOLO26-N fallback.
- **Blue Iris shape test:** response JSON keys match `ObjectDetectionYOLOv5-6.2` exactly.
- **A/B + latency:** compare CCTV-default vs YOLO26 vs the old module on `TestData/Objects/*`;
  record measured RTX 5090 and CPU latency per tier.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Generic COCO model → more CCTV false alarms | CCTV-domain model is the **default**; generic is opt-in |
| Legacy custom `.pt` won't load in Ultralytics | yolov5 engine runs default CCTV + legacy custom |
| IPcam models lack a clean size ladder | Tier to available variants; CPU falls back to YOLO26-N |
| AGPL (YOLO26) / IPcam model licenses | Documented per weight; generic path is opt-in |
| CPU too slow | Fast tier + YOLO26-N fallback; ONNX/OpenVINO noted future |
| YOLO26 churn | Generic family is a one-line config swap → YOLOv12 / YOLO11 |
| Driver too old for cu130 | Documented; CPU path always works |
| Windows untested from Linux dev box | Reuse verified cross-platform patterns; validate on the box |
| Blue Iris response drift | Response-shape test pinned to the existing module |

## 12. Open items to resolve during implementation

- Which IPcam model to ship as the default and its exact CDN asset name(s); what size variants
  MikeLud provides for CCTV-family tiering (and confirm the CPU→YOLO26-N fallback threshold).
- Exact YOLO26 size per tier (X vs L; M vs S) — confirm VRAM peak.
- Whether Ultralytics reliably auto-downloads YOLO26 weights offline vs needing an explicit URL.
- Confirm `yolov5` + `ultralytics` coexist cleanly in one venv on torch 2.13 (Task 1 spike).
- Whether legacy IPcam models run correctly under yolov5==6.2.3 + torch 2.13 (they should, per
  the `ObjectDetectionYOLOv5-6.2` fixes) — verify in the dual-engine custom test.
