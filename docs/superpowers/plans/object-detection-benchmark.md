# ObjectDetection Module — A/B + Latency Benchmark

| Field          | Value                                      |
|----------------|--------------------------------------------|
| Module         | ObjectDetection v1.0.0                    |
| Date           | 2026-07-19                                |
| Hardware       | NVIDIA GeForce RTX 5090 (34 GB VRAM)      |
| Default model  | ipcam-combined (YOLOv5 engine, GPU tier)  |
| PyTorch        | 2.13.0+cu130 / CUDA 13.0                  |
| Server version | 2.9.6                                     |

---

## Per-image Object Counts

Both `/v1/vision/detection` (default ipcam-combined via detect route) and
`/v1/vision/custom/ipcam-combined` (explicit custom route) are compared.
Counts are identical — same model, same engine, both routes routed to `ObjectDetection`.

| Image                 | Default detect count | Default inferenceMs | Custom count | Custom inferenceMs |
|-----------------------|---------------------:|--------------------:|-------------:|-------------------:|
| cat-on-wall.jpg       |                    1 |                 115 |            1 |                128 |
| intersection.jpg      |                    9 |                  17 |            9 |                 17 |
| is-it-a-dog.jpg       |                    1 |                  18 |            1 |                 21 |
| kitchen.jpg           |                    0 |                   9 |            0 |                  7 |
| living-room.jpg       |                    0 |                   8 |            0 |                 11 |
| man-at-desk.jpg       |                    1 |                   8 |            1 |                  9 |
| menagerie.jpg         |                   10 |                   8 |           10 |                  8 |
| office-presentation.jpg |                4 |                   8 |            4 |                  9 |
| parrot.jpg            |                    1 |                  18 |            1 |                 21 |
| quail.JPG             |                    1 |                  18 |            1 |                 17 |
| street-at-night.jpg   |                    4 |                  10 |            4 |                  9 |
| study-group.jpg       |                    8 |                   8 |            8 |                  9 |
| traffic.jpeg          |                   24 |                  22 |           24 |                 13 |
| traffic.jpg           |                   22 |                   8 |           22 |                  8 |
| traffic2.jpg          |                   20 |                   8 |           20 |                  8 |

Notes:
- `cat-on-wall.jpg` shows higher first-call latency (115/128 ms) due to model warm-up on first GPU use.
- Subsequent calls drop to 7–22 ms, consistent with model already resident on GPU.
- `kitchen.jpg` and `living-room.jpg` return 0 objects (no vehicles/persons detected at 0.4 confidence — expected for indoor scenes without supported classes).

---

## GPU Latency (50 calls on traffic.jpg, /v1/vision/detection)

| Metric         | inferenceMs |
|----------------|------------:|
| Median         |         8.0 |
| Mean           |         8.0 |
| Min            |           7 |
| Max            |          14 |

All 50 calls reporting `inferenceDevice: GPU`. The warm model inference is extremely fast on the RTX 5090 — 8 ms median for a 640px YOLOv5s pass.

---

## CPU Tier (MODEL_TIER=fast)

Not measured in this run. The `fast` tier falls back to YOLO26-N (ultralytics nano) when no
CCTV-fast variant exists. Measuring it requires restarting the module with `MODEL_TIER=fast`
env override, which was not done here to avoid service interruption. A separate benchmark run
can capture this; expected range: 150–400 ms on CPU for YOLO26-N.

---

## Route Ownership Confirmation

| Route                          | moduleId        | inferenceDevice |
|--------------------------------|-----------------|-----------------|
| POST /v1/vision/detection      | ObjectDetection | GPU             |
| POST /v1/vision/custom/ipcam-combined | ObjectDetection | GPU      |
| POST /v1/vision/custom/list    | ObjectDetection | GPU             |

`ObjectDetectionYOLOv5-6.2` status: `NoAutoStart` (disabled via `AutoStart: false`).
