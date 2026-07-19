# ImprovedFaceProcessing A/B Benchmark

| Field | Value |
|---|---|
| Module | ImprovedFaceProcessing |
| Version | 1.0.0 |
| Date | 2026-07-19 |
| Hardware | NVIDIA GeForce RTX 5090 |
| CUDA | 12.0 |
| Inference tier | GPU (CUDAExecutionProvider via onnxruntime-gpu) |
| Detection model | SCRFD (ONNX) |
| Recognition model | AdaFace IR (PyTorch) |
| Timing image | family-on-couch.jpg |
| Timing iterations | 50 sequential calls |

## GPU Latency (sequential, single image)

| Metric | Value |
|---|---|
| Median roundtrip ms | 11.1 |
| Mean roundtrip ms | 11.2 |
| Min roundtrip ms | 10.3 |
| Max roundtrip ms | 12.4 |
| Median inferenceMs | 5.0 |
| Mean inferenceMs | 4.7 |
| StdDev inferenceMs | 0.6 |

## Per-image Detection Results

| Image | Face Count | inferenceMs | Roundtrip ms |
|---|---|---|---|
| Chris-Hemsworth-2.jpg | 1 | 62 | 73.2 |
| Robert-Downey-Jr-2.jpg | 2 | 14 | 26.1 |
| Robert-Downey-Jr-3.jpg | 1 | 15 | 23.1 |
| chris-hemsworth-1.jpg | 1 | 13 | 19.5 |
| chris-hemsworth-3.jpg | 1 | 13 | 21.5 |
| crowd.jpg | 219 | 17 | 28.4 |
| family-on-couch.jpg | 4 | 15 | 23.2 |
| kate-winslet-1.jpg | 1 | 10 | 14.4 |
| robert-downey-jr-1.jpg | 1 | 11 | 15.9 |
| sailors-all-hands-navy-military.jpg | 224 | 11 | 19.6 |
| scarlett-johanson-1.jpg | 1 | 10 | 16.8 |
| scarlett-johanson-2.jpg | 1 | 5 | 11.9 |
| two-people-depth-of-field.jpg | 2 | 5 | 13.1 |
| two-people-selfie.jpg | 2 | 20 | 27.4 |
| woman-low-lighting.jpg | 1 | 4 | 13.2 |
| woman-rainy-window.jpg | 1 | 5 | 14.0 |
| woman-with-flowers.jpg | 0 | 5 | 11.8 |

## CPU Tier (fast tier: SCRFD-2.5G + AdaFace IR-50)

Measured 2026-07-19 on the same box with the GPU hidden (`CUDA_VISIBLE_DEVICES=""`),
which drives the module's auto-tiering to select the **fast** tier and run on CPU
(`CPUExecutionProvider` + CPU torch). Caveat: this box has `onnxruntime-gpu`, so SCRFD
ran via its CPU execution provider; a true CPU-only install uses the plain `onnxruntime`
package (`requirements.txt`) — identical computation, different package.

| Metric | Value |
|---|---|
| Auto-selected tier | fast (SCRFD-2.5G + AdaFace IR-50) |
| Providers | `['CPUExecutionProvider']` |
| Recognition threshold | 0.64 (remapped cosine) |
| Model load time (CPU) | 0.88 s |
| Detect median (family-on-couch.jpg) | 13 ms |
| Detect min | 12 ms |
| Detect face count | 4 |
| Recognize (register + match Chris Hemsworth) | matched `chris` @ 0.88 confidence |

For contrast, the accurate-tier detector (SCRFD-10G) was ~1.3 s per image on CPU in the
design spike — the fast tier is ~100× faster on CPU, which is why the module drops to it
automatically when no capable GPU is present.

## Notes

- CPU tier verified end-to-end on hardware (fast-tier selection, CPU providers, CPU torch
  inference, detect/register/recognize) — see the CPU Tier section above.
- FaceProcessing (predecessor) set to AutoStart=false; ImprovedFaceProcessing is now the sole handler of `/v1/vision/face*` routes.
- Route coexistence resolved by **deterministic, ownership-aware route registration** (`BackendRouteMap.Register` + `ModuleProcessServices`): an enabled (`AutoStart=true`) module wins a shared route regardless of registration order, is never displaced by a disabled module, and routes are re-registered when a module is enabled at runtime (`StartProcess`). The exposed `/v1/vision/face*` routes and methods are unchanged (external-client compatible).
- Smoke test confirmed: `moduleId=ImprovedFaceProcessing`, `inferenceDevice=GPU`, `success=true`.
