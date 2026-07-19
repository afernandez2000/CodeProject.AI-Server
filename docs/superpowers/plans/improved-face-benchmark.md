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

## Notes

- CPU-tier numbers not measured (GPU only environment).
- FaceProcessing (predecessor) set to AutoStart=false; ImprovedFaceProcessing is now the sole handler of `/v1/vision/face*` routes.
- Route coexistence resolved by adding an AutoStart guard in `ModuleProcessServices.SetupQueueAndRoutes` — modules with `AutoStart=false` no longer register routes, preventing them from overwriting active module routes.
- Smoke test confirmed: `moduleId=ImprovedFaceProcessing`, `inferenceDevice=GPU`, `success=true`.
