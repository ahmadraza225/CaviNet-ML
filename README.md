# cavinet_ml

The CaviNet AI package. The same code is used by the training toolkit (on the GPU
computer) and by the application worker, so preprocessing is identical in both.

| Module | Scope section | What it does |
|---|---|---|
| `io.dicom` | 11.1 step 1 | Read a DICOM series with SimpleITK/GDCM (JPEG-Lossless included), sorted by position, in HU, LPS orientation |
| `preprocessing` | 11.1 steps 2–7 | Clip to −1024 HU; lungmask R231 lung mask (body-outline fallback with a warning); resample to 1.5 mm; crop to lungs + 10 mm; lung window [−1350, 150] → [0, 1]; resize to 128³ float16 |
| `model.network` | 11.2 | MONAI 3D ResNet-18, 1 channel, 1 logit, dropout 0.3 |
| `model.bundle` | 11.6 | Save/load `cavinet_model.pth` with `torch.load(weights_only=True)` |
| `inference` | 11.5 | Ensemble logits → temperature → TB probability → class, confidence, band, explanation |
| `previews` | FR-05.3 | 48 lung-window preview PNGs and 3 representative slices |
| `analysis` | M-05 | What the application worker calls for one scan |
| `demo` | FR-05.6 | Trains the demo model on synthetic volumes (`is_demo = true`) |
| `fetch` | 11.6 | `make fetch-model`: download `MODEL_URL`, else build the demo model |

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU PyTorch first
pip install -e "ml[dev]"
cavinet-ml --version
cavinet-ml fetch-model --out models/cavinet_model.pth   # or: cavinet-ml build-demo
```

The lungmask weights (`unet_r231-d5d2fc3d.pth`) are baked into the Docker image. Outside
Docker, download them from the lungmask release and set `LUNGMASK_WEIGHTS` to run the real
segmenter test; without them preprocessing uses the documented body-outline fallback.
