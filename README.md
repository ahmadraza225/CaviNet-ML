# cavinet_ml

The CaviNet AI package. The same code is used by the training toolkit (on the GPU
computer) and by the application worker, so preprocessing is identical in both. The training
toolkit's step-by-step guide is [`docs/TRAINING_RUNBOOK.md`](../docs/TRAINING_RUNBOOK.md).

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
| `dataset.source` | 7.1 | Read the Kaggle dataset from its zip or folder, one patient at a time |
| `dataset.patient_index` | 7.1, 7.3 | Read PatientIndex.xlsx (legend rows ignored, blank = absent, duplicates reported) |
| `dataset.index` | FR-10.1 | `cavinet-ml index`: manifest.csv from the DICOM headers and the index |
| `dataset.cache` | FR-10.2 | `cavinet-ml preprocess`: section 11.1 into a float16 cache, QC report, resumable, parallel |
| `dataset.split` | FR-10.3, 11.3 | `cavinet-ml split`: locked 20% test set and 5 stratified folds (seed 42) |
| `dataset.synthetic` | — | Synthetic look-alike of the Kaggle dataset for tests and `make rehearsal` |
| `training.config` | 11.4 | The training configuration (`configs/train.yaml`), strictly validated |
| `training.augment` | 11.4 | Flip, rotation, scaling, translation, intensity and noise, on the GPU |
| `training.pretrained` | 11.2 | MedicalNet ResNet-18 start, or training from scratch (recorded) |
| `training.trainer` | FR-10.4 | `cavinet-ml train --fold k`: AMP, accumulation, early stopping, resume, logs, OOM handling |
| `training.calibrate` | FR-10.5, 11.5 | `cavinet-ml calibrate`: temperature on the out-of-fold logits |
| `training.export` | FR-10.6, 11.6 | `cavinet-ml export`: the bundle, model_card.json and docs/MODEL_CARD.md |
| `evaluation.metrics`, `.stats` | 11.7 | Metrics, bootstrap 95% CIs, calibration error, DeLong test |
| `evaluation.baselines` | FR-10.8 | `cavinet-ml baseline` / `shortcut-check`: clinical-only and metadata-only models |
| `evaluation.evaluate` | FR-10.7, FR-10.8 | `cavinet-ml evaluate` (test split once, every run logged) and `compare` |

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU PyTorch first
pip install -e "ml[dev,train]"       # train: the training toolkit's extra libraries
cavinet-ml --version
cavinet-ml fetch-model --out models/cavinet_model.pth   # or: cavinet-ml build-demo
cavinet-ml --help                    # every toolkit command
```

The application's Docker image installs the package without the `train` extra.

The lungmask weights (`unet_r231-d5d2fc3d.pth`) are baked into the Docker image. Outside
Docker, download them from the lungmask release and set `LUNGMASK_WEIGHTS` to run the real
segmenter test; without them preprocessing uses the documented body-outline fallback.
