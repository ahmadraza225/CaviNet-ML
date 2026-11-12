# Symptom-assisted prediction parameters (M-12)

One `<model version>-fusion.json` per released model, written by `cavinet-ml fusion`
(`work/fusion/cavinet_fusion.json`). Each file names the model weights it was fitted for
(`bundle_weights_sha256`); `cavinet-ml fetch-model` copies the matching file next to the
model bundle as `cavinet_fusion.json`. The files hold only regression coefficients and
training averages, no patient data.
