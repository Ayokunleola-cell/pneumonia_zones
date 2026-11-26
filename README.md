# Pneumonia Zone Localization and Severity Grading (frontal chest X-ray)

Implements the Meeting Report / Proposed Algorithm for paediatric SCA pneumonia:
six radiographic zones (RUZ, RMZ, RLZ, LUZ, LMZ, LLZ), per-lung percentage involvement
(each lung is its own 100%), and per-lung severity grade.

| Grade | % of that lung | Label |
|---|---|---|
| 0 | 0 | No radiographic involvement |
| 1 | >0-25 | Mild |
| 2 | >25-50 | Moderate |
| 3 | >50-75 | Severe |
| 4 | >75-100 | Very severe |

Thresholds are a proposed research scale and need radiologist validation.

## Files
- `pneumonia_zones.py`      zones, percentages, grading, overlays, batch labelling CLI
- `train_opacity.py`        U-Net training for opacity segmentation + inference helper
- `make_synthetic_data.py`  synthetic data for smoke-testing only (not medical data)
- `test_pneumonia_zones.py` unit tests (`pytest`)
- `demo_overlay.png`        example output

## Install
    pip install -r requirements.txt

## Data layout
    images/    <stem>.png          chest X-rays
    masks/     <stem>.png|.json    opacity annotation (missing file = normal image)
    lungs/     <stem>.png          combined lung masks (both lungs)
    patients.csv                   image,patient_id  (train/val split is by patient)

## Train
    python train_opacity.py --images images --masks masks --patients patients.csv \
        --lungs lungs --size 512 --epochs 60 --out opacity_unet.pt

## Label new X-rays
    python pneumonia_zones.py
Defaults to `../BDCXR-3257` (scans nested subfolders recursively) and writes `labels.csv`, `labels.json`, and `<image>_overlay.png` per image under `results/`.
With pretrained TorchXRayVision lung segmentation and experimental RSNA Grad-CAM opacity estimation:
    python pneumonia_zones.py --lung-backend xrv --opacity-backend xrv-rsna --out results/
You can also pass a custom folder:
    python pneumonia_zones.py new_xrays/ --lung-dir lungs/ --opacity-model opacity_unet.pt --out results/
Use `--opacity-dir` to use radiologist annotations instead of the model, `--lung-backend xrv`
for pretrained lung segmentation (needs torch + torchxrayvision), `--opacity-backend xrv-rsna`
for experimental pretrained RSNA Grad-CAM opacity estimation (not calibrated for clinical grading),
`--flip-lr` for mirrored images.

## Conventions and caveats
- Patient's right lung is on the image's left (standard display).
- Zones are thirds of each lung's own vertical extent, clipped to the lung mask; they describe
  projected height, not anatomical lobes or volume.
- Percentages are 2D projected area, not lung volume.
- Open items for the radiologist: "indeterminate" cutoff (`present_min_frac`), minimum overlap
  for a zone to count as affected (`zone_min_overlap_frac`), and the grading thresholds.
- Only tested on synthetic data; validate on real annotated X-rays before any clinical interpretation.
