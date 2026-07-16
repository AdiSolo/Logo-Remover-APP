# Training the watermark detector (Colab) + deploying it

Goal: a small YOLO model that reliably finds the "Trust Encar" watermark across all
Encar backgrounds (light/dark), replacing the brittle colour/template heuristic.
Inference runs on CPU on the VPS; only **training** needs a GPU (Colab).

## 1. Get the dataset
The dataset is auto-labelled from real Encar photos by `scripts/make_dataset.py`
(watermark-only, class 0). It lives on the VPS at `/tmp/dataset` after the build.
Package + download it:
```bash
# on the VPS
docker exec rebrand-rebrand-1 sh -c 'cd /tmp && tar czf dataset.tgz dataset'
docker cp rebrand-rebrand-1:/tmp/dataset.tgz /root/dataset.tgz
# then from your machine:
scp root@85.31.236.157:/root/dataset.tgz .
```
Layout: `dataset/images/{train,val}`, `dataset/labels/{train,val}`, `dataset/data.yaml`.

## 2. Train in Colab (Runtime → GPU)
```python
!pip -q install ultralytics
from google.colab import files
files.upload()                      # upload dataset.tgz
!tar xzf dataset.tgz -C /content
# point data.yaml at the Colab path
import re, pathlib
y = pathlib.Path('/content/dataset/data.yaml'); t = y.read_text()
y.write_text(re.sub(r'^path:.*', 'path: /content/dataset', t, flags=re.M))

from ultralytics import YOLO
model = YOLO('yolo11n.pt')          # nano: fast CPU inference
model.train(data='/content/dataset/data.yaml',
            epochs=80, imgsz=1024, batch=16, patience=20,
            degrees=0, shear=0, perspective=0,   # watermark isn't rotated
            hsv_v=0.4, scale=0.3, translate=0.1) # vary lighting/position a bit
# quick sanity check
model.val()
files.download('runs/detect/train/weights/best.pt')
```
Tips: watermark is smallish → `imgsz=1024` (or 1280 if VRAM allows) matters for recall.
If val mAP looks low, add more data (raise `--max` in make_dataset) and re-train.

## 3. Deploy the weights
Drop the trained model into the repo and redeploy — the tool auto-uses it if present:
```bash
cp best.pt assets/wm_detector.pt
git add assets/wm_detector.pt && git commit -m "Add trained watermark detector" && git push
# on the VPS:
cd /docker/rebrand && docker compose build && docker compose up -d
```
`rebrand_core.detect_watermark_box` prefers the ML model when `assets/wm_detector.pt`
exists and falls back to the heuristic otherwise — so nothing breaks if it's absent.

## 4. Validate before backfilling
Re-run the spot-check (light + dark + colored cars) via the API and eyeball results.
Only then consider the catalog backfill (main-photo-first).

## Iterating
- More/cleaner data is the main lever. Bump `--max`, and optionally review labels in
  Roboflow/Label Studio (import YOLO format) to prune the occasional bad box.
- Add the **plate** class later: relabel front-only shots (rear tail-lights are false
  positives) and retrain with `names: {0: watermark, 1: plate}`.
