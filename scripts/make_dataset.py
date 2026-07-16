"""
Auto-label real Encar photos to train a watermark+plate detector (YOLO format).

The heuristic detector in rebrand_core is accurate WHEN CONFIDENT (colour path on
light backgrounds; high-score template match on dark). We run it over many real
photos, keep only what it confidently finds, and emit YOLO labels. A trained YOLO
model then generalises to the low-confidence cases the heuristic misses — all
in-domain, no manual labelling, no need for watermark-free backgrounds (which don't
exist: Encar bakes the mark into the source at every resolution).

Classes: 0 = watermark ("Trust Encar"), 1 = plate (red dealer plate).

Usage (run where rebrand_core + its deps are importable, e.g. inside the tool image):
    python make_dataset.py urls.txt out_dir [--val 0.1] [--max 4000]
    # urls.txt: one image URL per line
Produces: out_dir/{images,labels}/{train,val} + out_dir/data.yaml
"""
import os
import sys
import random
import urllib.request

import cv2
import numpy as np

import rebrand_core as core


def fetch(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "dataset/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = r.read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def to_yolo(x0, y0, x1, y1, W, H, cls):
    cx, cy = ((x0 + x1) / 2) / W, ((y0 + y1) / 2) / H
    w, h = (x1 - x0) / W, (y1 - y0) / H
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    urls_file, out = sys.argv[1], sys.argv[2]
    val_frac = float(sys.argv[sys.argv.index("--val") + 1]) if "--val" in sys.argv else 0.1
    maxn = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else None
    # v1: watermark-only → cleaner set. Plate labels are noisy (tail-lights), so skip
    # them; images with no confident watermark are excluded (no unlabeled positives).
    wm_only = "--watermark-only" in sys.argv

    urls = [l.strip() for l in open(urls_file) if l.strip()]
    random.seed(0)
    random.shuffle(urls)
    if maxn:
        urls = urls[:maxn]

    for split in ("train", "val"):
        os.makedirs(f"{out}/images/{split}", exist_ok=True)
        os.makedirs(f"{out}/labels/{split}", exist_ok=True)

    seen, labeled, with_wm, with_plate = 0, 0, 0, 0
    for i, u in enumerate(urls):
        try:
            img = fetch(u)
        except Exception:
            continue
        if img is None:
            continue
        seen += 1
        H, W = img.shape[:2]
        red = core._red_mask(img)
        wm = core.detect_watermark_box(img, red)      # (x0,y0,x1,y1) or None
        pl = None if wm_only else core.detect_plate_box(img, red)  # (x,y,w,h) or None

        labels = []
        if wm:
            labels.append(to_yolo(wm[0], wm[1], wm[2], wm[3], W, H, 0))
            with_wm += 1
        if pl:
            labels.append(to_yolo(pl[0], pl[1], pl[0] + pl[2], pl[1] + pl[3], W, H, 1))
            with_plate += 1
        if not labels:
            continue

        split = "val" if random.random() < val_frac else "train"
        name = f"img{i:06d}"
        cv2.imwrite(f"{out}/images/{split}/{name}.jpg", img)
        with open(f"{out}/labels/{split}/{name}.txt", "w") as f:
            f.write("\n".join(labels) + "\n")
        labeled += 1
        if labeled % 100 == 0:
            print(f"  labeled {labeled} (seen {seen})")

    names = "  0: watermark\n" if wm_only else "  0: watermark\n  1: plate\n"
    with open(f"{out}/data.yaml", "w") as f:
        f.write(
            f"path: {os.path.abspath(out)}\n"
            "train: images/train\nval: images/val\n"
            "names:\n" + names
        )
    print(f"DONE: {labeled} labeled / {seen} fetched  (watermark={with_wm}, plate={with_plate})")


if __name__ == "__main__":
    main()
