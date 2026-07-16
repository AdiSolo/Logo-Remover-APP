#!/usr/bin/env python3
"""
Automatic logo/watermark removal pipeline.

Pipeline:
  1. DETECT   - multi-scale template matching to locate the logo in each image
  2. MASK     - build a (dilated) binary mask around the detected region
  3. INPAINT  - remove the logo using either:
                  - OpenCV inpainting (fast, no extra deps, good for simple backgrounds)
                  - LaMa via IOPaint  (slower, much better quality on complex backgrounds)

Usage
-----
1) One-time: crop a small PNG/JPG of ONLY the logo from one of your images
   (as tight as possible) and save it, e.g. logo_template.png

2) Run:
   python remove_logo.py \
       --input ./images_in \
       --output ./images_out \
       --template ./logo_template.png \
       --method opencv        # or: lama

Options of note:
   --threshold   match confidence required to accept a detection (0-1, default 0.75)
   --pad         pixels to grow the mask outward from the detected box (default 6)
   --min-scale / --max-scale  scale range to search when the logo size varies
   --dry-run     only draw detected boxes on a copy, don't inpaint (useful for tuning)
"""

import argparse
import os
import sys
import glob

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Detection: multi-scale template matching
# --------------------------------------------------------------------------
def detect_logo(image_gray, template_gray, threshold=0.75, min_scale=0.5, max_scale=1.5, steps=15):
    """
    Slide the template over the image at multiple scales.
    Returns (x, y, w, h, score) of the best match, or None if nothing clears the threshold.
    """
    best = None
    th, tw = template_gray.shape[:2]

    for scale in np.linspace(min_scale, max_scale, steps):
        rt = cv2.resize(template_gray, (max(1, int(tw * scale)), max(1, int(th * scale))))
        rt_h, rt_w = rt.shape[:2]

        if rt_h >= image_gray.shape[0] or rt_w >= image_gray.shape[1]:
            continue

        res = cv2.matchTemplate(image_gray, rt, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold and (best is None or max_val > best[4]):
            best = (max_loc[0], max_loc[1], rt_w, rt_h, max_val)

    return best


def build_mask(image_shape, box, pad=6):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    x, y, w, h, _ = box
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(image_shape[1], x + w + pad), min(image_shape[0], y + h + pad)
    mask[y0:y1, x0:x1] = 255
    return mask


# --------------------------------------------------------------------------
# Inpainting backends
# --------------------------------------------------------------------------
def inpaint_opencv(image_bgr, mask):
    return cv2.inpaint(image_bgr, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)


def inpaint_lama(image_bgr, mask):
    """
    Requires: pip install iopaint
    Runs LaMa locally (CPU or GPU) for much cleaner results on complex backgrounds.
    """
    try:
        from iopaint.model_manager import ModelManager
        from iopaint.schema import InpaintRequest
    except ImportError:
        print("ERROR: iopaint not installed. Run: pip install iopaint --break-system-packages", file=sys.stderr)
        sys.exit(1)

    global _LAMA_MODEL
    if "_LAMA_MODEL" not in globals():
        _LAMA_MODEL = ModelManager(name="lama", device="cpu")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = _LAMA_MODEL(image_rgb, mask, InpaintRequest())
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------
# Main batch pipeline
# --------------------------------------------------------------------------
def process_folder(input_dir, output_dir, template_path, method="opencv",
                    threshold=0.75, pad=6, min_scale=0.5, max_scale=1.5, dry_run=False):
    os.makedirs(output_dir, exist_ok=True)

    template = cv2.imread(template_path)
    if template is None:
        print(f"ERROR: could not read template image at {template_path}", file=sys.stderr)
        sys.exit(1)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(input_dir, e)))
        files.extend(glob.glob(os.path.join(input_dir, e.upper())))

    if not files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(files)} images. Method={method}, threshold={threshold}")

    n_detected, n_skipped = 0, 0

    for path in sorted(files):
        img = cv2.imread(path)
        if img is None:
            print(f"  [skip] could not read {path}")
            n_skipped += 1
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        box = detect_logo(img_gray, template_gray, threshold, min_scale, max_scale)

        fname = os.path.basename(path)
        out_path = os.path.join(output_dir, fname)

        if box is None:
            print(f"  [no match] {fname} — copying unchanged")
            cv2.imwrite(out_path, img)
            n_skipped += 1
            continue

        x, y, w, h, score = box
        n_detected += 1

        if dry_run:
            preview = img.copy()
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imwrite(out_path, preview)
            print(f"  [dry-run] {fname} — match {score:.2f} at ({x},{y},{w},{h})")
            continue

        mask = build_mask(img.shape, box, pad=pad)

        if method == "lama":
            result = inpaint_lama(img, mask)
        else:
            result = inpaint_opencv(img, mask)

        cv2.imwrite(out_path, result)
        print(f"  [removed] {fname} — match {score:.2f} at ({x},{y},{w},{h})")

    print(f"\nDone. {n_detected} logos removed, {n_skipped} skipped/unchanged.")
    print(f"Output written to: {output_dir}")


def main():
    p = argparse.ArgumentParser(description="Automatic logo/watermark removal")
    p.add_argument("--input", required=True, help="Folder of input images")
    p.add_argument("--output", required=True, help="Folder to write cleaned images to")
    p.add_argument("--template", required=True, help="Cropped image of ONLY the logo")
    p.add_argument("--method", choices=["opencv", "lama"], default="opencv")
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--pad", type=int, default=6)
    p.add_argument("--min-scale", type=float, default=0.5)
    p.add_argument("--max-scale", type=float, default=1.5)
    p.add_argument("--dry-run", action="store_true", help="Draw detected boxes instead of removing")
    args = p.parse_args()

    process_folder(
        input_dir=args.input,
        output_dir=args.output,
        template_path=args.template,
        method=args.method,
        threshold=args.threshold,
        pad=args.pad,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
