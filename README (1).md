# Logo/Watermark Remover

Automatically detects and removes a recurring logo or watermark from a folder of images using template matching + inpainting.

## How it works

1. **Detect** — multi-scale template matching locates the logo in each image, even if its size varies slightly between images.
2. **Mask** — a padded mask is built around the detected region.
3. **Inpaint** — the masked area is filled in using either:
   - `opencv` — fast, no extra dependencies, works well on simple/uniform backgrounds
   - `lama` — slower, much cleaner results on complex or detailed backgrounds (via [IOPaint](https://github.com/Sanster/IOPaint))

## Requirements

```bash
pip install opencv-python numpy --break-system-packages
```

Optional, only needed for the higher-quality `lama` method:

```bash
pip install iopaint --break-system-packages
```

## Setup

Crop a small, tight image containing **only the logo** from one of your source images (no surrounding background if possible) and save it, e.g. `logo_template.png`. This is the reference the script searches for in every image.

## Usage

### 1. Dry run (recommended first step)

Draws a red box around each detected logo instead of removing it, so you can confirm detection is accurate before processing anything for real:

```bash
python remove_logo.py \
  --input ./images_in \
  --output ./preview \
  --template ./logo_template.png \
  --dry-run
```

Check the output images in `./preview`. If boxes are missing or misplaced, adjust `--threshold` (see below) and re-run.

### 2. Remove the logo

```bash
python remove_logo.py \
  --input ./images_in \
  --output ./images_out \
  --template ./logo_template.png \
  --method opencv
```

Use `--method lama` instead of `opencv` if results look smeared or blurry on busy backgrounds:

```bash
python remove_logo.py \
  --input ./images_in \
  --output ./images_out \
  --template ./logo_template.png \
  --method lama
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Folder of source images |
| `--output` | *(required)* | Folder to write cleaned images to |
| `--template` | *(required)* | Path to the cropped logo image |
| `--method` | `opencv` | Inpainting backend: `opencv` or `lama` |
| `--threshold` | `0.75` | Match confidence required to accept a detection (0–1). Lower = more permissive (catches more, but riskier false positives). Raise if it's matching the wrong region. |
| `--pad` | `6` | Pixels to grow the mask outward from the detected logo box. Higher values give cleaner edges but remove more surrounding detail. |
| `--min-scale` / `--max-scale` | `0.5` / `1.5` | Range of sizes to search for the logo, relative to the template. Widen if your logo varies a lot in size across images. |
| `--dry-run` | off | Preview detected boxes without removing anything |

## Supported image formats

`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`

## Notes & limitations

- Works best when the logo has a **consistent design** even if its **position or size varies** (multi-scale search handles size differences automatically).
- If the logo appears in a **completely different location** in every image with no consistent visual pattern to match against, template matching won't find it reliably — that would require a trained object detector instead.
- Images where no match clears the threshold are copied to the output folder unchanged (not skipped entirely), so your output folder always has the full set.
- Check that you have the rights to remove watermarks/logos from images before redistributing them — many are there to mark copyright or source attribution.

## Troubleshooting

**Logo not being detected at all**
Lower `--threshold` (try `0.6`) and re-run with `--dry-run` to see if it starts catching matches.

**False positives (wrong area being removed)**
Raise `--threshold` (try `0.85`+), or crop a more distinctive/less generic section of the logo for your template.

**Inpainted area looks smeared or blurry**
Switch to `--method lama` for AI-based inpainting instead of OpenCV's classic algorithm.

**`iopaint` import error**
Run `pip install iopaint --break-system-packages` — it's an optional dependency only needed for the `lama` method.
