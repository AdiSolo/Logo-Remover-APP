"""
Reel generation: turn an ordered list of already-hosted car photo URLs plus a
small set of pre-formatted display strings into a silent, vertical (1080x1920)
MP4 suitable for Instagram/TikTok/Facebook Reels.

Pipeline (all local, no network beyond downloading the source photos):
  1. Download each photo (skip on failure — a few bad URLs shouldn't sink the
     whole render; too few survivors raises).
  2. Pillow: composite each photo into a fixed 1080x1920 "story" frame — a
     blurred/darkened cover-crop of the same photo fills the canvas, the photo
     itself is contain-fit centered on top. This avoids cropping the car out of
     frame on landscape source photos (Encar's photos are always ~1160x696).
  3. Pillow: render ONE branded overlay PNG (bottom gradient scrim, title/spec/
     price text, logo watermark bug) — reused unchanged across the whole video,
     composited onto every frame at step 2 (not as a separate ffmpeg overlay
     pass) so the ffmpeg side only ever deals with plain, uniform JPEGs.
  4. ffmpeg: Ken Burns zoom per photo (zoompan) + crossfade between consecutive
     photos (xfade), muxed with a silent AAC track, encoded as H.264/yuv420p
     with +faststart for social-platform compatibility.

This module is pure processing (mirrors rebrand_core.py's role) — no FastAPI,
no job-queue logic; api.py owns the async job wrapper around render_reel().
"""
import os
import subprocess
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FONT_BOLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")
FONT_REGULAR = os.path.join(FONT_DIR, "DejaVuSans.ttf")

CANVAS = (1080, 1920)

MIN_PHOTOS = int(os.environ.get("VIDEO_MIN_PHOTOS", "4"))
MAX_PHOTOS = int(os.environ.get("VIDEO_MAX_PHOTOS", "8"))
CLIP_DUR = float(os.environ.get("VIDEO_CLIP_DUR", "3.0"))  # seconds each photo holds
XFADE_DUR = float(os.environ.get("VIDEO_XFADE_DUR", "0.6"))  # crossfade overlap
FPS = 30
VIDEO_ALGO_VERSION = os.environ.get("VIDEO_ALGO_VERSION", "1")  # bump on template redesign

_FETCH_TIMEOUT = 20


class ReelError(Exception):
    """Raised for any unrecoverable failure in render_reel (caught by api.py)."""


def _download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rebrand-api/1.0"})
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as r:
            dest.write_bytes(r.read())
        return True
    except Exception:
        return False


def _pick_photos(photo_urls: list[str], n: int) -> list[str]:
    """Evenly spread the selection across the full gallery (front/side/interior/
    etc.) rather than just taking the first N, which tend to be similar angles."""
    if len(photo_urls) <= n:
        return list(photo_urls)
    step = len(photo_urls) / n
    return [photo_urls[int(i * step)] for i in range(n)]


def _prepare_frame(src: Path, dest: Path) -> None:
    """Just the photo, story-composited onto the 1080x1920 canvas — NO overlay
    text here. The overlay is composited once, after the crossfade chain, in
    the ffmpeg graph (see _build_filter_complex) — baking it into each frame
    would make it double-expose/ghost during every xfade transition, since the
    same static text would be blending with itself (confirmed visually while
    testing this)."""
    img = Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")

    cw, ch = CANVAS
    bg = ImageOps.fit(img, CANVAS, method=Image.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(28))
    bg = ImageEnhance.Brightness(bg).enhance(0.55)

    iw, ih = img.size
    scale = min(cw / iw, ch / ih)
    fw, fh = int(iw * scale), int(ih * scale)
    fg = img.resize((fw, fh), Image.LANCZOS)

    canvas = bg.convert("RGB")
    canvas.paste(fg, ((cw - fw) // 2, (ch - fh) // 2))
    canvas.save(dest, quality=90)


def _render_overlay(fields: dict) -> Image.Image:
    cw, ch = CANVAS
    overlay = Image.new("RGBA", CANVAS, (0, 0, 0, 0))

    scrim_h = int(ch * 0.42)
    scrim = Image.new("L", (1, scrim_h), color=0)
    for y in range(scrim_h):
        scrim.putpixel((0, y), int(200 * (y / scrim_h) ** 1.4))
    scrim = scrim.resize((cw, scrim_h))
    black = Image.new("RGBA", (cw, scrim_h), (0, 0, 0, 255))
    black.putalpha(scrim)
    overlay.paste(black, (0, ch - scrim_h), black)

    draw = ImageDraw.Draw(overlay)
    pad = 56
    title_font = ImageFont.truetype(FONT_BOLD, 64)
    trim_font = ImageFont.truetype(FONT_REGULAR, 34)
    spec_font = ImageFont.truetype(FONT_REGULAR, 36)
    price_font = ImageFont.truetype(FONT_BOLD, 46)
    site_font = ImageFont.truetype(FONT_BOLD, 30)

    y = ch - scrim_h + 40
    title = f"{fields['brand']} {fields['model']}".strip()
    draw.text((pad, y), title, font=title_font, fill=(255, 255, 255, 255))
    y += 78
    if fields.get("trim"):
        draw.text((pad, y), str(fields["trim"]), font=trim_font, fill=(230, 230, 230, 255))
        y += 46

    spec_parts = [fields.get(k) for k in ("year", "mileage", "fuel", "transmission", "body")]
    spec = " · ".join(str(p) for p in spec_parts if p)
    if spec:
        draw.text((pad, y), spec, font=spec_font, fill=(210, 210, 210, 255))

    price_text = str(fields.get("price") or "")
    if price_text:
        tb = draw.textbbox((0, 0), price_text, font=price_font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pill_w, pill_h = tw + 64, th + 44
        px0, py0 = cw - pill_w - 40, 120
        # radius == pill_h // 2 exactly (a full pill cap) hits a Pillow 9.5.0
        # rounded_rectangle edge case ("y1 must be greater than or equal to
        # y0") with certain font metrics (confirmed with real DejaVu on the
        # production font) — stay a couple px under the boundary.
        radius = max(1, pill_h // 2 - 2)
        draw.rounded_rectangle([px0, py0, px0 + pill_w, py0 + pill_h], radius=radius, fill=(20, 110, 90, 235))
        draw.text((px0 + 32, py0 + 22 - th // 2 - tb[1]), price_text, font=price_font, fill=(255, 255, 255, 255))

    logo_path = os.path.join(ASSET_DIR, "logo.png")
    if os.path.exists(logo_path):
        logo = Image.open(logo_path).convert("RGBA")
        logo_w = 260
        logo_h = int(logo.height * (logo_w / logo.width))
        logo = logo.resize((logo_w, logo_h), Image.LANCZOS)
        alpha = logo.split()[3].point(lambda a: int(a * 0.8))
        logo.putalpha(alpha)
        overlay.paste(logo, (pad, ch - logo_h - 44), logo)
    elif fields.get("site_url"):
        draw.text((pad, ch - 50), str(fields["site_url"]), font=site_font, fill=(255, 255, 255, 160))

    return overlay


def _build_filter_complex(n: int) -> tuple[str, float]:
    """Ken Burns (zoompan) per photo input, chained crossfades (xfade) between
    consecutive clips, then ONE overlay of the branded PNG (input index n) on
    top of the finished crossfade chain — composited after the fades, not
    baked into each frame, so the static text never double-exposes during a
    transition. Returns (filter_complex_string, total_duration_seconds)."""
    hold_frames = int((CLIP_DUR + XFADE_DUR) * FPS)
    parts = []
    for i in range(n):
        zoom_in = i % 2 == 0
        if zoom_in:
            z_expr = "min(zoom+0.0015,1.18)"
        else:
            z_expr = "if(eq(on,1),1.18,max(1.18-0.0015*on,1.0))"
        parts.append(
            f"[{i}:v]zoompan=z='{z_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={hold_frames}:s={CANVAS[0]}x{CANVAS[1]}:fps={FPS},setsar=1[z{i}]"
        )

    # Cumulative offset: each xfade starts XFADE_DUR before the running total ends.
    running = CLIP_DUR
    prev = "z0"
    for i in range(1, n):
        out_label = f"x{i}"
        offset = running - XFADE_DUR
        parts.append(f"[{prev}][z{i}]xfade=transition=fade:duration={XFADE_DUR}:offset={offset:.3f}[{out_label}]")
        running = running + CLIP_DUR - XFADE_DUR
        prev = out_label
    total = running

    parts.append(f"[{prev}][{n}:v]overlay=0:0:format=auto[vout]")

    filter_complex = ";\n".join(parts)
    return filter_complex, total


def _run_ffmpeg(frame_paths: list[Path], overlay_path: Path, out_path: Path) -> None:
    n = len(frame_paths)
    filter_complex, total_duration = _build_filter_complex(n)
    hold_seconds = CLIP_DUR + XFADE_DUR

    cmd = ["ffmpeg", "-y"]
    for p in frame_paths:
        cmd += ["-loop", "1", "-framerate", str(FPS), "-t", f"{hold_seconds:.3f}", "-i", str(p)]
    cmd += ["-loop", "1", "-i", str(overlay_path)]  # input n: the branded overlay PNG
    cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]  # input n+1: silent audio

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", f"{n + 1}:a",
        "-t", f"{total_duration:.3f}",
        "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart",
        str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=180)
    if result.returncode != 0:
        raise ReelError(result.stderr.decode(errors="replace")[-2000:])


def render_reel(photo_urls: list[str], fields: dict, workdir: Path) -> Path:
    """Download photos, build story frames + overlay, render the MP4 into
    workdir, return its path. Raises ReelError on any unrecoverable failure."""
    workdir.mkdir(parents=True, exist_ok=True)
    candidates = _pick_photos(photo_urls, MAX_PHOTOS)

    downloaded: list[Path] = []
    for i, url in enumerate(candidates):
        dest = workdir / f"src_{i:02d}.jpg"
        if _download(url, dest):
            downloaded.append(dest)

    if len(downloaded) < MIN_PHOTOS:
        raise ReelError(f"only {len(downloaded)} of {len(candidates)} photos downloaded successfully (need >= {MIN_PHOTOS})")

    overlay_path = workdir / "overlay.png"
    _render_overlay(fields).save(overlay_path)

    frame_paths = []
    for i, src in enumerate(downloaded):
        dest = workdir / f"frame_{i:02d}.jpg"
        _prepare_frame(src, dest)
        frame_paths.append(dest)

    out_path = workdir / "reel.mp4"
    _run_ffmpeg(frame_paths, overlay_path, out_path)
    return out_path
