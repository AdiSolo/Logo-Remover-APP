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
     photos (xfade), then the branded overlay from step 3 composited once over
     that whole chain. A static end-card (CTA text + logo, own dark
     background — no title/price overlay on it) is concatenated on afterward.
     Muxed with a silent AAC track, encoded as H.264/yuv420p with +faststart
     for social-platform compatibility.

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
OUTRO_DUR = float(os.environ.get("VIDEO_OUTRO_DUR", "3.0"))  # seconds the end-card holds
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


# _prepare_frame contain-fits the (consistently ~1160x696, per Encar's own
# resize params) photo onto the 1080x1920 canvas, centered — so its top/bottom
# edge is a known constant, not something read per-photo here. Text lives
# entirely BELOW that edge, in its own solid card, so it never sits on top of
# the car (the earlier gradient-scrim-over-the-photo design did, and looked
# like it was competing with the image).
_ENCAR_PHOTO_W, _ENCAR_PHOTO_H = 1160, 696
_contain_scale = min(CANVAS[0] / _ENCAR_PHOTO_W, CANVAS[1] / _ENCAR_PHOTO_H)
_displayed_photo_h = int(_ENCAR_PHOTO_H * _contain_scale)
PHOTO_ZONE_BOTTOM = (CANVAS[1] + _displayed_photo_h) // 2  # == 1284

INFO_BG = (10, 10, 14, 255)


def _render_overlay(fields: dict) -> Image.Image:
    cw, ch = CANVAS
    overlay = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Solid info card filling the rest of the canvas below the photo.
    draw.rectangle([0, PHOTO_ZONE_BOTTOM, cw, ch], fill=INFO_BG)

    pad = 56
    title_font = ImageFont.truetype(FONT_BOLD, 64)
    trim_font = ImageFont.truetype(FONT_REGULAR, 34)
    spec_font = ImageFont.truetype(FONT_REGULAR, 36)
    price_font = ImageFont.truetype(FONT_BOLD, 46)
    cta_font = ImageFont.truetype(FONT_BOLD, 38)

    max_text_width = cw - 2 * pad

    y = PHOTO_ZONE_BOTTOM + 36
    title = f"{fields['brand']} {fields['model']}".strip()
    for line in _wrap_words(draw, title, title_font, max_text_width):
        draw.text((pad, y), line, font=title_font, fill=(255, 255, 255, 255))
        y += 84
    if fields.get("trim"):
        draw.text((pad, y), str(fields["trim"]), font=trim_font, fill=(230, 230, 230, 255))
        y += 56

    spec_parts = [str(p) for p in (fields.get(k) for k in ("year", "mileage", "fuel", "transmission", "body")) if p]
    for line in _wrap_parts(draw, spec_parts, " · ", spec_font, max_text_width):
        draw.text((pad, y), line, font=spec_font, fill=(210, 210, 210, 255))
        y += 44

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

    # AutoCo.ro's own brand mark — NOT assets/logo.png (that one is the
    # Encar-lookalike wordmark pasted onto photos by the unrelated /rebrand
    # logo=titanic feature; wrong brand for a reel we post ourselves). Top-left,
    # roughly level with the price pill on the top-right.
    logo_path = os.path.join(ASSET_DIR, "autoco-logo.png")
    if os.path.exists(logo_path):
        logo = Image.open(logo_path).convert("RGBA")
        logo_w = 200
        logo_h = int(logo.height * (logo_w / logo.width))
        logo = logo.resize((logo_w, logo_h), Image.LANCZOS)
        alpha = logo.split()[3].point(lambda a: int(a * 0.9))
        logo.putalpha(alpha)
        overlay.paste(logo, (pad, 128), logo)

    # CTA pill, stacked right below the text block within the same info card
    # (not pinned to the bottom edge — it just follows wherever the text ends,
    # so 1- or 2-line specs both look right).
    y += 30
    cta_text = "Vezi detalii →"
    cb = draw.textbbox((0, 0), cta_text, font=cta_font)
    cta_w, cta_h = cb[2] - cb[0], cb[3] - cb[1]
    cta_pill_w, cta_pill_h = cta_w + 56, cta_h + 38
    cta_radius = max(1, cta_pill_h // 2 - 2)
    draw.rounded_rectangle([pad, y, pad + cta_pill_w, y + cta_pill_h], radius=cta_radius, fill=(20, 110, 90, 235))
    draw.text((pad + 28, y + 19 - cta_h // 2 - cb[1]), cta_text, font=cta_font, fill=(255, 255, 255, 255))

    return overlay


def _wrap_words(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Greedy word-wrap so long brand+model combos don't run off the canvas
    (confirmed happening: DejaVu renders noticeably wider than the Arial used
    to eyeball this locally, and real listing text is longer/shorter than
    whatever sample was tested)."""
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for w in words:
        candidate = f"{current} {w}".strip()
        if not current or draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def _wrap_parts(draw: ImageDraw.ImageDraw, parts: list[str], sep: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Like _wrap_words, but wraps at `sep`-joined part boundaries (e.g. the
    spec line's " · "-separated fields) so a single part like "71.973 km"
    never splits mid-way."""
    lines: list[str] = []
    current: list[str] = []
    for p in parts:
        candidate = sep.join(current + [p])
        if not current or draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current.append(p)
        else:
            lines.append(sep.join(current))
            current = [p]
    if current:
        lines.append(sep.join(current))
    return lines


def _centered_x(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, cw: int) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    return (cw - w) // 2 - bbox[0]


def _render_outro_frame() -> Image.Image:
    """Static end-card appended after the last photo: CTA text + our logo, on
    its own dark background (NOT a car photo, no title/price overlay) — a
    clean closing beat rather than another listing slide."""
    cw, ch = CANVAS
    canvas = Image.new("RGB", CANVAS, (12, 14, 20))
    draw = ImageDraw.Draw(canvas)

    cta_font = ImageFont.truetype(FONT_REGULAR, 42)
    site_font = ImageFont.truetype(FONT_BOLD, 68)
    cta_text = "Vezi mai multe oferte pe:"
    site_text = "autoco.ro"

    logo_path = os.path.join(ASSET_DIR, "autoco-logo.png")
    logo_img = None
    logo_w = logo_h = 0
    if os.path.exists(logo_path):
        logo_img = Image.open(logo_path).convert("RGBA")
        logo_w = 340
        logo_h = int(logo_img.height * (logo_w / logo_img.width))
        logo_img = logo_img.resize((logo_w, logo_h), Image.LANCZOS)

    cta_h = draw.textbbox((0, 0), cta_text, font=cta_font)[3]
    site_h = draw.textbbox((0, 0), site_text, font=site_font)[3]
    gap = 32
    block_h = cta_h + gap + site_h + gap * 2 + logo_h
    y = (ch - block_h) // 2

    draw.text((_centered_x(draw, cta_text, cta_font, cw), y), cta_text, font=cta_font, fill=(210, 210, 210, 255))
    y += cta_h + gap
    draw.text((_centered_x(draw, site_text, site_font, cw), y), site_text, font=site_font, fill=(255, 255, 255, 255))
    y += site_h + gap * 2
    if logo_img:
        canvas.paste(logo_img, ((cw - logo_w) // 2, y), logo_img)

    return canvas


def _build_filter_complex(n: int) -> tuple[str, float]:
    """Ken Burns (zoompan) per photo input, chained crossfades (xfade) between
    consecutive clips, then ONE overlay of the branded PNG (input index n) on
    top of the finished crossfade chain — composited after the fades, not
    baked into each frame, so the static text never double-exposes during a
    transition. Returns (filter_complex_string, total_duration_seconds)."""
    hold_seconds = CLIP_DUR + XFADE_DUR
    hold_frames = int(hold_seconds * FPS)
    parts = []
    for i in range(n):
        zoom_in = i % 2 == 0
        if zoom_in:
            z_expr = "min(zoom+0.0015,1.18)"
        else:
            z_expr = "if(eq(on,1),1.18,max(1.18-0.0015*on,1.0))"
        # zoompan's `d` does NOT reliably bound its own output length when fed a
        # looped still image (confirmed empirically: without an explicit trim,
        # a solo zoompan clip ran well past 60s instead of stopping at 108
        # frames/3.6s) — an explicit trim+setpts makes each clip's length exact
        # and deterministic, which the xfade offset math below depends on.
        # `fps=` is repeated explicitly after trim/setpts — on at least one
        # ffmpeg build (the production VPS, not reproduced locally) trim
        # drops the constant-frame-rate metadata xfade requires ("current
        # rate of 1/0 is invalid"); re-stamping it here is cheap and harmless
        # where it wasn't actually needed.
        parts.append(
            f"[{i}:v]zoompan=z='{z_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={hold_frames}:s={CANVAS[0]}x{CANVAS[1]}:fps={FPS},"
            f"trim=duration={hold_seconds:.3f},setpts=PTS-STARTPTS,fps={FPS},setsar=1[z{i}]"
        )

    # Cumulative offset: each xfade starts XFADE_DUR before the running total ends.
    # xfade's real output length is offset + len(second_input) — verified empirically
    # (a solid-color 2-clip test: offset=2.4, second input 3.6s long -> output 6.0s,
    # not 5.4s as a naive offset+duration guess would give). So the running total
    # must start at hold_seconds (len of z0 as fed into the first merge), not
    # CLIP_DUR — starting from CLIP_DUR undercounts every offset by XFADE_DUR and
    # compounds, silently truncating the last photo(s) (and anything appended after,
    # like the outro card) once -t total_duration cuts the output short.
    running = CLIP_DUR + XFADE_DUR
    prev = "z0"
    for i in range(1, n):
        out_label = f"x{i}"
        offset = running - XFADE_DUR
        parts.append(f"[{prev}][z{i}]xfade=transition=fade:duration={XFADE_DUR}:offset={offset:.3f}[{out_label}]")
        # new_running = offset + len(z_i) = (running - XFADE_DUR) + hold_seconds = running + CLIP_DUR
        running = running + CLIP_DUR
        prev = out_label
    total = running

    # Named [main_out], not [vout] — the outro end-card (its own segment, no
    # title/price overlay) is concatenated on afterward in _run_ffmpeg to
    # produce the actual final [vout].
    # shortest=1 is essential: the overlay PNG input is an infinitely-looped
    # still (-loop 1, no -t), and overlay's default shortest=0 holds/freezes
    # the main chain's LAST frame until the longer (infinite) input ends —
    # confirmed empirically this silently ate the rest of the timeline (the
    # outro concat never got reached; -t just cut off mid-freeze).
    parts.append(f"[{prev}][{n}:v]overlay=0:0:format=auto:shortest=1[main_out]")

    filter_complex = ";\n".join(parts)
    return filter_complex, total


def _run_ffmpeg(frame_paths: list[Path], overlay_path: Path, outro_path: Path, out_path: Path) -> None:
    n = len(frame_paths)
    filter_complex, main_duration = _build_filter_complex(n)
    hold_seconds = CLIP_DUR + XFADE_DUR
    total_duration = main_duration + OUTRO_DUR

    cmd = ["ffmpeg", "-y"]
    for p in frame_paths:
        cmd += ["-loop", "1", "-framerate", str(FPS), "-t", f"{hold_seconds:.3f}", "-i", str(p)]
    cmd += ["-loop", "1", "-i", str(overlay_path)]  # input n: the branded overlay PNG
    cmd += ["-loop", "1", "-framerate", str(FPS), "-t", f"{OUTRO_DUR:.3f}", "-i", str(outro_path)]  # input n+1: end-card
    cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]  # input n+2: silent audio

    outro_idx = n + 1
    filter_complex += (
        f";\n[{outro_idx}:v]fps={FPS},setsar=1[outro_z]"
        f";\n[main_out][outro_z]concat=n=2:v=1:a=0[vout]"
    )

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", f"{n + 2}:a",
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

    outro_path = workdir / "outro.jpg"
    _render_outro_frame().save(outro_path, quality=90)

    frame_paths = []
    for i, src in enumerate(downloaded):
        dest = workdir / f"frame_{i:02d}.jpg"
        _prepare_frame(src, dest)
        frame_paths.append(dest)

    out_path = workdir / "reel.mp4"
    _run_ffmpeg(frame_paths, overlay_path, outro_path, out_path)
    return out_path
