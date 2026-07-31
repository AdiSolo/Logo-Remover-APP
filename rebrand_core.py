"""
Core rebranding pipeline: detect Encar watermark + plate, AI-inpaint them out,
paste our own logo in both spots.

Detection is color/position adaptive (works across image sizes):
  - watermark: red/orange strokes in the top region of a plain background
  - plate:     largest red blob on the grille (lower half), white text masked

Inpainting uses LaMa (via iopaint) for structure-aware fills, with an OpenCV
fallback that needs no model.
"""
import os
import cv2
import numpy as np

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


# --------------------------------------------------------------------------
# Logo handling
# --------------------------------------------------------------------------
def load_logo(path=None):
    """Load the BGRA logo (white+blue on transparent)."""
    path = path or os.path.join(ASSET_DIR, "logo.png")
    logo = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        raise FileNotFoundError(f"logo not found at {path}")
    if logo.shape[2] == 3:  # add opaque alpha if missing
        logo = cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)
    return logo


def dark_variant(logo, color=(58, 42, 34, 255)):
    """Recolor the white part to dark navy so it reads on light backgrounds."""
    out = logo.copy()
    b, g, r, a = cv2.split(out)
    whiteish = (b > 150) & (g > 150) & (r > 150) & (a > 0)
    out[whiteish] = color
    return out


def paste(dst, logo, cx, cy, target_w):
    """Alpha-composite `logo` centered at (cx, cy), scaled to `target_w` px wide."""
    h, w = logo.shape[:2]
    scale = target_w / w
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    lg = cv2.resize(logo, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = int(cx - nw / 2), int(cy - nh / 2)
    x1, y1 = x0 + nw, y0 + nh
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(dst.shape[1], x1), min(dst.shape[0], y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    lx0, ly0 = dx0 - x0, dy0 - y0
    roi = dst[dy0:dy1, dx0:dx1]
    sub = lg[ly0:ly0 + roi.shape[0], lx0:lx0 + roi.shape[1]]
    al = (sub[:, :, 3] / 255.0)[..., None]
    dst[dy0:dy1, dx0:dx1] = (sub[:, :, :3] * al + roi * (1 - al)).astype(np.uint8)


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def _red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return (((h < 20) | (h > 165)) & (s > 60) & (v > 60)).astype(np.uint8) * 255


# Known "Trust Encar" watermark variants (assets/logos-to-remove/*.png), loaded
# once and matched against dark-background photos where colour detection can't help.
_WM_TMPLS = None


def _wm_templates():
    global _WM_TMPLS
    if _WM_TMPLS is None:
        _WM_TMPLS = []
        d = os.path.join(ASSET_DIR, "logos-to-remove")
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    t = cv2.imread(os.path.join(d, fn), cv2.IMREAD_GRAYSCALE)
                    if t is not None:
                        _WM_TMPLS.append(t)
    return _WM_TMPLS


def _match_watermark_topright(img, min_score=0.62):
    """Locate the watermark by matching known logo templates in the top-right region.
    Returns a box only on a HIGH-confidence match — low scores mislocalise onto the
    car, so we'd rather skip (leave a faint mark) than risk inpainting the vehicle."""
    tmpls = _wm_templates()
    if not tmpls:
        return None
    H, W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x0, y1 = int(0.42 * W), int(0.45 * H)  # watermark always sits in the top-right
    sub = g[0:y1, x0:W]
    best = None
    for t in tmpls:
        th, tw = t.shape
        tf = t.astype(np.float32)
        for s in np.linspace(0.30, 1.20, 16):
            w, h = int(tw * s), int(th * s)
            if w < 40 or h < 20 or w >= sub.shape[1] or h >= sub.shape[0]:
                continue
            r = cv2.matchTemplate(sub, cv2.resize(tf, (w, h)), cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if best is None or mx > best[0]:
                best = (mx, (loc[0] + x0, loc[1]), w, h)
    if best and best[0] >= min_score:
        _, loc, w, h = best
        return (loc[0], loc[1], loc[0] + w, loc[1] + h)
    return None


# Optional trained detector (assets/wm_detector.pt). Preferred when present; the
# heuristic below is the fallback so the tool works with or without the model.
_WM_MODEL = None


def _wm_model():
    global _WM_MODEL
    if _WM_MODEL is None:
        p = os.path.join(ASSET_DIR, "wm_detector.pt")
        if os.path.exists(p):
            try:
                from ultralytics import YOLO
                _WM_MODEL = YOLO(p)
            except Exception as e:
                print(f"[wm] could not load detector: {e}")
                _WM_MODEL = False
        else:
            _WM_MODEL = False
    return _WM_MODEL


# Confidence floor for the trained detector. The v2 model is precision-first
# (val P≈0.97 @ mAP50 0.99), so 0.40 keeps recall high while not firing on the
# clean/dealer photos that must be left untouched. Tune via env if needed.
WM_ML_CONF = float(os.environ.get("REBRAND_WM_ML_CONF", "0.40"))

# Max half-stroke-width (px) a masked component may have before it's treated as a
# solid car-body intrusion (not watermark text) and dropped. See build_mask.
WM_MAX_STROKE_HALF = float(os.environ.get("REBRAND_WM_MAX_STROKE_HALF", "12"))


# Low internal floor for the initial detection pass: on hard backgrounds (near-white
# walls, colour-matching gradients) the model often splits ONE watermark into two
# overlapping boxes — a tight high-confidence box around bold "Encar" plus a looser,
# lower-confidence box that also spans the "Trust" script above it. Detecting at a
# low floor lets us see that companion box; WM_ML_CONF still gates whether we act at
# all (see below).
WM_ML_DETECT_FLOOR = float(os.environ.get("REBRAND_WM_ML_DETECT_FLOOR", "0.15"))


def _boxes_overlap(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def _detect_watermark_ml(img, conf=None):
    m = _wm_model()
    if not m:
        return None
    conf = WM_ML_DETECT_FLOOR if conf is None else conf
    try:
        res = m.predict(img, conf=conf, verbose=False)[0]
    except Exception:
        return None
    H, W = img.shape[:2]
    cand = []
    for b in res.boxes:
        if int(b.cls) != 0:  # only the watermark class
            continue
        x0, y0, x1, y1 = [int(v) for v in b.xyxy[0].tolist()]
        # The 'Trust Encar' watermark is ALWAYS top-right and fixed-position. Reject
        # boxes elsewhere (a dealer's own centre/left logo text) so clean dealer photos
        # are never inpainted/smudged — precision matters far more than one extra catch.
        cx = (x0 + x1) / 2
        if cx < 0.45 * W or y0 > 0.40 * H:
            continue
        cand.append((float(b.conf), (x0, y0, x1, y1)))
    if not cand:
        return None
    cand.sort(key=lambda c: -c[0])
    # Require a genuinely confident primary detection (WM_ML_CONF) before acting at
    # all — this is what keeps dealer signage from ever triggering a clean.
    if cand[0][0] < WM_ML_CONF:
        return None
    px0, py0, px1, py1 = cand[0][1]
    uy0, uy1 = py0, py1
    for c, box in cand[1:]:
        # Only extend the VERTICAL range from an overlapping companion box, keep the
        # primary (tighter, higher-confidence) horizontal range as-is. Unioning width
        # too would sweep in extra car surface at the sides on some angles.
        if _boxes_overlap((px0, py0, px1, py1), box):
            x0, y0, x1, y1 = box
            uy0, uy1 = min(uy0, y0), max(uy1, y1)
    return (px0, uy0, px1, uy1)


# Edge-template detector. Matches the "Encar" wordmark by its EDGE structure
# (polarity-independent → works whether the semi-transparent watermark renders
# dark-on-light or light-on-dark), then expands the box up/right to also cover the
# small "Trust" script. This is the primary detector; it is far more reliable across
# Encar's backgrounds than colour/ML, which missed dark-background watermarks.
_WM_EDGE_TMPL = None
# Precision-first cutoff: 0.40 excludes bold dealer signage (e.g. "TOYOTA CERTIFIED"
# ~0.34) while keeping real watermarks (0.40+). ~63% recall; the faint tail is left
# untouched (no harm) rather than risking damage to clean/dealer photos.
WM_EDGE_MIN_SCORE = float(os.environ.get("REBRAND_WM_MIN_SCORE", "0.40"))


def _edges(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _wm_edge_template():
    """Edge map of the 'Encar' wordmark, cropped from the averaged template asset."""
    global _WM_EDGE_TMPL
    if _WM_EDGE_TMPL is None:
        t = cv2.imread(os.path.join(ASSET_DIR, "wm_template.jpg"), cv2.IMREAD_GRAYSCALE)
        _WM_EDGE_TMPL = _edges(t[35:105, 80:300]) if t is not None else False
    return _WM_EDGE_TMPL


def _detect_watermark_edge(img):
    """(score, full_box) — box covers 'Trust Encar', expanded up/right from the
    'Encar' match. box is None if no template asset or no location found."""
    tmpl = _wm_edge_template()
    if tmpl is False:
        return 0.0, None
    TH, TW = tmpl.shape
    H, W = img.shape[:2]
    ox = int(0.46 * W)
    roi = img[0:int(0.32 * H), ox:W]
    if roi.shape[0] < TH or roi.shape[1] < TW:
        return 0.0, None
    e = _edges(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
    best, bloc, bs = -1.0, None, 1.0
    for s in (0.8, 0.9, 1.0, 1.1, 1.25):
        tw, th = int(TW * s), int(TH * s)
        if e.shape[0] < th or e.shape[1] < tw:
            continue
        r = cv2.matchTemplate(e, cv2.resize(tmpl, (tw, th)), cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(r)
        if mx > best:
            best, bloc, bs = mx, ml, s
    if bloc is None:
        return best, None
    tw, th = int(TW * bs), int(TH * bs)
    ex0, ey0 = ox + bloc[0], bloc[1]        # 'Encar' top-left in full image
    x0 = int(ex0 - 0.15 * tw); x1 = int(ex0 + tw + 0.45 * tw)
    y0 = int(ey0 - 1.55 * th); y1 = int(ey0 + th + 0.18 * th)
    return best, (max(0, x0), max(0, y0), min(W, x1), min(H, y1))


# Below this local-background brightness a real "Trust Encar" watermark can render
# fully achromatic (grey/white on a dark-ish studio wall) — see
# _watermark_color_consistent. Above it, a real watermark reliably keeps its
# red/orange tint, so the colour check applies. Calibrated on a 100-photo scan of a
# real cleaning batch: achromatic renders clustered at bg<=70 (up to a Hyundai
# Ioniq6 at 69-70), coloured renders started at bg>=152 — a clean gap, so 100 sits
# safely in the middle with margin on both sides.
WM_DARK_BG_MAX = float(os.environ.get("REBRAND_WM_DARK_BG_MAX", "100"))
# Minimum fraction of the kept (text-stroke) pixels that must fall in the red/orange
# hue range on a light background. Same 100-photo scan: every genuine light-bg
# watermark showed >=0.017 (even faint ones); the only exact 0.000 was the known
# false positive (blue/grey "DEUTSCH AUTOWORLD" dealer signage, twice). 0.01 keeps a
# safety margin below the real floor while still requiring more than stray-pixel noise.
WM_MIN_RED_FRAC = float(os.environ.get("REBRAND_WM_MIN_RED_FRAC", "0.01"))


def _watermark_color_consistent(img, box):
    """Final safety net: on a light background, a real 'Trust Encar' watermark's
    strokes are reliably red/orange-tinted. Dealer signage in other colours (e.g.
    "DEUTSCH AUTOWORLD" in blue/grey) can otherwise pass position + shape gating —
    this catches it by colour instead. Skipped on dark backgrounds, where the real
    watermark itself can render achromatic (grey/white), so the check would
    otherwise reject genuine dark-bg detections."""
    x0, y0, x1, y1 = box
    reg = img[y0:y1, x0:x1]
    if reg.size == 0:
        return False
    gray = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
    if int(np.median(gray)) <= WM_DARK_BG_MAX:
        return True
    keepm = _watermark_stroke_mask(gray)
    total = int((keepm > 0).sum())
    if total == 0:
        return False
    red_reg = _red_mask(reg)
    frac = float(((keepm > 0) & (red_reg > 0)).sum()) / total
    return frac >= WM_MIN_RED_FRAC


def detect_watermark_box(img, red):
    """Locate the 'Trust Encar' watermark. UNION of two precision-guarded detectors:
      1. trained YOLO model (assets/wm_detector.pt) — high recall, catches the
         dark-background / faint watermarks the edge matcher missed (~50% recall wall);
      2. edge-template match — the backstop, used when the model doesn't fire.
    The model is tried first; on a miss we fall back to the edge match. Both only
    return a box on a confident detection, so clean/dealer photos stay untouched.
    The legacy colour heuristic runs only if neither the model nor the template ship.
    Every candidate box then passes through _watermark_color_consistent as a final
    cross-check before being accepted."""
    box = None
    ml = _detect_watermark_ml(img)
    if ml is not None:
        box = ml
    elif _wm_edge_template() is not False:
        score, edge_box = _detect_watermark_edge(img)
        box = edge_box if (edge_box is not None and score >= WM_EDGE_MIN_SCORE) else None
    else:
        H, W = img.shape[:2]
        top = red.copy()
        top[int(H * 0.30):, :] = 0
        n, _, st, _ = cv2.connectedComponentsWithStats(top)
        keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 150]
        if keep:
            xs = [st[i, cv2.CC_STAT_LEFT] for i in keep] + [st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] for i in keep]
            ys = [st[i, cv2.CC_STAT_TOP] for i in keep] + [st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] for i in keep]
            pad = 22
            box = (int(max(0, min(xs) - pad)), int(max(0, min(ys) - pad)),
                   int(min(W, max(xs) + pad)), int(min(H, max(ys) + pad)))
        else:
            box = _match_watermark_topright(img)

    if box is not None and not _watermark_color_consistent(img, box):
        return None
    return box


def detect_plate_box(img, red):
    """Largest PLATE-SHAPED red blob in the lower half — the dealer plate.
    Size/aspect constrained so a red car panel is never mistaken for a plate
    (otherwise inpainting would carve into the vehicle)."""
    H, W = img.shape[:2]
    low = red.copy()
    low[:int(H * 0.45), :] = 0
    n, _, st, _ = cv2.connectedComponentsWithStats(low)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < 500:
        return None
    x, y, w, h = (int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP]),
                  int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT]))
    # A plate is small and wide. Reject car-panel-sized/shaped blobs so we never
    # inpaint into the vehicle (e.g. a red-bodied car fills the lower half).
    if not (0.03 * W <= w <= 0.28 * W and 0.02 * H <= h <= 0.12 * H and 1.3 <= w / max(h, 1) <= 6.0):
        return None
    return (x, y, w, h)


# Fraction of the watermark box (from the top) treated as the "text zone" — the
# bold "Encar" + script "Trust" always live here, and a bold, large rendering of
# the wordmark can have strokes several px thicker than a car body's edge sliver at
# small scale, so a single global thickness gate can't tell them apart reliably.
# Below this fraction is the "danger zone", the strip nearest the box's bottom edge
# where a car roof/body actually can intrude — only there is the strict
# WM_MAX_STROKE_HALF gate applied. This split removes the whole wordmark cleanly
# while still refusing to ever mask a car that intrudes from below.
WM_TEXT_ZONE_FRAC = float(os.environ.get("REBRAND_WM_TEXT_ZONE_FRAC", "0.80"))


def _thresh_components(gray, area_cap, max_half):
    """Components of `gray` differing from local background by >22, area-capped,
    and (if max_half is set) dropped when their peak half-thickness exceeds it."""
    bg = int(np.median(gray))
    txt = (np.abs(gray.astype(np.int16) - bg) > 22).astype(np.uint8) * 255
    nn, lab, stt, _ = cv2.connectedComponentsWithStats(txt)
    dist = cv2.distanceTransform(txt, cv2.DIST_L2, 3) if max_half is not None else None
    keep = np.zeros_like(txt)
    for i in range(1, nn):
        a = stt[i, cv2.CC_STAT_AREA]
        if a < 5 or a > area_cap:
            continue
        if max_half is not None and float(dist[lab == i].max()) > max_half:
            continue
        keep[lab == i] = 255
    return keep


def _watermark_stroke_mask(reg_gray):
    """Which pixels of the watermark-box crop to inpaint. Zoned: the top
    WM_TEXT_ZONE_FRAC is pure text territory (no thickness gate — keep everything
    stroke-shaped, however bold), the bottom strip is the only place a car body could
    intrude so it alone gets the strict WM_MAX_STROKE_HALF gate."""
    RH, RW = reg_gray.shape[:2]
    area_cap = 0.30 * RW * RH
    split = int(RH * WM_TEXT_ZONE_FRAC)
    keepm = np.zeros((RH, RW), np.uint8)
    keepm[:split] = _thresh_components(reg_gray[:split], area_cap, None)
    keepm[split:] = _thresh_components(reg_gray[split:], area_cap, WM_MAX_STROKE_HALF)
    return keepm


def build_mask(img, red, wm_box, plate_box, full_plate=False):
    """Pixels to inpaint.

    Watermark: ALWAYS targeted to the red/orange stroke pixels only (never the whole
    box) — the watermark box can overlap the top of the car, so a full-box fill would
    erase the roof. Masking only the coloured strokes removes the text safely.

    Plate: full_plate=True (clean-only mode) removes the ENTIRE plate box → a clean
    bumper. Otherwise only the white text is removed (so a pasted logo covers the rest).
    The plate box is size/shape-guarded upstream, so a full fill can't eat the car."""
    H, W = img.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    if wm_box:
        x0, y0, x1, y1 = wm_box
        # Polarity-agnostic: mask pixels that differ from the local background (the
        # watermark strokes, red OR white). Keep text-sized STROKES so both "Trust"
        # (script) and "Encar" (bold) are removed.
        reg = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        keepm = _watermark_stroke_mask(reg)
        # Merge the kept strokes into a contiguous patch so LaMa fills cleanly (a
        # stroke-thin mask leaves coloured residue on smooth walls). Gentle kernels —
        # the car was already dropped above, so this can't grow onto the vehicle.
        keepm = cv2.morphologyEx(keepm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        keepm = cv2.dilate(keepm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
        mask[y0:y1, x0:x1] = cv2.bitwise_or(mask[y0:y1, x0:x1], keepm)
    if plate_box:
        px, py, pw, ph = plate_box
        pm = np.zeros((H, W), np.uint8)
        if full_plate:
            pm[py:py + ph, px:px + pw] = 255
        else:
            rs = cv2.cvtColor(img[py:py + ph, px:px + pw], cv2.COLOR_BGR2HSV)
            white = ((rs[:, :, 1] < 80) & (rs[:, :, 2] > 140)).astype(np.uint8) * 255
            pm[py:py + ph, px:px + pw] = white
        # The plate box is size/shape-guarded upstream, so it can dilate freely.
        pm = cv2.dilate(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)
        mask = cv2.bitwise_or(mask, pm)
    return mask


# --------------------------------------------------------------------------
# Inpainting backends
# --------------------------------------------------------------------------
_LAMA = None


def _get_lama():
    global _LAMA
    if _LAMA is None:
        from iopaint.model_manager import ModelManager
        _LAMA = ModelManager(name="lama", device=os.environ.get("REBRAND_DEVICE", "cpu"))
    return _LAMA


def inpaint(img, mask, method="lama"):
    if method == "opencv":
        return cv2.inpaint(img, mask, 10, cv2.INPAINT_TELEA)
    from iopaint.schema import InpaintRequest, HDStrategy
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    req = InpaintRequest(hd_strategy=HDStrategy.CROP, hd_strategy_crop_margin=48,
                         hd_strategy_crop_trigger_size=800, hd_strategy_resize_limit=2048)
    out = _get_lama()(rgb, mask, req).astype(np.uint8)
    # iopaint returns BGR already; guard shape just in case
    return out if (out.ndim == 3 and out.shape[2] == 3) else cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
def rebrand(img, logo, method="lama", paste_logo=True):
    """Return (result_bgr, info). Removes Encar marks; optionally pastes our logo.

    paste_logo=True  → rebrand: inpaint out the marks and stamp our logo (default).
    paste_logo=False → clean only: inpaint out the marks, leave the area bare
                       (use for neutral/white-label catalogs).
    """
    red = _red_mask(img)
    wm_box = detect_watermark_box(img, red)
    plate_box = detect_plate_box(img, red)
    info = {"watermark": bool(wm_box), "plate": bool(plate_box), "method": method, "logo": bool(paste_logo)}

    if not wm_box and not plate_box:
        return img, info  # nothing detected; return unchanged

    mask = build_mask(img, red, wm_box, plate_box, full_plate=not paste_logo)
    out = inpaint(img, mask, method=method)

    if paste_logo:
        if plate_box:
            px, py, pw, ph = plate_box
            paste(out, logo, px + pw / 2, py + ph / 2, pw * 0.82)
        if wm_box:
            x0, y0, x1, y1 = wm_box
            paste(out, dark_variant(logo), (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 1.05)
    return out, info
