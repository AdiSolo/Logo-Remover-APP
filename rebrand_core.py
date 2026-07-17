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


def _detect_watermark_ml(img, conf=0.35):
    m = _wm_model()
    if not m:
        return None
    try:
        res = m.predict(img, conf=conf, verbose=False)[0]
    except Exception:
        return None
    best = None
    for b in res.boxes:
        if int(b.cls) == 0:  # watermark
            c = float(b.conf)
            if best is None or c > best[0]:
                best = (c, [int(v) for v in b.xyxy[0].tolist()])
    if best:
        x0, y0, x1, y1 = best[1]
        return (x0, y0, x1, y1)
    return None


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


def detect_watermark_box(img, red):
    """Locate the 'Trust Encar' watermark. Primary: edge-template match (precision-
    first — only returns a box on a confident match, else None so clean/dealer photos
    are left untouched). Legacy ML/colour/template heuristic is the fallback used only
    when no template asset ships."""
    if _wm_edge_template() is not False:
        score, box = _detect_watermark_edge(img)
        return box if (box is not None and score >= WM_EDGE_MIN_SCORE) else None
    ml = _detect_watermark_ml(img)
    if ml is not None:
        return ml
    H, W = img.shape[:2]
    top = red.copy()
    top[int(H * 0.30):, :] = 0
    n, _, st, _ = cv2.connectedComponentsWithStats(top)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 150]
    if keep:
        xs = [st[i, cv2.CC_STAT_LEFT] for i in keep] + [st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] for i in keep]
        ys = [st[i, cv2.CC_STAT_TOP] for i in keep] + [st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] for i in keep]
        pad = 22
        return (int(max(0, min(xs) - pad)), int(max(0, min(ys) - pad)),
                int(min(W, max(xs) + pad)), int(min(H, max(ys) + pad)))
    return _match_watermark_topright(img)


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
        # watermark strokes, red OR white). Keep ALL text-sized components so BOTH
        # "Trust" (small script) and "Encar" (bold) are removed — only large solid
        # components (a car panel/roof caught in the box) are dropped, so the vehicle
        # is never inpainted. The box comes from the precise edge detector, so it sits
        # on the top-right watermark; there is no separate windshield element to guard.
        reg = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        bg = int(np.median(reg))
        txt = (np.abs(reg.astype(np.int16) - bg) > 28).astype(np.uint8) * 255
        nn, lab, stt, _ = cv2.connectedComponentsWithStats(txt)
        area_box = reg.shape[0] * reg.shape[1]
        keepm = np.zeros_like(txt)
        for i in range(1, nn):
            a = stt[i, cv2.CC_STAT_AREA]
            if 5 <= a <= 0.30 * area_box:  # all watermark strokes; drop large solids (car)
                keepm[lab == i] = 255
        mask[y0:y1, x0:x1] = cv2.bitwise_or(mask[y0:y1, x0:x1], keepm)
    if plate_box:
        px, py, pw, ph = plate_box
        if full_plate:
            mask[py:py + ph, px:px + pw] = 255
        else:
            rs = cv2.cvtColor(img[py:py + ph, px:px + pw], cv2.COLOR_BGR2HSV)
            white = ((rs[:, :, 1] < 80) & (rs[:, :, 2] > 140)).astype(np.uint8) * 255
            mask[py:py + ph, px:px + pw] = cv2.bitwise_or(mask[py:py + ph, px:px + pw], white)
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)


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
