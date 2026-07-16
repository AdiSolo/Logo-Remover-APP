"""
Rebranding API.

Endpoints
---------
GET  /health                      -> liveness + which inpaint methods are ready
POST /rebrand                     -> clean/rebrand an image
        multipart file:  file=@car.jpg
        or JSON/form:    url=https://.../car.webp
     query params:
        method=lama|opencv        (default: lama)
        format=jpg|png            (default: jpg)
        logo=titanic|none         (default: titanic; 'none' = clean only, no logo)
        output=bytes|url          (default: bytes; 'url' hosts the result in
                                   object storage and returns {"url": ...} JSON.
                                   Idempotent per (url,method,format,logo).)

Run:
    .venv-lama/bin/uvicorn api:app --host 0.0.0.0 --port 8000
"""
import hashlib
import os
import cv2
import numpy as np
import urllib.request
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Header, Depends
from fastapi.responses import Response, JSONResponse

import rebrand_core as core
import storage

app = FastAPI(title="Car Image Rebrand API", version="1.0")
LOGO = core.load_logo()

# Bump when the cleaning algorithm changes so hosted-output cache keys refresh
# (otherwise an improved result would collide with the old cached image).
ALGO_VERSION = os.environ.get("REBRAND_ALGO_VERSION", "4")

# Optional API-key auth: set REBRAND_API_KEY in the environment to require it.
API_KEY = os.environ.get("REBRAND_API_KEY")


def require_key(x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.on_event("startup")
def _preload():
    # Warm the LaMa model at boot so the first real request isn't slow.
    if os.environ.get("REBRAND_PRELOAD", "1") == "1":
        try:
            core._get_lama()
        except Exception as e:  # non-fatal; opencv method still works
            print(f"[startup] LaMa preload skipped: {e}")


def _decode(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="could not decode image")
    return img


def _fetch(url: str) -> bytes:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rebrand-api/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not fetch url: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "methods": ["opencv", "lama"], "default": "lama", "hosting": storage.is_configured()}


@app.post("/rebrand")
async def rebrand_endpoint(
    file: UploadFile = File(None),
    url: str = Form(None),
    method: str = Query("lama", pattern="^(lama|opencv)$"),
    format: str = Query("jpg", pattern="^(jpg|png)$"),
    logo: str = Query("titanic", pattern="^(titanic|none)$"),
    output: str = Query("bytes", pattern="^(bytes|url)$"),
    _auth: None = Depends(require_key),
):
    paste_logo = logo != "none"

    # Hosted output: idempotent by (source, params). On a cache hit we skip the
    # expensive inpaint entirely and return the already-stored URL.
    key = None
    if output == "url":
        if not storage.is_configured():
            raise HTTPException(status_code=503, detail="hosting not configured (set STORAGE_* env)")
        if not url:
            raise HTTPException(status_code=400, detail="output=url requires a 'url' input")
        seed = f"{url}|{method}|{format}|{logo}|v{ALGO_VERSION}"
        key = storage.make_key(seed, format)
        if storage.exists(key):
            return JSONResponse({"url": storage.public_url(key), "cached": True})

    if file is not None:
        data = await file.read()
    elif url:
        data = _fetch(url)
    else:
        raise HTTPException(status_code=400, detail="provide a 'file' upload or a 'url'")

    img = _decode(data)
    result, info = core.rebrand(img, LOGO, method=method, paste_logo=paste_logo)

    ext = ".png" if format == "png" else ".jpg"
    params = [] if format == "png" else [cv2.IMWRITE_JPEG_QUALITY, 92]
    ok, buf = cv2.imencode(ext, result, params)
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode result")
    media = "image/png" if format == "png" else "image/jpeg"

    if output == "url":
        public = storage.upload(key, buf.tobytes(), media)
        return JSONResponse({
            "url": public,
            "cached": False,
            "watermark": info["watermark"],
            "plate": info["plate"],
            "logo": info["logo"],
            "method": info["method"],
        })

    headers = {
        "X-Watermark-Replaced": str(info["watermark"]),
        "X-Plate-Replaced": str(info["plate"]),
        "X-Logo-Pasted": str(info["logo"]),
        "X-Method": info["method"],
    }
    return Response(content=buf.tobytes(), media_type=media, headers=headers)


@app.post("/detect")
async def detect_endpoint(file: UploadFile = File(None), url: str = Form(None)):
    """Diagnostic: report what would be detected, without processing."""
    data = await file.read() if file is not None else _fetch(url) if url else None
    if data is None:
        raise HTTPException(status_code=400, detail="provide a 'file' upload or a 'url'")
    img = _decode(data)
    red = core._red_mask(img)
    return JSONResponse({
        "size": {"w": img.shape[1], "h": img.shape[0]},
        "watermark_box": core.detect_watermark_box(img, red),
        "plate_box": core.detect_plate_box(img, red),
    })
