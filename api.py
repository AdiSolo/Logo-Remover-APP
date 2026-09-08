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
POST /video                       -> start a reel render job (async)
        JSON: {"photos": [url, ...], "fields": {brand, model, trim, year,
               mileage, price, fuel, transmission, body, site_url}}
        -> 202 {"job_id": ...}. Requires hosting (STORAGE_*) configured.
GET  /video/{job_id}              -> poll a reel render job
        -> {"status": "queued"|"processing"|"done"|"error", "url": ..., "error": ...}
        -> 404 if unknown (e.g. the container restarted mid-render — retry)

Run:
    .venv-lama/bin/uvicorn api:app --host 0.0.0.0 --port 8000
"""
import hashlib
import os
import threading
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Optional, TypedDict

import cv2
import numpy as np
import urllib.request
from fastapi import BackgroundTasks, FastAPI, UploadFile, File, Form, Query, HTTPException, Header, Depends
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

import rebrand_core as core
import video_core
import storage

app = FastAPI(title="Car Image Rebrand API", version="1.0")
LOGO = core.load_logo()

# Bump when the cleaning algorithm changes so hosted-output cache keys refresh
# (otherwise an improved result would collide with the old cached image).
ALGO_VERSION = os.environ.get("REBRAND_ALGO_VERSION", "16")  # v16: despeckle pass removes LaMa's residual multicoloured confetti noise on flat backgrounds

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


# ---------------------------------------------------------------------------
# Reel generation — async job. No queue infra exists in this service (every
# other endpoint is synchronous request/response); a render takes tens of
# seconds, far too slow for one HTTP request. Given this is a low-volume,
# single-admin, on-demand feature — not a batch job — an in-memory job store
# is the simplest thing that satisfies the requirements: no new infra (no
# Redis/Celery/SQLite), and losing an in-flight job on a container restart is
# an accepted tradeoff (the caller just retries; see GET /video/{job_id}'s 404).
# ---------------------------------------------------------------------------

class VideoJob(TypedDict):
    status: Literal["queued", "processing", "done", "error"]
    url: Optional[str]
    error: Optional[str]
    created_at: float


class VideoFields(BaseModel):
    brand: str
    model: str
    trim: Optional[str] = None
    year: Optional[int] = None
    mileage: Optional[str] = None
    price: Optional[str] = None
    fuel: Optional[str] = None
    transmission: Optional[str] = None
    body: Optional[str] = None
    site_url: Optional[str] = None


class VideoRequest(BaseModel):
    photos: list[str]
    fields: VideoFields


_VIDEO_JOBS: dict[str, VideoJob] = {}
_VIDEO_JOBS_LOCK = threading.Lock()
# Serializes reel renders against each other on this CPU-only box (does not
# also guard against a concurrent /rebrand LaMa call — accepted v1 tradeoff).
_VIDEO_SEMAPHORE = threading.Semaphore(1)
_VIDEO_JOB_TTL_SECONDS = 2 * 60 * 60


def _prune_video_jobs() -> None:
    cutoff = time.time() - _VIDEO_JOB_TTL_SECONDS
    with _VIDEO_JOBS_LOCK:
        stale = [jid for jid, job in _VIDEO_JOBS.items() if job["created_at"] < cutoff]
        for jid in stale:
            del _VIDEO_JOBS[jid]


def _run_video_job(job_id: str, photos: list[str], fields: dict) -> None:
    """Runs in a background thread (Starlette's run_in_threadpool via
    BackgroundTasks — this MUST stay a plain `def`, not `async def`, or it
    would run inline on the event loop and block every other request, same
    as /rebrand's already-synchronous behaviour)."""
    with _VIDEO_JOBS_LOCK:
        _VIDEO_JOBS[job_id]["status"] = "processing"
    try:
        with _VIDEO_SEMAPHORE:
            with TemporaryDirectory(prefix=f"reel-{job_id}-") as tmp:
                out_path = video_core.render_reel(photos, fields, Path(tmp))
                data = out_path.read_bytes()
                key = storage.make_key(f"{job_id}|v{video_core.VIDEO_ALGO_VERSION}", "mp4")
                url = storage.upload(key, data, "video/mp4")
        with _VIDEO_JOBS_LOCK:
            _VIDEO_JOBS[job_id]["status"] = "done"
            _VIDEO_JOBS[job_id]["url"] = url
    except Exception as e:
        with _VIDEO_JOBS_LOCK:
            _VIDEO_JOBS[job_id]["status"] = "error"
            _VIDEO_JOBS[job_id]["error"] = str(e)[-2000:]


@app.post("/video", status_code=202)
async def start_video_endpoint(
    payload: VideoRequest,
    background_tasks: BackgroundTasks,
    _auth: None = Depends(require_key),
):
    if not storage.is_configured():
        raise HTTPException(status_code=503, detail="hosting not configured (set STORAGE_* env)")
    if len(payload.photos) < video_core.MIN_PHOTOS:
        raise HTTPException(status_code=400, detail=f"at least {video_core.MIN_PHOTOS} photos are required")

    _prune_video_jobs()
    job_id = uuid.uuid4().hex
    with _VIDEO_JOBS_LOCK:
        _VIDEO_JOBS[job_id] = {"status": "queued", "url": None, "error": None, "created_at": time.time()}
    background_tasks.add_task(_run_video_job, job_id, payload.photos, payload.fields.model_dump())
    return {"job_id": job_id}


@app.get("/video/{job_id}")
def video_status_endpoint(job_id: str, _auth: None = Depends(require_key)):
    with _VIDEO_JOBS_LOCK:
        job = _VIDEO_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job_id (lost on restart, or never existed)")
    return {"status": job["status"], "url": job["url"], "error": job["error"]}
