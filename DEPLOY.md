# Deploying the Rebrand API

Removes the Encar watermark + plate text from car photos and pastes the
TITANIC AUTO logo in their place. Meant to be called by your backend at
**image-ingest time** (when importing from carapis), not on every page view —
LaMa takes several seconds per image on CPU, so rebrand once and store the result.

## What's in the box
- `api.py` — FastAPI service (`/health`, `/rebrand`, `/detect`)
- `rebrand_core.py` — detection + AI inpaint + logo compositing
- `assets/logo.png` — the TITANIC AUTO logo (transparent PNG)
- `requirements.txt`, `Dockerfile`, `.dockerignore`

## Endpoints
| Method | Path | Notes |
|---|---|---|
| GET | `/health` | liveness + `hosting` flag (is object storage configured) |
| POST | `/rebrand` | `file=@img` **or** `url=...`; query `method=lama\|opencv`, `format=jpg\|png`, `logo=titanic\|none`, `output=bytes\|url` |
| POST | `/detect` | diagnostic: returns detected boxes as JSON |

`logo=none` → clean only (inpaint the marks, paste no logo — for neutral/white-label catalogs).
`output=url` → host the result in object storage and return `{"url": ...}` JSON instead of image bytes.
It is **idempotent** per `(url, method, format, logo)`: a repeat call returns the stored URL (`"cached": true`) without re-running LaMa.

Response headers on `/rebrand` (bytes mode): `X-Watermark-Replaced`, `X-Plate-Replaced`, `X-Logo-Pasted`, `X-Method`.

## Config (environment variables)
| Var | Default | Purpose |
|---|---|---|
| `REBRAND_API_KEY` | *(unset)* | If set, callers must send header `X-API-Key: <key>`. Leave unset for internal-only networks. |
| `REBRAND_DEVICE` | `cpu` | `cpu`, `cuda` (NVIDIA GPU), or `mps` (Apple Silicon). GPU drops LaMa from ~8s to <1s. |
| `REBRAND_PRELOAD` | `1` | Warm the model at startup so the first request isn't slow. |

### Object storage (for `output=url`)
S3-compatible; same code works with Cloudflare R2 or AWS S3. Leave unset to disable hosting (API then only returns bytes).

| Var | Example | Purpose |
|---|---|---|
| `STORAGE_ENDPOINT` | `https://<acct>.r2.cloudflarestorage.com` | R2/S3 endpoint (omit for AWS S3) |
| `STORAGE_REGION` | `auto` | `auto` for R2; a real region for S3 |
| `STORAGE_ACCESS_KEY_ID` | | access key |
| `STORAGE_SECRET_ACCESS_KEY` | | secret key |
| `STORAGE_BUCKET` | `encar-clean` | bucket name |
| `STORAGE_PUBLIC_URL` | `https://cdn.yourdomain.com` | public read base (bucket public URL or CDN) |
| `STORAGE_PREFIX` | `clean` | key prefix (optional) |

## Run with Docker (recommended)
```bash
docker build -t rebrand-api .          # bakes the LaMa model into the image
docker run -p 8000:8000 \
  -e REBRAND_API_KEY=change-me \
  rebrand-api
```

## Run without Docker
```bash
python3.12 -m venv .venv-lama
.venv-lama/bin/pip install -r requirements.txt
.venv-lama/bin/iopaint download --model lama   # one-time model download
REBRAND_API_KEY=change-me .venv-lama/bin/uvicorn api:app --host 0.0.0.0 --port 8000
```

## Calling it from your backend (example)
```bash
curl -X POST "https://rebrand.internal/rebrand?method=lama&format=jpg" \
     -H "X-API-Key: change-me" \
     -F "url=https://api.carapis.com/media/vehicles/.../image.webp" \
     -o rebranded.jpg
```

## Scaling notes
- Use **one worker per container** (each worker loads its own model copy). Scale
  horizontally with more replicas behind a load balancer, not with `--workers N`.
- CPU is fine for ingest-time batch processing. Use a GPU only if you need
  near-real-time rebranding.
- `opencv` method (~0.25s) is a fast fallback for simple/plain backgrounds.
