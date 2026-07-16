# Deploying the rebrand API on Hostinger — agent runbook

Audience: an automation agent (or engineer) standing this service up on Hostinger.
Follow the **decisions** top-to-bottom, then run the **steps** for the chosen path.
This service is the `encar-logo-remover` FastAPI app (LaMa inpainting) that
EncarTrade calls at ingest to strip Encar watermarks/plates off car photos.

Ground truth about the workload (drives every decision below):
- Heavy Python ML deps (PyTorch + iopaint/LaMa). Needs a real Linux box + Docker.
- Long-lived server (uvicorn). NOT a drop-in-a-folder website.
- LaMa on CPU ≈ 8s/image; used at **ingest time, result stored** — never per page view.
- Called by a backend over HTTPS with an `X-API-Key`.

---

## Decision 1 — Which Hostinger product?

| If… | Choose | Why |
|---|---|---|
| (default) | **VPS hosting (KVM plan)** | Full root, install Docker, run uvicorn. The only viable option. |
| You only have Shared/Cloud/Web hosting | ❌ Stop | No long-running Python, no PyTorch, too little RAM. Upgrade to a VPS. |

**Rule: this tool requires a Hostinger VPS (KVM). Do not attempt shared/cloud hosting.**

## Decision 2 — Which VPS size?

Pick by RAM (the binding constraint — PyTorch + model):

| Plan RAM | Verdict |
|---|---|
| 4 GB (KVM 1) | Minimum. Works for a slow one-time backfill; may be tight. |
| **8 GB (KVM 2)** | **Recommended.** Comfortable headroom for LaMa + concurrency. |
| 16 GB+ | Only if running several workers/replicas. |

- CPU-only (Hostinger VPS have **no GPU**). Acceptable: backfill runs in the
  background. Do not expect real-time. `opencv` method (~0.25s) is the fast fallback.
- Disk: any KVM plan (≥50 GB) is plenty (torch install + model ≈ a few GB, plus
  cleaned images if using local storage — see Decision 3).
- OS template: **Ubuntu 22.04 or 24.04** (or "Ubuntu + Docker" template if offered).

## Decision 3 — Where do cleaned images live?

| If… | Choose | Set |
|---|---|---|
| (default) single VPS | **Local disk**, served by the reverse proxy | `STORAGE_LOCAL_DIR` + `STORAGE_PUBLIC_URL` |
| Multiple replicas / want a CDN / durability | **Cloudflare R2** (S3-compatible) | `STORAGE_*` S3 vars |

Local disk is simplest here: the VPS has persistent disk, so no extra service and
no egress fees. Switch to R2 later by only changing env — no code change.

---

## Steps

### 0. Provision
1. Buy the chosen KVM VPS, OS = Ubuntu 24.04.
2. In DNS, point a subdomain at the VPS IP: `rebrand.YOURDOMAIN.com  A  <vps-ip>`.
3. SSH in: `ssh root@<vps-ip>`.

### 1. Base setup + firewall
```bash
apt update && apt -y upgrade
apt -y install ufw git
ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw --force enable
```

### 2. Install Docker
```bash
curl -fsSL https://get.docker.com | sh
docker --version
```

### 3. Get the code onto the VPS
Option A — git (if the repo is pushed somewhere):
```bash
git clone <your-repo-url> /opt/rebrand && cd /opt/rebrand
```
Option B — copy from your machine:
```bash
# run locally, from the encar-logo-remover folder:
rsync -av --exclude '.venv*' --exclude '__pycache__' --exclude 'images_*' \
  ./ root@<vps-ip>:/opt/rebrand/
```

### 4. Build the image (bakes the LaMa model in)
```bash
cd /opt/rebrand
docker build -t rebrand-api .
```

### 5. Run the container

Pick ONE env block from Decision 3.

**Local-disk storage (default):**
```bash
mkdir -p /srv/clean
docker run -d --name rebrand --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -v /srv/clean:/srv/clean \
  -e REBRAND_API_KEY='CHOOSE-A-LONG-RANDOM-KEY' \
  -e REBRAND_DEVICE=cpu \
  -e STORAGE_BACKEND=local \
  -e STORAGE_LOCAL_DIR=/srv/clean \
  -e STORAGE_PUBLIC_URL='https://rebrand.YOURDOMAIN.com/img' \
  rebrand-api
```

**R2 storage (alternative):**
```bash
docker run -d --name rebrand --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -e REBRAND_API_KEY='CHOOSE-A-LONG-RANDOM-KEY' \
  -e REBRAND_DEVICE=cpu \
  -e STORAGE_BACKEND=s3 \
  -e STORAGE_ENDPOINT='https://<acct>.r2.cloudflarestorage.com' \
  -e STORAGE_REGION=auto \
  -e STORAGE_ACCESS_KEY_ID='...' \
  -e STORAGE_SECRET_ACCESS_KEY='...' \
  -e STORAGE_BUCKET='encar-clean' \
  -e STORAGE_PUBLIC_URL='https://<your-r2-public-domain>' \
  rebrand-api
```
Bind to `127.0.0.1:8000` so only the reverse proxy (next step) can reach it.

### 6. Reverse proxy + automatic HTTPS (Caddy — simplest)
```bash
apt -y install debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
apt update && apt -y install caddy
```
Write `/etc/caddy/Caddyfile`:
```
rebrand.YOURDOMAIN.com {
    # API
    reverse_proxy 127.0.0.1:8000

    # Local-disk images only: serve /srv/clean at /img/* (skip if using R2)
    handle_path /img/* {
        root * /srv/clean
        file_server
    }
}
```
```bash
systemctl reload caddy   # Caddy fetches a Let's Encrypt cert automatically
```
> If using R2, delete the `handle_path /img/*` block — images are served by R2.

### 7. Verify
```bash
curl https://rebrand.YOURDOMAIN.com/health
# → {"status":"ok", ... ,"hosting":true}   ← hosting MUST be true

curl -X POST "https://rebrand.YOURDOMAIN.com/rebrand?logo=none&output=url" \
  -H "X-API-Key: CHOOSE-A-LONG-RANDOM-KEY" \
  -F "url=https://ci.encar.com/<some-real-encar-photo>.jpg"
# → {"url":"https://rebrand.YOURDOMAIN.com/img/clean/<hash>.jpg","cached":false,...}
# Open that URL — the watermark/plate should be gone, no logo pasted.
```
`hosting:false` or a 503 on `output=url` ⇒ storage env is wrong (recheck Decision 3 vars).

### 8. Wire into EncarTrade
In EncarTrade admin → **Chei API → Imagini**:
- `LOGO_TOOL_URL` = `https://rebrand.YOURDOMAIN.com/rebrand`
- `LOGO_TOOL_KEY` = the `REBRAND_API_KEY` you chose
Then run the backfill from the EncarTrade repo:
```bash
npx tsx --env-file=.env --env-file=.env.local scripts/clean-photos.ts 50   # test on 50 first
```

---

## Ops notes
- **One worker per container.** Each worker loads its own model copy. Scale with more
  containers behind Caddy, not `--workers N`.
- **Model preload** is on by default (`REBRAND_PRELOAD=1`) so the first request isn't slow.
- **Updating:** `cd /opt/rebrand && git pull` (or rsync) → `docker build -t rebrand-api .`
  → `docker rm -f rebrand` → re-run the `docker run` from step 5.
- **Logs:** `docker logs -f rebrand`.
- **Idempotent:** re-running the backfill is cheap — the tool returns the stored URL
  (`"cached":true`) without re-running LaMa.
- **Fail-safe:** if this service is down, EncarTrade keeps the original Encar photos.
- **Speed:** if the backfill is too slow, add `method=opencv` on the EncarTrade adapter
  for plain backgrounds, or move to a GPU host later (env `REBRAND_DEVICE=cuda`).
