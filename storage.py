"""
Storage for hosted rebrand output. Two backends:

  local  — write files to a directory on disk, served by nginx/Caddy. Best for a
           single VPS (persistent disk); no extra service, no egress fees.
  s3     — S3-compatible object storage (Cloudflare R2, AWS S3, …). Best when the
           API scales to multiple replicas or you want a CDN in front.

Backend is chosen by STORAGE_BACKEND, or auto-detected: if STORAGE_LOCAL_DIR is set
→ local; else if S3 creds are set → s3; else hosting is disabled (API returns bytes).

Common:
    STORAGE_PUBLIC_URL   public read base, e.g. https://rebrand.yourdomain.com/img
    STORAGE_PREFIX       key prefix (default "clean")

Local backend:
    STORAGE_LOCAL_DIR    directory nginx serves at STORAGE_PUBLIC_URL, e.g. /srv/clean

S3 backend:
    STORAGE_ENDPOINT          e.g. https://<account>.r2.cloudflarestorage.com (omit for AWS S3)
    STORAGE_REGION            "auto" for R2, e.g. "eu-central-1" for S3
    STORAGE_ACCESS_KEY_ID
    STORAGE_SECRET_ACCESS_KEY
    STORAGE_BUCKET
"""
import hashlib
import os

_client = None


def _cfg():
    return {
        "backend": (os.environ.get("STORAGE_BACKEND") or "").strip().lower() or None,
        "local_dir": os.environ.get("STORAGE_LOCAL_DIR") or None,
        "endpoint": os.environ.get("STORAGE_ENDPOINT") or None,
        "region": os.environ.get("STORAGE_REGION", "auto"),
        "key_id": os.environ.get("STORAGE_ACCESS_KEY_ID"),
        "secret": os.environ.get("STORAGE_SECRET_ACCESS_KEY"),
        "bucket": os.environ.get("STORAGE_BUCKET"),
        "public_url": (os.environ.get("STORAGE_PUBLIC_URL") or "").rstrip("/"),
        "prefix": os.environ.get("STORAGE_PREFIX", "clean").strip("/"),
    }


def _backend(c=None) -> str | None:
    """Resolve the active backend: explicit STORAGE_BACKEND, else auto-detect."""
    c = c or _cfg()
    if c["backend"] in ("local", "s3"):
        return c["backend"]
    if c["local_dir"]:
        return "local"
    if c["key_id"] and c["secret"] and c["bucket"]:
        return "s3"
    return None


def is_configured() -> bool:
    c = _cfg()
    b = _backend(c)
    if not c["public_url"]:
        return False
    if b == "local":
        return bool(c["local_dir"])
    if b == "s3":
        return bool(c["key_id"] and c["secret"] and c["bucket"])
    return False


def _get_client():
    global _client
    if _client is None:
        import boto3  # imported lazily so the API runs without boto3 installed/configured

        c = _cfg()
        _client = boto3.client(
            "s3",
            endpoint_url=c["endpoint"],
            region_name=c["region"],
            aws_access_key_id=c["key_id"],
            aws_secret_access_key=c["secret"],
        )
    return _client


def make_key(seed: str, ext: str) -> str:
    """Deterministic object key from a seed (source URL or content hash) → idempotency."""
    c = _cfg()
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    ext = ext.lstrip(".")
    return f"{c['prefix']}/{digest}.{ext}"


def public_url(key: str) -> str:
    return f"{_cfg()['public_url']}/{key}"


def exists(key: str) -> bool:
    c = _cfg()
    if _backend(c) == "local":
        return os.path.exists(os.path.join(c["local_dir"], key))
    try:
        _get_client().head_object(Bucket=c["bucket"], Key=key)
        return True
    except Exception:
        return False


def upload(key: str, data: bytes, content_type: str) -> str:
    """Store bytes at key, return the public URL."""
    c = _cfg()
    if _backend(c) == "local":
        dest = os.path.join(c["local_dir"], key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        return public_url(key)
    _get_client().put_object(
        Bucket=c["bucket"],
        Key=key,
        Body=data,
        ContentType=content_type,
        CacheControl="public, max-age=31536000, immutable",
    )
    return public_url(key)
