FROM python:3.12-slim

# System libs OpenCV needs at runtime (headless build still needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the LaMa model into the image so the container needs no download at boot.
# (~196MB) Cache dir matches iopaint's default.
RUN python -c "from iopaint.download import cli_download_model; cli_download_model('lama')" || \
    (mkdir -p /root/.cache/torch/hub/checkpoints && \
     curl -L -o /root/.cache/torch/hub/checkpoints/big-lama.pt \
       https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt)

COPY rebrand_core.py api.py ./
COPY assets ./assets

ENV REBRAND_DEVICE=cpu \
    REBRAND_PRELOAD=1

EXPOSE 8000
# Single worker keeps one model copy in memory. Scale with more replicas, not workers.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
