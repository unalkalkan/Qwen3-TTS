# Experimental Dockerfile for newer GPUs / newer CUDA toolchains
#
# Target: RTX 5090 (Blackwell-class). Your driver reports CUDA 13.0 capability.
#
# IMPORTANT REALITY CHECK (as of 2026-01):
# - "torch==2.9.0+cu130" is not a commonly published PyTorch wheel version.
# - CUDA 13 base images / PyTorch cu130 wheels may or may not exist depending on release timing.
#
# This Dockerfile is therefore written to be EASY TO ADAPT:
# - If NVIDIA publishes nvidia/cuda:13.0.* images, set CUDA_BASE accordingly.
# - If PyTorch publishes cu130 wheels, set TORCH_INDEX_URL + TORCH_VERSION accordingly.
#
# Keep the existing Dockerfile (cu118) for your GTX 1050 Ti; use this one for RTX 5090.

# ---- Choose a CUDA runtime base image ------------------------------------------------
# If you have a CUDA 13.0 runtime image available, use it:
#   nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04
# If not, fall back to the newest available CUDA runtime you can pull (e.g. 12.8/12.6/12.4).
ARG CUDA_BASE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
FROM ${CUDA_BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REQUIRE_CUDA=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=0

# System deps (audio + build toolchain for flash-attn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    sox \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- PyTorch install ---------------------------------------------------------------
# For CUDA 13, you WANT to use the correct wheel index URL.
# Example pattern for PyTorch wheel indexes:
#   https://download.pytorch.org/whl/cu124
#   https://download.pytorch.org/whl/cu121
# If/when cu130 exists, it would likely be:
#   https://download.pytorch.org/whl/cu130
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_VERSION=2.5.1
ARG TORCHAUDIO_VERSION=2.5.1

RUN python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install \
    --index-url ${TORCH_INDEX_URL} \
    torch==${TORCH_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION}

# Python deps (from pyproject.toml)
COPY pyproject.toml /app/pyproject.toml
RUN python3 -m pip install -e /app

# ---- FlashAttention ---------------------------------------------------------------
# FlashAttention is optional but recommended on modern GPUs.
# It often requires building from source; this can take a while.
# If it fails, you can comment it out and the server will still work.
#
# You may need to pin flash-attn to a version compatible with your torch + cuda.
RUN python3 -m pip install --no-build-isolation flash-attn || true

# App code
COPY . /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/v1/models >/dev/null || exit 1

CMD ["uvicorn", "openai_server:app", "--host", "0.0.0.0", "--port", "8000"]
