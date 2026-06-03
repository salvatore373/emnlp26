# ===========================================================================
# Stage 1: Build llama-server from source
# ---------------------------------------------------------------------------
# We need CUDA 13.0 + the full CUDA toolkit (for nvcc) to compile llama.cpp
# with MMQ kernels for Blackwell (RTX 5090 / sm_120).
# This stage is ~4GB but is DISCARDED after build — only the binary is kept.
# ===========================================================================
FROM nvidia/cuda:13.0.3-devel-ubuntu22.04 AS llama-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake ninja-build \
    gcc g++ build-essential \
    pkg-config libssl-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Pin to a specific llama.cpp release tag for reproducibility.
# Check https://github.com/ggml-org/llama.cpp/releases for the latest tag.
ARG LLAMA_VERSION=b9550

RUN git clone --depth 1 --branch ${LLAMA_VERSION} \
    https://github.com/ggml-org/llama.cpp /llama.cpp

RUN cmake /llama.cpp -B /llama.cpp/build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="86;120" \
    -DGGML_CURL=ON \
    -DLLAMA_OPENSSL=ON \
    -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    && cmake --build /llama.cpp/build \
             --config Release \
             --target llama-server \
             -j$(nproc)

# Gather all runtime `.so` dependencies into a single directory for easy copying.
# This prevents path-guessing between /usr/local/cuda/lib64 and /usr/lib/x86_64-linux-gnu/
RUN mkdir -p /export-libs && \
    cp /usr/local/cuda/lib64/libcudart.so.13 /export-libs/ && \
    cp /usr/local/cuda/lib64/libcublas.so.13 /export-libs/ && \
    cp /usr/local/cuda/lib64/libcublasLt.so.13 /export-libs/ && \
    find /usr -type f -name "libnccl.so.2*" -exec cp {} /export-libs/ \; -quit


# ===========================================================================
# Stage 2: Runtime image
# ---------------------------------------------------------------------------
# Starts from the PyTorch runtime base.
# We install only the minimal CUDA compiler tools needed by vLLM's JIT kernel
# compilation (nvcc + dev headers), without pulling in the full devel image.
# ===========================================================================
FROM ubuntu:22.04

LABEL maintainer="salvatorerago"
LABEL org.opencontainers.image.authors="salvatorerago"
LABEL project="DoToolFailuresHelp"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HOME=/root/.cache/huggingface
# Force llama.cpp's MMQ kernel path at runtime (replaces the non-existent cmake flag).
# MMQ is the fast quantized matmul path on Blackwell; without this it may fall back
# to slower cuBLAS depending on the operation.
ENV GGML_CUDA_FORCE_MMQ=1

# Install system dependencies in a single layer, clean up apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    ca-certificates git wget \
    libffi-dev libssl-dev \
    ffmpeg \
    gcc g++ build-essential ninja-build \
    libreoffice poppler-utils \
    libcurl4 \
    && rm -rf /var/lib/apt/lists/*

# ── Step 2: add NVIDIA CUDA apt repo, then install nvcc + dev headers ─────
# The pytorch runtime base does not include the NVIDIA package repo,
# so we register it via the official cuda-keyring before installing
# the compiler tools that vLLM needs for JIT kernel compilation.
RUN wget -qO /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
    cuda-nvcc-13-0 \
    cuda-cudart-dev-13-0 \
    libcublas-dev-13-0 \
    libcurand-dev-13-0 \
    && rm -rf /var/lib/apt/lists/*

# Expose nvcc and CUDA headers to vLLM and any other CUDA-dependent tool
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
# ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/targets/x86_64-linux/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Copy the compiled binary and shared libraries from llama-builder
COPY --from=llama-builder /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=llama-builder /export-libs/* /usr/local/lib/
RUN ldconfig

# Install uv for blazing fast dependency resolution and downloading
RUN pip3 install --no-cache-dir uv

WORKDIR /workspace

# Copy ONLY requirements first (critical for caching)
COPY requirements.txt .

# Use BuildKit cache for uv downloads (prevents torch re-download and speeds up resolution)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --prerelease=allow -r requirements.txt

# ── vLLM isolated environment ──────────────────────────────────────────────
# packaging>=24.2 (required by flashinfer-python 0.6.11) needs Python 3.11+.
# Ubuntu 22.04 ships Python 3.10, so we pull 3.12 via uv's Python manager.
RUN uv python install 3.12
RUN uv venv /opt/vllm_env --python 3.12
# 1. Strictly install the CUDA 13.0 PyTorch backend FIRST.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/vllm_env \
    --index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"

# 2. Pre-install ALL build dependencies required for compiling without isolation
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/vllm_env \
    setuptools setuptools-rust setuptools_scm wheel ninja packaging cmake

# 3. Force vLLM to compile from source (--no-binary) against the installed Torch
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/vllm_env \
    --no-build-isolation \
    transformers==5.6.2 vllm==0.22.0

# Persist HuggingFace model cache across container runs
VOLUME ["/root/.cache/huggingface"]

# Copy rest of project (does NOT invalidate pip layer)
# COPY . .

CMD ["python3", "main.py"]