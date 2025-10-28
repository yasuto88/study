# Dockerfile
FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

ARG DEBIAN_FRONTEND=noninteractive

# --- OS 依存 ---
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    libusb-1.0-0 libgl1 libglib2.0-0 \
    git ca-certificates curl; \
    apt-get clean; rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存だけ先にコピー（キャッシュ効かせる）
COPY realsense/requirements-local.txt /tmp/requirements-local.txt

# venv をイメージに焼き込み（起動後すぐ使える）
RUN set -eux; \
    python3 -m venv /opt/rsenv311; \
    . /opt/rsenv311/bin/activate; \
    pip install --upgrade pip wheel setuptools; \
    pip install -r /tmp/requirements-local.txt; \
    pip install --no-cache-dir pyrealsense2; \
    python -c "import cv2, pyrealsense2 as rs; print('cv2:', cv2.__version__); print('rs:', rs.__version__)"

ENV VIRTUAL_ENV=/opt/rsenv311
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# アプリ本体（開発時はホストをマウントするので軽い）
COPY . /app

CMD ["bash"]
