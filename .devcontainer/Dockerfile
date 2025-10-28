# 最小ベース（slim系より mcr base の方が安定しがち）
FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_MIRROR=http://archive.ubuntu.com/ubuntu

# ---- APT：必要最低限 + キャッシュ削減 ----
RUN set -eux; \
    sed -i "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g" /etc/apt/sources.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    libusb-1.0-0 libgl1 libglib2.0-0 \
    git ca-certificates curl; \
    apt-get clean; rm -rf /var/lib/apt/lists/*

# ---- ワークスペース配下の realsense に venv を作って依存を焼き込む ----
# 先に requirements をコピーしてレイヤー分割・キャッシュ効くように
WORKDIR /workspaces/study
COPY realsense/requirements-local.txt /tmp/requirements-local.txt

# venv 作成（/workspaces/study/realsense/rsenv311）
RUN set -eux; \
    python3 -m venv /workspaces/study/realsense/rsenv311; \
    . /workspaces/study/realsense/rsenv311/bin/activate; \
    pip install --upgrade pip wheel setuptools; \
    # OpenCVはheadlessで固定
    pip install -r /tmp/requirements-local.txt; \
    # RealSense Python バインディング
    pip install --no-cache-dir pyrealsense2; \
    # 動作確認（import だけ）
    python - <<'PY'\nimport cv2, pyrealsense2 as rs; print('cv2 OK:', cv2.__version__); print('rs OK:', rs.__version__)\nPY

# デフォルトシェル環境で venv が使えるようにしておく（親切設定）
RUN echo 'source /workspaces/study/realsense/rsenv311/bin/activate' >> /etc/bash.bashrc
