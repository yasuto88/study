#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
ENV_DIR="$SCRIPT_DIR/../rsenv311"
# 優先順で自動検出（環境変数 PYTHON があればそれを優先）
CANDIDATES=(${PYTHON:-} python3.12 python3.11 python3.10 python3 python)
PYTHON_CMD=""

for c in "${CANDIDATES[@]}"; do
  if [ -n "$c" ] && command -v "$c" >/dev/null 2>&1; then
    PYTHON_CMD="$c"
    break
  fi
done

if [ -z "$PYTHON_CMD" ]; then
  echo "❌ Python が見つかりません。python3.10/3.11 などをインストールしてください。"
  exit 1
fi

echo "Using interpreter: $PYTHON_CMD"
echo "Target env dir: $(python3 - <<'PY'
import os,sys;print(os.path.abspath(sys.argv[1]))
PY
"$ENV_DIR"
)"

if [ -d "$ENV_DIR" ]; then
  echo "Environment directory already exists: $ENV_DIR"
  read -r -p "Recreate it? [y/N] " ans
  if [[ "$ans" =~ ^[Yy]$ ]]; then rm -rf "$ENV_DIR"; else exit 0; fi
fi

echo "Creating virtual environment..."
"$PYTHON_CMD" -m venv "$ENV_DIR"

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

echo "Upgrading pip, wheel, setuptools..."
pip install --upgrade pip wheel setuptools

REQ_LOCAL="$SCRIPT_DIR/../requirements-local.txt"
REQ_PROJ="$SCRIPT_DIR/../librealsense/wrappers/python/requirements.txt"

if [ -f "$REQ_LOCAL" ]; then
  echo "Installing from requirements-local.txt"
  pip install -r "$REQ_LOCAL"
elif [ -f "$REQ_PROJ" ]; then
  echo "Installing from librealsense/wrappers/python/requirements.txt"
  pip install -r "$REQ_PROJ"
else
  echo "No requirements file found; installing common packages: numpy, opencv-python, torch (cpu)"
  pip install numpy opencv-python torch --extra-index-url https://download.pytorch.org/whl/cpu
fi

cat <<EOF

✅ Done.
To activate:
  source "$ENV_DIR/bin/activate"

ℹ️  pyrealsense2 は環境依存です。必要なら別途 wheel を入れるかビルドしてください。
EOF
