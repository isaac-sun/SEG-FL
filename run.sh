#!/usr/bin/env bash
# Launcher: checks dependencies (keeps the platform's own PyTorch build),
# prepares the data if missing (fetch_data.sh), then runs main.py.
set -euo pipefail
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

PIP_PKGS="$PROJECT_DIR/pip_pkgs"
export PYTHONPATH="$PIP_PKGS${PYTHONPATH:+:$PYTHONPATH}"
if ! "$PYTHON_BIN" -c "import torch, torchvision, numpy, pandas, scipy, openpyxl, matplotlib, PIL" 2>/dev/null; then
    echo "===== 安装依赖到 pip_pkgs/（保留平台自带 PyTorch） ====="
    "$PYTHON_BIN" -c "import torch, torchvision" || { echo "需要先有 PyTorch 和 torchvision" >&2; exit 1; }
    mkdir -p "$PIP_PKGS"
    "$PYTHON_BIN" -m pip install --requirement requirements-cloud.txt \
        --target "$PIP_PKGS" --no-cache-dir ${PIP_INDEX_URL:+-i "$PIP_INDEX_URL"}
fi
"$PYTHON_BIN" - <<'PY'
import torch, torchvision, numpy, pandas
print(f"torch={torch.__version__} torchvision={torchvision.__version__} "
      f"cuda={torch.cuda.is_available()} numpy={numpy.__version__} pandas={pandas.__version__}")
PY
[ "${1:-}" = "--help" ] && exec "$PYTHON_BIN" main.py --help
bash fetch_data.sh
exec "$PYTHON_BIN" main.py --data-dir "${DATA_DIR:-dataset/voc_seg}" "$@"
