#!/usr/bin/env bash
set -euo pipefail

# One-command setup script for the neurips-bci Conda environment.
# Pass --prefix /path/to/env to install into a custom directory (for example on cluster pool storage),
# or omit --prefix to install as a named Conda environment `neurips-bci`.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="neurips-bci"
ENV_PREFIX=""
TORCH_CHANNEL="${TORCH_CHANNEL:-cu130}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      ENV_PREFIX="$2"
      shift 2
      ;;
    --name)
      ENV_NAME="$2"
      shift 2
      ;;
    --cpu)
      TORCH_CHANNEL="cpu"
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh || true
    module load miniforge 2>/dev/null || true
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH. Install Miniforge/Conda or load your cluster miniforge module first." >&2
  exit 1
fi

if [[ -n "${ENV_PREFIX}" ]]; then
  conda create -y --prefix "${ENV_PREFIX}" -c conda-forge python=3.12 pip uv
  PYTHON_BIN="${ENV_PREFIX}/bin/python"
  UV_BIN="${ENV_PREFIX}/bin/uv"
else
  conda create -y -n "${ENV_NAME}" -c conda-forge python=3.12 pip uv
  CONDA_BASE="$(conda info --base)"
  PYTHON_BIN="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
  UV_BIN="${CONDA_BASE}/envs/${ENV_NAME}/bin/uv"
fi

"${UV_BIN}" pip install --python "${PYTHON_BIN}" \
  --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}" \
  torch==2.14.1 torchvision==0.29.1 torchaudio==2.11.0

"${UV_BIN}" pip install --python "${PYTHON_BIN}" \
  --extra-index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}" \
  --index-strategy unsafe-best-match \
  -r "${REPO_ROOT}/requirements-lock.txt"

"${UV_BIN}" pip install --python "${PYTHON_BIN}" --no-deps -e "${REPO_ROOT}"

"${PYTHON_BIN}" -c "import neurips_bci, torch; print('neurips-bci environment ready | PyTorch:', torch.__version__)"
