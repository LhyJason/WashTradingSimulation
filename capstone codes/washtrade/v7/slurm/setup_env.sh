#!/bin/bash
# One-time environment setup on the cluster. Run from the REPO ROOT on a login node:
#   PY_MODULE=python/3.12.1 bash "capstone codes/washtrade/v7/slurm/setup_env.sh"
# Creates .venv-cluster with the pinned versions of slurm/requirements-cluster.txt.
# PY_MODULE: a Python 3.12 module from `module avail python` (default python/3.12). Its name is
# saved in .venv-cluster/PY_MODULE, and the job scripts load the same module before activating.
set -euo pipefail

PY_MODULE="${PY_MODULE:-python/3.12}"
set +u    # Lmod is not safe under set -u
module load "$PY_MODULE"
set -u
python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version" \
    || { echo "need Python 3.12, got $(python --version)"; exit 1; }

python -m venv .venv-cluster
echo "$PY_MODULE" > .venv-cluster/PY_MODULE
source .venv-cluster/bin/activate
pip install --upgrade pip
# CPU-only torch wheel (no CUDA download); same version as locally
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r "capstone codes/washtrade/v7/slurm/requirements-cluster.txt"
python -c "import numpy, torch, pandas, scipy, tqdm; print('numpy', numpy.__version__, '| torch', torch.__version__, '| pandas', pandas.__version__, '| scipy', scipy.__version__, '| tqdm', tqdm.__version__)"
