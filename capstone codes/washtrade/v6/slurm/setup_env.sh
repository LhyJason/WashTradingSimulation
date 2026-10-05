#!/bin/bash
# One-time environment setup on the cluster. Run from the REPO ROOT on a login node:
#   bash "capstone codes/washtrade/v6/slurm/setup_env.sh"
# Creates .venv-cluster with the pinned versions of slurm/requirements-cluster.txt.
set -euo pipefail

# Python 3.12, to match the local environment. Check the exact module name with
#   module avail python
module load python/3.12

python -m venv .venv-cluster
source .venv-cluster/bin/activate
pip install --upgrade pip
# CPU-only torch wheel (no CUDA download); same version as locally
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r "capstone codes/washtrade/v6/slurm/requirements-cluster.txt"
python -c "import numpy, torch, pandas, scipy; print('numpy', numpy.__version__, '| torch', torch.__version__, '| pandas', pandas.__version__, '| scipy', scipy.__version__)"
