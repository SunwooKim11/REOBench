#!/usr/bin/env bash
# Flip-filter TTA on the saved features: batch 64, then batch 16. Small tensors; runs on the GPU next to other jobs
# (this machine is limited to 12 CPU cores, so a CPU run starves the data loaders of GPU jobs).
set -uo pipefail
cd "$(dirname "$0")/../../.."
LOG=experiments/2026-11-17_flip_tta/run.log
mkdir -p "$(dirname "$LOG")"
PY=~/REOBench/.venv/bin/python
{
  $PY -u scripts/analysis/feat_tta.py --bs 64 --lrs 1e-4 1e-3 1e-2 1e-1 --device cuda --threads 2 && echo "BS64 DONE $(date '+%F %T')" \
    && $PY -u scripts/analysis/feat_tta.py --bs 16 --lrs 1e-4 1e-3 1e-2 1e-1 --device cuda --threads 2 && echo "ALL DONE $(date '+%F %T')" \
    || echo "FAILED $(date '+%F %T')"
} 2>&1 | grep --line-buffered -v -i "warn" >> "$LOG"
