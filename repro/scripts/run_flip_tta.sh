#!/usr/bin/env bash
# Flip-filter TTA on the saved features (CPU only, the DiT dummy load is left alone): batch 64, then batch 16.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=experiments/2026-11-17_flip_tta/run.log
mkdir -p "$(dirname "$LOG")"
PY=~/REOBench/.venv/bin/python
{
  CUDA_VISIBLE_DEVICES= $PY -u scripts/analysis/flip_filter_tta.py --bs 64 --lrs 1e-4 1e-3 1e-2 1e-1 --threads 48 && echo "BS64 DONE $(date '+%F %T')" \
    && CUDA_VISIBLE_DEVICES= $PY -u scripts/analysis/flip_filter_tta.py --bs 16 --lrs 1e-4 1e-3 1e-2 1e-1 --threads 48 && echo "ALL DONE $(date '+%F %T')" \
    || echo "FAILED $(date '+%F %T')"
} 2>&1 | grep --line-buffered -v -i "warn" >> "$LOG"
