#!/usr/bin/env bash
# Weighted Flips reproduction on ImageNet-C (96 splits, ResNet-50). Stops the DiT dummy load first, restarts it after.
set -uo pipefail
cd "$(dirname "$0")/../../.."
LOG=experiments/2026-11-19_weighted_flips_repro/run.log
mkdir -p "$(dirname "$LOG")"
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
PY=~/REOBench/.venv/bin/python
{
  $PY -u scripts/verify/weighted_flips_repro.py && echo "ALL DONE $(date '+%F %T')" || echo "FAILED $(date '+%F %T')"
} 2>&1 | grep --line-buffered -v -i "warn" >> "$LOG"
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
