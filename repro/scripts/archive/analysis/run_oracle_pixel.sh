#!/usr/bin/env bash
# Shift-direction oracle (all backbones, AID + RESISC45) then PSNR / MS-SSIM / LPIPS. Stops the DiT dummy load first, restarts it after.
set -uo pipefail
cd "$(dirname "$0")/../../.."
LOG=experiments/2026-11-17_shift_oracle/run.log
mkdir -p "$(dirname "$LOG")"
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
PY=~/REOBench/.venv/bin/python
{
  $PY -u scripts/analysis/shift_direction_oracle.py && echo "ORACLE DONE $(date '+%F %T')" \
    && $PY -u scripts/analysis/pixel_metrics.py && echo "ALL DONE $(date '+%F %T')" || echo "FAILED $(date '+%F %T')"
} 2>&1 | grep --line-buffered -v -i "warn" >> "$LOG"
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
