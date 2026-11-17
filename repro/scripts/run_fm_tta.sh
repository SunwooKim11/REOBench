#!/usr/bin/env bash
# Full-backbone TTA of the 17 stage-1 backbones (GPU, long). Order: AID-C protocol lr -> AID-C lr x10 (tent / filter
# variants) -> RESISC45-C protocol lr. Every (dataset, model, method) is saved on its own, so the run can be stopped and
# resumed at any point. Stops the DiT dummy load first, restarts it after.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=experiments/2026-11-17_fm_tta/run.log
mkdir -p "$(dirname "$LOG")"
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
PY=~/REOBench/.venv/bin/python
{
  $PY -u scripts/tta/cls_tta_fm.py --datasets aid && echo "AID DONE $(date '+%F %T')" \
    && $PY -u scripts/tta/cls_tta_fm.py --datasets aid --lr_mult 10 --methods tent entf flipf bothf tent_al flipf_al \
    && echo "AID X10 DONE $(date '+%F %T')" \
    && $PY -u scripts/tta/cls_tta_fm.py --datasets r45 && echo "ALL DONE $(date '+%F %T')" || echo "FAILED $(date '+%F %T')"
} 2>&1 | grep --line-buffered -v -i "warn" >> "$LOG"
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
