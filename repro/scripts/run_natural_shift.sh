#!/usr/bin/env bash
# RESISC45 <-> UC Merced natural shift: features for the EarthShift 19-class splits, then natural_shift.py.
# Run from anywhere; logs to experiments/2026-11-17_natural_shift/run.log. Stops the DiT dummy load first, restarts it after.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=experiments/2026-11-17_natural_shift/run.log
mkdir -p "$(dirname "$LOG")"
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
PY=~/REOBench/.venv/bin/python
{
  $PY -u scripts/analysis/extract_feats.py --sets es_r45_train es_r45_val es_r45_test es_ucm_train es_ucm_val es_ucm_test \
    && $PY -u scripts/analysis/natural_shift.py && echo "ALL DONE $(date '+%F %T')" || echo "FAILED $(date '+%F %T')"
} 2>&1 | grep --line-buffered -v -i "warn" >> "$LOG"
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
