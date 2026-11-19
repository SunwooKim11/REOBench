#!/usr/bin/env bash
# Rerun of EATA with the SAR / DeYO parameter set (final norm and top blocks excluded) on AID-C and RESISC45-C.
# Stops the DiT dummy load first, restarts it after.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=experiments/2026-11-17_fm_tta/run_eata.log
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
PY=~/REOBench/.venv/bin/python
{
  $PY -u scripts/tta/cls_tta_fm.py --datasets aid r45 --methods eata --workers 10 && echo "ALL DONE $(date '+%F %T')" || echo "FAILED $(date '+%F %T')"
} 2>&1 | grep --line-buffered -v -i "warn\|Loading weights" >> "$LOG"
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
