#!/bin/bash
# Tent protocol (norm layers inside the backbone) under class-skewed streams: AID-C, 17 backbones. tmux session: fmorder
# pass 1 = none / tent / dsbr for lt100, lt10, sorted; pass 2 = sar / flipf. Existing files are skipped.
cd "$(dirname "$0")/.." || exit 1
L=experiments/2026-11-17_fm_tta/run_order.log
(cd /home/irteam/dit_dummy && ./ditload kill) > /dev/null 2>&1
for M in "none tent dsbr" "sar flipf"; do
  for O in lt100 lt10 sorted; do
    echo "=== order $O methods $M $(date '+%F %T')" >> "$L"
    ~/REOBench/.venv/bin/python -u scripts/tta/cls_tta_fm.py --datasets aid --methods $M --order $O --workers 8 >> "$L" 2>&1
  done
  echo "PASS DONE ($M) $(date '+%F %T')" >> "$L"
done
echo "FMORDER ALL DONE $(date '+%F %T')" >> "$L"
(cd /home/irteam/dit_dummy && ./ditload run-once) > /dev/null 2>&1
