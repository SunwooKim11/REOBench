#!/bin/bash
# Stage 1: features of all backbones on AID(-C) and RESISC45(-C) (scripts/analysis/extract_feats.py), then metrics.
#   ~/.local/tmux-env/bin/tmux new -d -s feats 'scripts/run_extract_feats.sh > experiments/2026-11-16_backbone_feats/run.log 2>&1'
cd "$(dirname "$0")/.."
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
~/REOBench/.venv/bin/python -u scripts/analysis/extract_feats.py 2>&1 | grep --line-buffered -v -i "warning\|Loading weights\|meshgrid"
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
echo "[$(date '+%F %T')] DONE"
