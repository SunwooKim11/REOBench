#!/bin/bash
# Waits for scripts/archive/analysis/run_extract_feats.sh to finish, then computes stage-1 metrics on GPU (DiT dummy off meanwhile).
#   ~/.local/tmux-env/bin/tmux new -d -s fmetrics 'scripts/archive/analysis/run_feat_metrics.sh > experiments/2026-11-16_backbone_feats/metrics.log 2>&1'
cd "$(dirname "$0")/../../.."
until grep -q DONE experiments/2026-11-16_backbone_feats/run.log; do sleep 60; done
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
~/REOBench/.venv/bin/python -u scripts/analysis/feat_metrics.py 2>&1 | grep --line-buffered -v -i "warning"
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
echo "[$(date '+%F %T')] DONE"
