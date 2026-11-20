#!/bin/bash
# After feat_metrics.py (needs the saved linear heads): real-haze analysis (RRSHID) and natural-vs-RS corruption analysis.
#   ~/.local/tmux-env/bin/tmux new -d -s extra 'scripts/archive/analysis/run_stage1_extra.sh > experiments/2026-11-16_backbone_feats/extra.log 2>&1'
cd "$(dirname "$0")/../../.."
until grep -q DONE experiments/2026-11-16_backbone_feats/metrics.log; do sleep 60; done
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
~/REOBench/.venv/bin/python -u scripts/analysis/feat_metrics.py 2>&1 | grep --line-buffered -v -i warning | tail -3   # fills any (model, dataset) skipped earlier
~/REOBench/.venv/bin/python -u scripts/analysis/rrshid_analysis.py 2>&1 | grep --line-buffered -v -i "warning\|Loading weights\|meshgrid"
~/REOBench/.venv/bin/python -u scripts/analysis/natural_vs_rs.py 2>&1 | grep --line-buffered -v -i "warning\|Loading weights\|meshgrid"
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
echo "[$(date '+%F %T')] DONE"
