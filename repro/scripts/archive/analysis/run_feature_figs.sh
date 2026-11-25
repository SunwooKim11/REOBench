#!/bin/bash
# redraw docs/figs/2026-11-24_feature_dist (fig0-fig2: new fig1 title; fig3 from shape.json). tmux session: featfig
cd "$(dirname "$0")/.." || exit 1
L=experiments/2026-11-24_cluster_cot/logs/feature_figs.log
(cd /home/irteam/dit_dummy && ./ditload kill) > /dev/null 2>&1
TIMM_FUSED_ATTN=0 OMP_NUM_THREADS=4 /opt/conda/envs/segtta/bin/python scripts/analysis/plot_feature_dist.py > "$L" 2>&1
TIMM_FUSED_ATTN=0 OMP_NUM_THREADS=4 /opt/conda/envs/segtta/bin/python scripts/analysis/plot_feature_dist.py --shape_only >> "$L" 2>&1
(cd /home/irteam/dit_dummy && ./ditload run-once) > /dev/null 2>&1
echo "FIGS DONE" >> "$L"
