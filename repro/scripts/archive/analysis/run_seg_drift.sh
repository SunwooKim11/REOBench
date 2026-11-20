#!/bin/bash
# Stage 1 for segmentation (scripts/analysis/seg_drift.py), both Potsdam B models. Runs alongside the classification
# feature extraction; the DiT dummy is handled by run_extract_feats.sh / run_feat_metrics.sh.
#   ~/.local/tmux-env/bin/tmux new -d -s segdrift 'scripts/archive/analysis/run_seg_drift.sh > experiments/2026-11-16_backbone_feats/seg/run.log 2>&1'
cd "$(dirname "$0")/../../.."
for m in scalemae satmae; do
  echo "[$(date '+%F %T')] $m"
  TIMM_FUSED_ATTN=0 /opt/conda/envs/segtta/bin/python -u scripts/analysis/seg_drift.py --model $m 2>&1 \
    | grep --line-buffered -E "^(clean|[a-z_]+_s[0-9]) |Error|Traceback"
done
echo "[$(date '+%F %T')] DONE"
