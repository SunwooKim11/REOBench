#!/bin/bash
# class-shape analysis on stored clean features (17 backbones, AID + RESISC45). tmux session: shape
cd "$(dirname "$0")/.." || exit 1
L=experiments/2026-11-24_cluster_cot/logs/shape.log
TIMM_FUSED_ATTN=0 OMP_NUM_THREADS=4 /opt/conda/envs/segtta/bin/python scripts/analysis/cluster_shape.py > "$L" 2>&1
echo "SHAPE DONE exit $?" >> "$L"
