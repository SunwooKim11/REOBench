#!/bin/bash
# class imbalance + information density checks on stored features (17 backbones, AID + RESISC45). tmux session: rsprop
cd "$(dirname "$0")/.." || exit 1
mkdir -p experiments/2026-11-26_rs_properties/logs
L=experiments/2026-11-26_rs_properties/logs/run.log
(cd /home/irteam/dit_dummy && ./ditload kill) > /dev/null 2>&1
TIMM_FUSED_ATTN=0 OMP_NUM_THREADS=4 /opt/conda/envs/segtta/bin/python scripts/analysis/rs_properties.py > "$L" 2>&1
echo "RSPROP DONE exit $?" >> "$L"
(cd /home/irteam/dit_dummy && ./ditload run-once) > /dev/null 2>&1
