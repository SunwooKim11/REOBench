#!/bin/bash
# Cluster geometry + COT on the saved features, 17 backbones x (AID, RESISC45), 4 shards in parallel (CPU-bound: exact OT).
# tmux: ~/.local/tmux-env/bin/tmux new -d -s clcot 'bash scripts/run_cluster_cot.sh'
cd ~/REOBench/repro || exit 1
PY=/opt/conda/envs/segtta/bin/python
LOG=experiments/2026-11-24_cluster_cot/logs; mkdir -p $LOG
M=(mae_in satmae scalemae moco_in moco_fmow gassl_fmow dino_in_s16 satdino_s16 dino_in_s8 satdino_s8 dino_in_b16 satdino_b16 dinov3_web_l16 dinov3_sat_l16 sup_in_r50 sup_in_vitb16 sup_in_vitl16)
for ds in aid r45; do
  for k in 0 1 2 3; do
    sel=(); for i in "${!M[@]}"; do [ $((i % 4)) -eq $k ] && sel+=("${M[$i]}"); done
    OMP_NUM_THREADS=1 TIMM_FUSED_ATTN=0 $PY -W ignore scripts/analysis/cluster_cot.py --datasets $ds --models "${sel[@]}" >> $LOG/shard$k.log 2>&1 &
  done
  wait
  echo "DONE $ds $(date '+%F %T')" >> $LOG/main.log
done
$PY -W ignore scripts/analysis/cluster_cot.py --summary > $LOG/summary.log 2>&1
echo "ALL DONE $(date '+%F %T')" >> $LOG/main.log
