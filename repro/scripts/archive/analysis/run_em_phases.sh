#!/bin/bash
# Two phases of entropy minimisation at the feature level (tent vs DSBR, 2,000 steps), 17 backbones on AID, 2 shards.
# tmux: ~/.local/tmux-env/bin/tmux new -d -s emph 'bash scripts/run_em_phases.sh'
cd ~/REOBench/repro || exit 1
PY=/opt/conda/envs/segtta/bin/python
LOG=experiments/2026-11-24_cluster_cot/logs; mkdir -p $LOG
M=(mae_in scalemae moco_in moco_fmow gassl_fmow dino_in_s16 satdino_s16 dino_in_s8 satdino_s8 dino_in_b16 satdino_b16 dinov3_web_l16 dinov3_sat_l16 sup_in_r50 sup_in_vitb16 sup_in_vitl16 satmae)
for k in 0 1; do
  sel=(); for i in "${!M[@]}"; do [ $((i % 2)) -eq $k ] && sel+=("${M[$i]}"); done
  OMP_NUM_THREADS=1 TIMM_FUSED_ATTN=0 $PY -W ignore scripts/analysis/em_phases.py --datasets aid --models "${sel[@]}" >> $LOG/em_shard$k.log 2>&1 &
done
wait
$PY -W ignore scripts/analysis/em_phases.py --summary > $LOG/em_summary.log 2>&1
echo "EM ALL DONE $(date '+%F %T')" >> $LOG/main.log
