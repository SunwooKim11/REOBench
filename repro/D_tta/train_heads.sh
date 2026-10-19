#!/bin/bash
# Train + save the source linear heads used by tta.py (seed 0), via C_ablation/run.py.
#   P  = paper protocol (A0ep: Adam 1e-3, bs16, WarmRestarts per epoch, no normalization)
#   PN = P + per-model input normalization
#   setsid nohup ./train_heads.sh > logs/train_heads.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../C_ablation"
echo "[$(date '+%F %T')] train_heads start; stopping DiT dummy to free GPU memory"
(cd /home/irteam/dit_dummy && ./ditload kill)
cat <<JOBS | xargs -P 4 -L 1 ./one.sh
--model satmae --name P --seed 0 --optim adam_wr_epoch --save_head
--model satmae --name PN --seed 0 --optim adam_wr_epoch --norm --save_head
--model scalemae --name P --seed 0 --bb_train --optim adam_wr_epoch --save_head
--model scalemae --name PN --seed 0 --bb_train --optim adam_wr_epoch --norm --save_head
JOBS
ls -la outputs/*/P*_s0_head.pth
echo "[$(date '+%F %T')] train_heads DONE"
