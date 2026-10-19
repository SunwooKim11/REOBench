#!/bin/bash
# Potsdam baseline, setting A (released config + paper schedule 12 ep / bs 8). Train only; evaluation is eval_all.sh.
#   setsid nohup ./run_train.sh scalemae > logs/train_scalemae.log 2>&1 < /dev/null &
cd "$(dirname "$0")/seg"
export PYTHONPATH=. TIMM_FUSED_ATTN=0   # torch 2.0.1 fused SDPA backward crashes on H200 (illegal instruction)
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
m=$1; SET=${2:-A}
echo "[$(date '+%F %T')] start train $m $SET"
/opt/conda/envs/segtta/bin/python -u tools/train.py configs/repro/${m}_${SET}_12ep.py --work-dir ../work_dirs/${m}_${SET}_12ep --no-validate --seed 0
echo "[$(date '+%F %T')] end train $m $SET rc=$?"
