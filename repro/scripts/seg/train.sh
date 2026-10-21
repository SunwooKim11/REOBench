#!/bin/bash
# Train a Potsdam segmentor: setting A (released config + paper schedule 12 ep / bs 8) or B (fixed). Eval: eval_all.sh.
#   setsid nohup scripts/seg/train.sh scalemae B > experiments/2026-10-16_seg_baseline/logs/train_scalemae_B.log 2>&1 < /dev/null &
REPO="$(cd "$(dirname "$0")/../.." && pwd)"                       # repro/
EXP="$REPO/experiments/2026-10-16_seg_baseline"
cd "$REPO/third_party/reobench_seg"                                 # mmseg configs are relative to this folder
export PYTHONPATH=. TIMM_FUSED_ATTN=0   # torch 2.0.1 fused SDPA backward crashes on H200 (illegal instruction)
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
m=$1; SET=${2:-A}
echo "[$(date '+%F %T')] start train $m $SET"
/opt/conda/envs/segtta/bin/python -u tools/train.py configs/repro/${m}_${SET}_12ep.py --work-dir $EXP/work_dirs/${m}_${SET}_12ep --no-validate --seed 0
rc=$?
echo "[$(date '+%F %T')] end train $m $SET rc=$rc"
