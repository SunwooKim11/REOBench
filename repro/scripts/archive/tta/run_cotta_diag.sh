#!/bin/bash
# CoTTA collapse diagnosis (SatMAE PN, p_th 0.1): is it the update set (all ViT-L weights + head) or the lr?
# 7 streams: clean + {haze, gaussian noise, motion blur} x severity {3,5}.
#   setsid nohup ./run_cotta_diag.sh > logs/run_cotta_diag.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=/home1/irteam/REOBench/.venv/bin/python
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
run() {  # params lr
  echo "[$(date '+%F %T')] start params=$1 lr=$2"
  $PY -u tta.py --model satmae --head_cfg PN --method cotta --cotta_ap 0.1 --cotta_params $1 --lr $2 \
      --noises haze_update_new gaussian_noise_new motion_blur --sevs 3 5 --tag _diag > logs/cotta_diag_$1_lr$2.log 2>&1
  rc=$?
  echo "[$(date '+%F %T')] end   params=$1 lr=$2 rc=$rc"
}
export -f run; export PY
printf "all 0.01\nall 1e-05\nln 0.01\nln 0.001\n" | xargs -P 2 -L 1 bash -c 'run "$@"' _
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
echo "[$(date '+%F %T')] DONE"
