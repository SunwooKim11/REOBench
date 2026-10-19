#!/bin/bash
# Setting B: train ScaleMAE and SatMAE (sequential), evaluate both, summarize, restart DiT.
#   setsid nohup ./run_B.sh > logs/run_B.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
for m in scalemae satmae; do
  ./run_train.sh $m B > logs/train_${m}_B.log 2>&1
  echo "[$(date '+%F %T')] trained $m B: $(tail -1 logs/train_${m}_B.log)"
  ./eval_all.sh $m B > logs/eval_${m}_B.log 2>&1
  echo "[$(date '+%F %T')] evaluated $m B"
done
/home1/irteam/REOBench/.venv/bin/python summarize.py > summary.txt 2>&1; cat summary.txt
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
echo "[$(date '+%F %T')] DONE"
