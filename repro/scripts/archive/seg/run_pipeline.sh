#!/bin/bash
# After the running ScaleMAE training: evaluate ScaleMAE, then train + evaluate SatMAE, summarize, restart DiT.
cd "$(dirname "$0")"
while pgrep -f "run_train.sh scalemae" >/dev/null; do sleep 60; done
./eval_all.sh scalemae > logs/eval_scalemae.log 2>&1
./run_train.sh satmae > logs/train_satmae.log 2>&1
./eval_all.sh satmae > logs/eval_satmae.log 2>&1
/home1/irteam/REOBench/.venv/bin/python summarize.py > summary.txt 2>&1
cat summary.txt
(cd /home/irteam/dit_dummy && ./ditload run-once)
echo "[$(date '+%F %T')] pipeline DONE"
