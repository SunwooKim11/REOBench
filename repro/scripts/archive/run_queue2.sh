#!/bin/bash
# PN-config CoTTA reruns after the cotta_ref memory-leak fix in D_tta/tta.py (both OOM'd at ~stream 20).
# Waits for run_queue.sh (ImageNet-C verify + P-config CoTTA), then runs the PN ones, summarizes, restarts DiT.
#   setsid nohup ./run_queue2.sh > logs_queue2.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=/home1/irteam/REOBench/.venv/bin/python
echo "[$(date '+%F %T')] queue2 start (pid $$), waiting for run_queue.sh"
while pgrep -f "run_queue.sh [0-9]" >/dev/null; do sleep 60; done
(cd /home/irteam/dit_dummy && ./ditload kill) 2>/dev/null
for mc in "satmae PN" "scalemae PN"; do
  set -- $mc
  out=D_tta/outputs/$1/$2/cotta_s0.json
  if [ -f "$out" ]; then echo "skip $1 $2 cotta"; continue; fi
  echo "[$(date '+%F %T')] start $1 $2 cotta"
  (cd D_tta && $PY -u tta.py --model $1 --head_cfg $2 --method cotta > logs/$1_$2_cotta.log 2>&1)
  rc=$?
  echo "[$(date '+%F %T')] end   $1 $2 cotta rc=$rc"
done
(cd D_tta && $PY summarize_tta.py > summary_tta.txt 2>&1)
echo "[$(date '+%F %T')] restarting DiT dummy load"
(cd /home/irteam/dit_dummy && ./ditload run-once)
echo "[$(date '+%F %T')] queue2 DONE"
