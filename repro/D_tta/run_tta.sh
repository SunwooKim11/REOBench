#!/bin/bash
# Full mild-scenario TTA run: 2 models x 2 head configs (P, PN) x 4 methods (none, tent, sar, deyo).
# Waits for the heads from train_heads.sh, skips finished results, then summarizes and restarts DiT.
#   setsid nohup ./run_tta.sh > logs/run_tta.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=/home1/irteam/REOBench/.venv/bin/python
H=../C_ablation/outputs
echo "[$(date '+%F %T')] run_tta start (pid $$); waiting for heads"
until [ -f $H/satmae/P_s0_head.pth ] && [ -f $H/satmae/PN_s0_head.pth ] && \
      [ -f $H/scalemae/P_s0_head.pth ] && [ -f $H/scalemae/PN_s0_head.pth ]; do sleep 60; done
sleep 30
(cd /home/irteam/dit_dummy && ./ditload kill) 2>/dev/null
echo "[$(date '+%F %T')] heads ready"
for cfg in P PN; do for m in satmae scalemae; do for meth in none tent sar deyo; do
  echo "$m $cfg $meth"
done; done; done | xargs -P 4 -L 1 sh -c '
  out=outputs/$0/$1/$2_s0.json
  if [ -f "$out" ]; then echo "skip $0 $1 $2"; exit 0; fi
  echo "[$(date "+%F %T")] start $0 $1 $2"
  '"$PY"' -u tta.py --model $0 --head_cfg $1 --method $2 > logs/$0_$1_$2.log 2>&1
  echo "[$(date "+%F %T")] end   $0 $1 $2 rc=$?"'
$PY summarize_tta.py > summary_tta.txt 2>&1
cat summary_tta.txt
echo "[$(date '+%F %T')] restarting DiT dummy load"
(cd /home/irteam/dit_dummy && ./ditload run-once)
echo "[$(date '+%F %T')] run_tta DONE"
