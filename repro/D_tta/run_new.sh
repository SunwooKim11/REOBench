#!/bin/bash
# New baselines (EATA, CoTTA, LSCD-TTA) and input statistics alignment on all 4 sources, mild scenario.
# Waits for run_extra.sh (lr sweep) to finish, frees the GPU, runs, summarizes, restarts DiT.
#   setsid nohup ./run_new.sh > logs/run_new.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
export PY=/home1/irteam/REOBench/.venv/bin/python
echo "[$(date '+%F %T')] run_new start (pid $$), waiting for run_extra.sh"
while pgrep -f "^/bin/bash ./run_extra.sh" >/dev/null; do sleep 60; done
(cd /home/irteam/dit_dummy && ./ditload kill) 2>/dev/null

run() {  # args: model cfg method
  out=outputs/$1/$2/$3_s0.json
  if [ -f "$out" ]; then echo "skip $1 $2 $3"; return; fi
  echo "[$(date '+%F %T')] start $1 $2 $3"
  $PY -u tta.py --model $1 --head_cfg $2 --method $3 > logs/$1_$2_$3.log 2>&1
  rc=$?  # capture before $(date) resets $?
  echo "[$(date '+%F %T')] end   $1 $2 $3 rc=$rc"
}
export -f run

for cfg in PN P; do for m in satmae scalemae; do for meth in isa_batch isa_image eata lscd; do
  echo "$m $cfg $meth"
done; done; done | xargs -P 4 -L 1 bash -c 'run "$@"' _

# CoTTA updates all ViT-L weights and keeps 3 model copies: run 2 at a time
for cfg in PN P; do for m in satmae scalemae; do
  echo "$m $cfg cotta"
done; done | xargs -P 2 -L 1 bash -c 'run "$@"' _

$PY summarize_tta.py > summary_tta.txt 2>&1
cat summary_tta.txt
echo "[$(date '+%F %T')] restarting DiT dummy load"
(cd /home/irteam/dit_dummy && ./ditload run-once)
echo "[$(date '+%F %T')] run_new DONE"
