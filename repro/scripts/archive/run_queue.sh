#!/bin/bash
# GPU queue after the CoTTA OOM (two ViT-L CoTTA runs do not fit in 140 GB): everything sequential.
#   1. wait for the running scalemae PN cotta (pid given as $1), then drop the temporary temp -> repro symlink
#   2. satmae PN cotta
#   3. ImageNet-C verification (E_verify/run_verify.sh, 3 small jobs in parallel)
#   4. P-config cotta (satmae, scalemae), lower priority
#   5. summaries, restart the DiT dummy load
#   setsid nohup ./run_queue.sh <pid> > logs_queue.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=/home1/irteam/REOBench/.venv/bin/python
echo "[$(date '+%F %T')] queue start (pid $$), waiting for pid $1"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
[ -L ../temp ] && rm ../temp && echo "removed temp -> repro symlink"
(cd /home/irteam/dit_dummy && ./ditload kill) 2>/dev/null

cotta() {  # model cfg
  out=D_tta/outputs/$1/$2/cotta_s0.json
  if [ -f "$out" ]; then echo "skip $1 $2 cotta"; return; fi
  echo "[$(date '+%F %T')] start $1 $2 cotta"
  (cd D_tta && $PY -u tta.py --model $1 --head_cfg $2 --method cotta > logs/$1_$2_cotta.log 2>&1)
  rc=$?
  echo "[$(date '+%F %T')] end   $1 $2 cotta rc=$rc"
}

cotta satmae PN
(cd D_tta && $PY summarize_tta.py > summary_tta.txt 2>&1)
echo "[$(date '+%F %T')] ImageNet-C verification"
(cd E_verify && NO_DIT_RESTART=1 ./run_verify.sh > logs/run_verify.log 2>&1)
tail -n 60 E_verify/summary.txt
cotta satmae P
cotta scalemae P
(cd D_tta && $PY summarize_tta.py > summary_tta.txt 2>&1)
echo "[$(date '+%F %T')] restarting DiT dummy load"
(cd /home/irteam/dit_dummy && ./ditload run-once)
echo "[$(date '+%F %T')] queue DONE"
