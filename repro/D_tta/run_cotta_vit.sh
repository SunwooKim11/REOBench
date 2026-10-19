#!/bin/bash
# CoTTA rerun with the ViT settings of EATA-C (lr 0.005, p_th 0.1) — the first RS run (lr 0.01, p_th = 5% quantile
# of source confidence ~0.97, i.e. augmenting every batch) collapsed on every source. Sequential (ViT-L CoTTA ~70 GB).
#   setsid nohup ./run_cotta_vit.sh > logs/run_cotta_vit.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=/home1/irteam/REOBench/.venv/bin/python
(cd /home/irteam/dit_dummy && ./ditload kill) 2>/dev/null
for mc in "satmae PN" "scalemae PN" "satmae P" "scalemae P"; do
  set -- $mc
  out=outputs/$1/$2/cotta_s0_lr0.005_ap0.1.json
  if [ -f "$out" ]; then echo "skip $1 $2"; continue; fi
  echo "[$(date '+%F %T')] start $1 $2 cotta lr0.005 ap0.1"
  $PY -u tta.py --model $1 --head_cfg $2 --method cotta --lr 0.005 --cotta_ap 0.1 > logs/$1_$2_cotta_lr0.005_ap0.1.log 2>&1
  rc=$?
  echo "[$(date '+%F %T')] end   $1 $2 rc=$rc"
done
$PY summarize_tta.py > summary_tta.txt 2>&1
echo "[$(date '+%F %T')] restarting DiT dummy load"
(cd /home/irteam/dit_dummy && ./ditload run-once)
echo "[$(date '+%F %T')] DONE"
