#!/bin/bash
# (3) DeYO PLPD texture analysis, then (2) lr sensitivity (3e-4, 1e-4) for tent/sar/deyo on all 4 sources.
#   setsid nohup ./run_extra.sh > logs/run_extra.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=/home1/irteam/REOBench/.venv/bin/python
echo "[$(date '+%F %T')] run_extra start (pid $$); stopping DiT dummy"
(cd /home/irteam/dit_dummy && ./ditload kill) 2>/dev/null
for m in satmae scalemae; do for cfg in P PN; do
  [ -f outputs/plpd/${m}_${cfg}.json ] && continue
  $PY -u plpd_analysis.py --model $m --head_cfg $cfg > logs/plpd_${m}_${cfg}.log 2>&1
  echo "[$(date '+%F %T')] plpd $m $cfg rc=$?"
done; done
for lr in 0.0003 0.0001; do for cfg in P PN; do for m in satmae scalemae; do for meth in tent sar deyo; do
  echo "$m $cfg $meth $lr"
done; done; done; done | xargs -P 4 -L 1 sh -c '
  out=outputs/$0/$1/$2_s0_lr$3.json
  if [ -f "$out" ]; then echo "skip $0 $1 $2 lr$3"; exit 0; fi
  echo "[$(date "+%F %T")] start $0 $1 $2 lr$3"
  '"$PY"' -u tta.py --model $0 --head_cfg $1 --method $2 --lr $3 > logs/$0_$1_$2_lr$3.log 2>&1
  echo "[$(date "+%F %T")] end   $0 $1 $2 lr$3 rc=$?"'
$PY summarize_lr.py > summary_lr.txt 2>&1
cat summary_lr.txt
echo "[$(date '+%F %T')] restarting DiT dummy load"
(cd /home/irteam/dit_dummy && ./ditload run-once)
echo "[$(date '+%F %T')] run_extra DONE"
