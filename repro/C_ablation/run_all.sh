#!/bin/bash
# Full ablation pipeline, safe to run detached and to re-run (finished results are skipped):
#   setsid nohup ./run_all.sh > logs/run_all.log 2>&1 < /dev/null &
# 1) feature caches  2) 39 jobs, 6 in parallel  3) summary.txt  4) restart DiT dummy GPU load
cd "$(dirname "$0")"
PY=/home1/irteam/REOBench/.venv/bin/python
PARALLEL=${PARALLEL:-6}
mkdir -p logs outputs cache
echo "[$(date '+%F %T')] run_all start (pid $$)"

for m in satmae scalemae; do
  for tag in raw norm; do
    [ -f "cache/${m}_${tag}.pt" ] && continue
    flag=""; [ "$tag" = norm ] && flag="--norm"
    echo "[$(date '+%F %T')] prep $m $tag"
    $PY -u run.py --prep --model $m $flag > "logs/prep_${m}_${tag}.log" 2>&1 || echo "prep $m $tag FAILED"
  done
done

./jobs.sh | xargs -P "$PARALLEL" -L 1 ./one.sh

n=$(ls outputs/*/*.json 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] jobs finished: $n/39 results"
$PY summarize.py > summary.txt 2>&1
cat summary.txt

echo "[$(date '+%F %T')] restarting DiT dummy load"
(cd /home/irteam/dit_dummy && ./ditload run-once)
echo "[$(date '+%F %T')] run_all DONE"
