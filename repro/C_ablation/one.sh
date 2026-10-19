#!/bin/bash
# Run one ablation job. Args are passed to run.py: --model M --name N --seed S [flags...]
cd "$(dirname "$0")"
m=$2; n=$4; s=$6
[ -f "outputs/$m/${n}_s$s.json" ] && { echo "skip $m $n s$s (done)"; exit 0; }
echo "[$(date '+%F %T')] start $m $n s$s"
/home1/irteam/REOBench/.venv/bin/python -u run.py "$@" > "logs/${m}_${n}_s$s.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] end   $m $n s$s rc=$rc"
