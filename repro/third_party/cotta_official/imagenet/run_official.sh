#!/bin/bash
# Official CoTTA repo (imagenet/, only the RandomAffine API line patched) in our environment.
# Separates "our port" from "our environment": compare with E_verify cont_r50 cotta (66.73 err) and ViDA T1 (62.7).
#   setsid nohup ./run_official.sh > run_official.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=/home1/irteam/REOBench/.venv/bin/python
(cd /home/irteam/dit_dummy && ./ditload kill) 2>/dev/null
for c in std_cotta std_tent 10orders/cotta/cotta0; do
  name=$(basename $c)
  echo "[$(date '+%F %T')] start $c"
  $PY -u imagenetc.py --cfg cfgs/$c.yaml SAVE_DIR output/$name > output_$name.log 2>&1
  rc=$?
  echo "[$(date '+%F %T')] end   $c rc=$rc  mean err: $(grep -o 'error % \[[a-z_]*5\]: [0-9.]*' output_$name.log | awk '{s+=$NF; n++} END {if (n) printf "%.2f (%d domains)", s/n, n}')"
done
echo "[$(date '+%F %T')] restarting DiT dummy load"
(cd /home/irteam/dit_dummy && ./ditload run-once)
echo "[$(date '+%F %T')] official DONE"
