#!/bin/bash
# SAR with the official model-recovery threshold e_m = 0.2 (DeYO-repo copy defaults to 0.005), all 4 sources.
#   setsid nohup ./run_sar_rst.sh > logs/run_sar_rst.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=/home1/irteam/REOBench/.venv/bin/python
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
run() {
  out=outputs/$1/$2/sar_s0_rst0.2.json
  if [ -f "$out" ]; then echo "skip $1 $2"; return; fi
  echo "[$(date '+%F %T')] start $1 $2"
  $PY -u tta.py --model $1 --head_cfg $2 --method sar --sar_reset 0.2 > logs/$1_$2_sar_rst0.2.log 2>&1
  rc=$?
  echo "[$(date '+%F %T')] end   $1 $2 rc=$rc"
}
export -f run
export PY
printf "satmae PN\nscalemae PN\nsatmae P\nscalemae P\n" | xargs -P 2 -L 1 bash -c 'run "$@"' _
$PY summarize_tta.py > summary_tta.txt 2>&1
if ! pgrep -f "tools/(tent|cotta|tent_entropy).py" >/dev/null; then
  echo "[$(date '+%F %T')] restarting DiT dummy load"; (cd /home/irteam/dit_dummy && ./ditload run-once)
fi
echo "[$(date '+%F %T')] DONE"
