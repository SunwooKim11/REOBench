#!/bin/bash
# Detached runner for the paper-scheduler variant (A0ep). Re-runnable; finished results are skipped.
#   setsid nohup ./run_sched.sh > logs/run_sched.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
echo "[$(date '+%F %T')] run_sched start (pid $$)"
./jobs_sched.sh | xargs -P 6 -L 1 ./one.sh
/home1/irteam/REOBench/.venv/bin/python summarize.py > summary.txt 2>&1
cat summary.txt
echo "[$(date '+%F %T')] restarting DiT dummy load"
(cd /home/irteam/dit_dummy && ./ditload run-once)
echo "[$(date '+%F %T')] run_sched DONE"
