#!/bin/bash
# Re-launch of the data-gaps half of scripts/run_undo_acq.sh (the first start stopped on an indexing bug in the angle stage).
#   ~/.local/tmux-env/bin/tmux new -d -s undogaps 'bash scripts/run_undo_gaps.sh'
cd ~/REOBench/repro
E=experiments/2026-11-25_undo_acquisition
PY="$HOME/REOBench/.venv/bin/python -u -W ignore"
L=$E/logs/gaps_repair.log
echo "== gaps_repair (restart) $(date '+%F %T')" >> $L
$PY scripts/analysis/gaps_repair.py --workers 5 >> $L 2>&1 || echo "STEP FAILED: gaps_repair" >> $L
$PY scripts/analysis/gaps_repair.py --summary > $E/logs/gaps_repair_summary.log 2>&1 || echo "STEP FAILED: gaps_repair summary" >> $L
echo "FINISHED gaps_repair (restart) $(date '+%F %T')" >> $L
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
