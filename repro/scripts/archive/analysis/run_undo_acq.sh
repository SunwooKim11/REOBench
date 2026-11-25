#!/bin/bash
# Method candidate 1 ("undo the acquisition condition"), two upper-bound measurements on all 17 backbones, run side by side:
#   scale_search.py : 11 apparent magnifications + 2x2 tiling on clean / scale_image streams and RESISC45 <-> UC Merced
#   gaps_repair.py  : data_gaps streams with the invalid stripes filled (pixels) or excluded (ViT tokens)
#   ~/.local/tmux-env/bin/tmux new -d -s undo 'bash scripts/run_undo_acq.sh'
cd ~/REOBench/repro
E=experiments/2026-11-25_undo_acquisition
mkdir -p $E/logs
PY="$HOME/REOBench/.venv/bin/python -u -W ignore"
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
sleep 20
one() {
  S=$1; shift; L=$E/logs/$S.log
  echo "== $S $(date '+%F %T')" >> $L
  $PY scripts/analysis/$S.py "$@" >> $L 2>&1 || echo "STEP FAILED: $S" >> $L
  $PY scripts/analysis/$S.py --summary > $E/logs/${S}_summary.log 2>&1 || echo "STEP FAILED: $S summary" >> $L
  echo "FINISHED $S $(date '+%F %T')" >> $L
}
one scale_search --workers 5 &
one gaps_repair --workers 5 &
wait
echo "ALL DONE $(date '+%F %T')" >> $E/logs/scale_search.log
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
