#!/bin/bash
# Potsdam segmentation with all 17 frozen backbones + a linear head (scripts/seg/seg_probe.py). Two processes share the
# model list through lock files (12-core limit): process a starts when the scale search (tmux undo) has ended, process b
# when the Potsdam fill-in runs (tmux segfill) have ended. Each waits for the whole session to end, because those launchers
# restart the dummy GPU load as their last step, and kills it again before starting.
#   ~/.local/tmux-env/bin/tmux new -d -s segprobe 'bash scripts/run_seg_probe.sh'
cd ~/REOBench/repro
E=experiments/2026-11-25_seg_probe
T=~/.local/tmux-env/bin/tmux
mkdir -p $E/logs $E/signals
rm -f $E/signals/*.lock
PY="$HOME/REOBench/.venv/bin/python -u -W ignore"
after() {  # wait until a tmux session is gone, then stop the dummy GPU load
  while $T has-session -t "$1" 2>/dev/null; do sleep 60; done
  (cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
}
one() {
  L=$E/logs/$1.log; shift
  echo "== seg_probe $* $(date '+%F %T')" >> $L
  $PY scripts/seg/seg_probe.py "$@" >> $L 2>&1 || echo "STEP FAILED: seg_probe $*" >> $L
  echo "FINISHED $(date '+%F %T')" >> $L
}
( after undo; one a --workers 5 ) &
( after segfill; one b --workers 5 --reverse ) &
wait
$PY scripts/seg/seg_probe.py --summary > $E/logs/summary.log 2>&1 || echo "STEP FAILED: summary" >> $E/logs/a.log
echo "ALL DONE $(date '+%F %T')" >> $E/logs/a.log
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
