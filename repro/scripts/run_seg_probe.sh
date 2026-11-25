#!/bin/bash
# Potsdam segmentation with all 17 frozen backbones + a linear head (scripts/seg/seg_probe.py). Two processes share the
# model list (12-core limit): the first starts when the Potsdam fill-in runs are done, the second when the scale search is.
#   ~/.local/tmux-env/bin/tmux new -d -s segprobe 'bash scripts/run_seg_probe.sh'
cd ~/REOBench/repro
E=experiments/2026-11-25_seg_probe
mkdir -p $E/logs $E/signals
PY="$HOME/REOBench/.venv/bin/python -u -W ignore"
until [ "$(cat experiments/2026-11-23_seg_signals/logs/*_fill.log 2>/dev/null | grep -c 'FILL DONE')" -ge 2 ]; do sleep 30; done
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
rm -f $E/signals/*.lock
one() {
  L=$E/logs/$1.log; shift
  echo "== seg_probe $* $(date '+%F %T')" >> $L
  $PY scripts/seg/seg_probe.py "$@" >> $L 2>&1 || echo "STEP FAILED: seg_probe $*" >> $L
  echo "FINISHED $(date '+%F %T')" >> $L
}
one a --workers 5 &
( until grep -q "^ALL DONE" experiments/2026-11-25_undo_acquisition/logs/scale_search.log; do sleep 60; done; one b --workers 5 --reverse ) &
wait
$PY scripts/seg/seg_probe.py --summary > $E/logs/summary.log 2>&1 || echo "STEP FAILED: summary" >> $E/logs/a.log
echo "ALL DONE $(date '+%F %T')" >> $E/logs/a.log
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
