#!/bin/bash
# Fills the two blanks of the Potsdam 12-corruption table: sar and dsbr on BN batch statistics for the four generated
# corruptions (rotate, scale_image, translate_image, data_gaps). Waits for the data-gaps classification run (CPU limit).
#   ~/.local/tmux-env/bin/tmux new -d -s segfill 'bash scripts/run_seg_fill.sh'
cd ~/REOBench/repro
E=experiments/2026-11-23_seg_signals
until grep -q "^FINISHED gaps_repair (restart)" experiments/2026-11-25_undo_acquisition/logs/gaps_repair.log; do sleep 30; done
PY="/opt/conda/envs/segtta/bin/python -u"
export TIMM_FUSED_ATTN=0
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
NEW="rotate scale_image translate_image data_gaps"
one() {
  M=$1; L=$E/logs/${M}_fill.log
  run() { echo "== $* $(date '+%F %T')" >> $L; "$@" >> $L 2>&1 || echo "STEP FAILED: $*" >> $L; }
  for cfg in "sar --bn fixed" "dsbr --bn batch"; do
    run $PY scripts/seg/seg_tta.py --model $M --method $cfg --corruptions $NEW --no_clean --tag _geo4 --workers 3 --out_dir 2026-11-23_seg_signals
  done
  echo "FILL DONE $M $(date '+%F %T')" >> $L
}
one scalemae &
one satmae &
wait
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
