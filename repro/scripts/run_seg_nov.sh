#!/bin/bash
# Potsdam segmentation with the November signals / methods, both models in parallel (5 loader workers each: 12-core limit).
#   ~/.local/tmux-env/bin/tmux new -d -s segnov 'bash scripts/run_seg_nov.sh'
# Order per model: label-free signals (no learning) -> flipf -> dsbr -> dsbr_src -> the same three on BN batch statistics.
# Results: experiments/2026-11-23_seg_signals/{signals,outputs}/, logs in logs/<model>.log
cd ~/REOBench/repro
E=experiments/2026-11-23_seg_signals
PY="/opt/conda/envs/segtta/bin/python -u"
export TIMM_FUSED_ATTN=0
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
one() {
  M=$1; L=$E/logs/$M.log
  run() { echo "== $* $(date '+%F %T')" >> $L; "$@" >> $L 2>&1 || echo "STEP FAILED: $*" >> $L; }
  run $PY scripts/seg/seg_signals.py --model $M
  echo "SIGNALS DONE $M $(date '+%F %T')" >> $L
  for cfg in "flipf --bn fixed" "dsbr --bn fixed" "dsbr_src --bn fixed"; do
    run $PY scripts/seg/seg_tta.py --model $M --method $cfg --workers 5 --out_dir 2026-11-23_seg_signals
  done
  echo "STAGE2 DONE $M $(date '+%F %T')" >> $L
  for cfg in "flipf --bn batch" "dsbr_src --bn batch" "dsbr --bn batch"; do
    run $PY scripts/seg/seg_tta.py --model $M --method $cfg --workers 5 --out_dir 2026-11-23_seg_signals
  done
  echo "ALL DONE $M $(date '+%F %T')" >> $L
}
one scalemae &
one satmae &
wait
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
