#!/bin/bash
# The four Potsdam-C corruptions missing from the release (rotate, scale_image, translate_image, data_gaps): generate them,
# then signals and TTA methods for both models. Starts after scripts/run_seg_nov.sh has finished.
#   ~/.local/tmux-env/bin/tmux new -d -s seggeo 'bash scripts/run_seg_geo4.sh'
cd ~/REOBench/repro
E=experiments/2026-11-23_seg_signals
G=$E/logs/gen_missing.log
~/REOBench/.venv-corrupt/bin/python -u -W ignore scripts/tools/make_potsdam_c_missing.py --workers 3 > $G 2>&1 && echo "GEN DONE $(date '+%F %T')" >> $G || echo "STEP FAILED: generation" >> $G
until [ "$(grep -l 'ALL DONE' $E/logs/scalemae.log $E/logs/satmae.log 2>/dev/null | wc -l)" -ge 2 ]; do sleep 60; done
grep -q "GEN DONE" $G || exit 1
sleep 90
PY="/opt/conda/envs/segtta/bin/python -u"
export TIMM_FUSED_ATTN=0
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
ALL="brightness_contrast clouds compression_artifacts gaussian_blur gaussian_noise haze motion_blur salt_and_pepper_noise rotate scale_image translate_image data_gaps"
NEW="rotate scale_image translate_image data_gaps"
one() {
  M=$1; L=$E/logs/${M}_geo4.log
  run() { echo "== $* $(date '+%F %T')" >> $L; "$@" >> $L 2>&1 || echo "STEP FAILED: $*" >> $L; }
  run $PY scripts/seg/seg_signals.py --model $M --corruptions $ALL      # resumes: only the 20 new streams are computed
  echo "SIGNALS DONE $M $(date '+%F %T')" >> $L
  for cfg in "tent --bn fixed" "tent --bn batch" "dsbr --bn fixed" "flipf --bn fixed" "flipf --bn batch"; do
    run $PY scripts/seg/seg_tta.py --model $M --method $cfg --corruptions $NEW --no_clean --tag _geo4 --workers 5 --out_dir 2026-11-23_seg_signals
  done
  echo "ALL DONE $M $(date '+%F %T')" >> $L
}
one scalemae &
one satmae &
wait
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
