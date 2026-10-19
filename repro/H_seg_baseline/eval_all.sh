#!/bin/bash
# Evaluate a trained Potsdam model on clean + the 8 released corruptions x 5 severities (41 test sets, 2,016 patches each).
# Each run writes eval_*.json (mmseg) under results/<model>/<cond>[_s<k>]/.
#   setsid nohup ./eval_all.sh scalemae > logs/eval_scalemae.log 2>&1 < /dev/null &
cd "$(dirname "$0")/seg"
export PYTHONPATH=. TIMM_FUSED_ATTN=0
m=$1; SET=${2:-A}; CFG=configs/repro/${m}_${SET}_12ep.py; CKPT=../work_dirs/${m}_${SET}_12ep/latest.pth
R=$m; [ "$SET" != A ] && R=${m}_$SET   # results/<model> for A (kept), results/<model>_<set> otherwise
T=/data/rs/reobench/Potsdam_extracted/test
run() {  # name img_dir
  out=../results/$R/$1
  if ls $out/eval_*.json >/dev/null 2>&1; then echo "skip $1"; return; fi
  mkdir -p $out
  /opt/conda/envs/segtta/bin/python -u tools/test.py $CFG $CKPT --eval mIoU --work-dir $out \
      --cfg-options data.test.img_dir=$2 data.test.ann_dir=$T/clean_ann data.workers_per_gpu=8 > $out/log.txt 2>&1
  echo "[$(date '+%F %T')] $1 rc=$? $(grep -A3 'aAcc' $out/log.txt | tail -1)"
}
run clean $T/clean
for c in brightness_contrast clouds compression_artifacts gaussian_blur gaussian_noise haze motion_blur salt_and_pepper_noise; do
  for s in 1 2 3 4 5; do run ${c}_s$s $T/$c/severity_$s; done
done
echo "[$(date '+%F %T')] DONE"
