#!/bin/bash
# Cityscapes -> ACDC verification (CoTTA Table 5: SegFormer-B5, 960x540 input, 4 conditions x 10 rounds, bs 1).
# Code: CoTTA's ACDC code as copied in the SVDP repo (see README_REOBench.md for the patches).
# Runs next to the RS CoTTA job; keeps the DiT dummy off while running (it needs ~87 GB).
#   setsid nohup ./run_seg_verify.sh > logs/run_seg_verify.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
export PYTHONPATH=.
PY=/opt/conda/envs/segtta/bin/python
CFG=local_configs/segformer/B5/segformer.b5.1024x1024.acdc.160k.py
CKPT=/data/rs/pretrained_weights/segformer/segformer.b5.1024x1024.city.160k.pth
( while true; do (cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1; sleep 60; done ) &
KEEPER=$!
for c in ${SOURCE_CONDS:-fog night rain snow}; do
  echo "[$(date '+%F %T')] source $c"
  $PY -u tools/test.py --config $CFG --checkpoint $CKPT --eval mIoU \
      --options data.test.img_dir=rgb_anon/$c/train data.test.ann_dir=gt/$c/train > logs/source_$c.log 2>&1
  echo "  rc=$? $(grep -A4 Summary logs/source_$c.log | grep global)"
done
# tent.py takes --config/--checkpoint, cotta.py positional args (as in the repo).
# CoTTA (7 scales x flip, 3 models) is ~20 s/image here -> only round 1 (paper reports it per condition).
echo "[$(date '+%F %T')] start tent (10 rounds)"
ROUNDS=10 $PY -u tools/tent.py --config $CFG --checkpoint $CKPT --eval mIoU --options data.workers_per_gpu=8 > logs/tent.log 2>&1
echo "[$(date '+%F %T')] end   tent rc=$?"
echo "[$(date '+%F %T')] start cotta (round 1)"
ROUNDS=1 $PY -u tools/cotta.py $CFG $CKPT --eval mIoU --options data.workers_per_gpu=8 > logs/cotta.log 2>&1
echo "[$(date '+%F %T')] end   cotta rc=$?"
kill $KEEPER
if ! pgrep -f "tta.py --model" >/dev/null; then
  echo "[$(date '+%F %T')] restarting DiT dummy load"; (cd /home/irteam/dit_dummy && ./ditload run-once)
fi
echo "[$(date '+%F %T')] DONE"
