#!/bin/bash
# Original (entropy) Tent on Cityscapes -> ACDC, 4 conditions x 10 rounds; same settings as tools/tent.py (PL "TENT").
#   setsid nohup ./run_tent_entropy.sh > logs/run_tent_entropy.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
export PYTHONPATH=.
echo "[$(date '+%F %T')] start tent_entropy (10 rounds)"
ROUNDS=10 /opt/conda/envs/segtta/bin/python -u tools/tent_entropy.py \
  --config local_configs/segformer/B5/segformer.b5.1024x1024.acdc.160k.py \
  --checkpoint /data/rs/pretrained_weights/segformer/segformer.b5.1024x1024.city.160k.pth \
  --eval mIoU --options data.workers_per_gpu=8 > logs/tent_entropy.log 2>&1
echo "[$(date '+%F %T')] end   tent_entropy rc=$?"
