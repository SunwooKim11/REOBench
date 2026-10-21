#!/bin/bash
# Original Tent with BN on test-batch statistics (TENT_BN=batch), round 1 only. Compare with ATASeg T1 (source 56.9 -> 53.2).
cd "$(dirname "$0")"
export PYTHONPATH=.
echo "[$(date '+%F %T')] start tent_entropy_bnbatch (1 round)"
TENT_BN=batch ROUNDS=1 /opt/conda/envs/segtta/bin/python -u tools/tent_entropy.py \
  --config local_configs/segformer/B5/segformer.b5.1024x1024.acdc.160k.py \
  --checkpoint /data/rs/pretrained_weights/segformer/segformer.b5.1024x1024.city.160k.pth \
  --eval mIoU --options data.workers_per_gpu=8 > logs/tent_entropy_bnbatch.log 2>&1
echo "[$(date '+%F %T')] end   tent_entropy_bnbatch rc=$?"
