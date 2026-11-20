#!/bin/bash
# No-adaptation ImageNet-C accuracy of the two timm ViT-B/16 checkpoints with ImageNet mean/std input normalisation.
cd ~/REOBench/repro
L=experiments/2026-11-19_dsbr/run_norm.log
(cd /home/irteam/dit_dummy && ./ditload kill) >> $L 2>&1
for CK in augreg augreg2; do
  HF_HUB_OFFLINE=1 ~/REOBench/.venv/bin/python -u -W ignore scripts/verify/dsbr_imagenet_c.py --method none --ckpt $CK --norm imagenet >> $L 2>&1 || echo "STEP FAILED: $CK" >> $L
done
echo "NORM DONE $(date '+%F %T')" >> $L
(cd /home/irteam/dit_dummy && ./ditload run-once) >> $L 2>&1
