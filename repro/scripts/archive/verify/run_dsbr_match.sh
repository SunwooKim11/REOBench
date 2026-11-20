#!/bin/bash
# ImageNet-C check in the setting whose no-adaptation mean equals the paper's 39.8: timm augreg2 ViT-B/16 with ImageNet mean/std input.
cd ~/REOBench/repro
L=experiments/2026-11-19_dsbr/run_match.log
(cd /home/irteam/dit_dummy && ./ditload kill) >> $L 2>&1
run() { HF_HUB_OFFLINE=1 ~/REOBench/.venv/bin/python -u -W ignore scripts/verify/dsbr_imagenet_c.py --ckpt augreg2 --norm imagenet "$@" >> $L 2>&1 || echo "STEP FAILED: $*" >> $L; }
run --method dsbr --alpha 0.9 --opt adam --lr 0.00025
run --method tent --opt sgd --lr 0.00025
echo "MATCH1 DONE $(date '+%F %T')" >> $L
run --method tent --opt adam --lr 0.00025
run --method dsbr --alpha 0.99 --opt adam --lr 0.00025
echo "MATCH ALL DONE $(date '+%F %T')" >> $L
(cd /home/irteam/dit_dummy && ./ditload run-once) >> $L 2>&1
