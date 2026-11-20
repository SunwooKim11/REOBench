#!/usr/bin/env bash
# DSBR (arXiv 2606.02339): 17 backbones on AID-C / RESISC45-C (scripts/tta/cls_tta_fm.py) and the ImageNet-C check of the
# implementation (scripts/verify/dsbr_imagenet_c.py). Waits for the Weighted Flips run, stops the DiT dummy load, restarts it after.
set -uo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p experiments/2026-11-19_dsbr
LOG=experiments/2026-11-19_dsbr/run.log
until grep -q "ALL DONE\|FAILED" experiments/2026-11-19_weighted_flips_repro/run.log; do sleep 20; done
sleep 60
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
PY=~/REOBench/.venv/bin/python
export HF_HUB_OFFLINE=1
fm() { $PY -u scripts/tta/cls_tta_fm.py --workers 10 "$@" || echo "STEP FAILED: fm $*"; }
inc() { $PY -u scripts/verify/dsbr_imagenet_c.py "$@" || echo "STEP FAILED: inc $*"; }
{
  fm --datasets aid --methods dsbr --lr_mult 10
  fm --datasets aid --methods dsbr
  echo "STAGE1 DONE $(date '+%F %T')"
  inc --method none --ckpt augreg2
  inc --method none --ckpt augreg
  CK=$($PY -c "
import json
d='experiments/2026-11-19_dsbr/imagenet_c/'
m={k: json.load(open(d+k+'_none.json'))['mean'] for k in ('augreg','augreg2')}
print(min(m, key=lambda k: abs(m[k]-39.8)))")
  echo "checkpoint closest to the paper's 39.8: $CK"
  inc --ckpt "$CK" --method dsbr --alpha 0.9 --opt adam --lr 0.00025
  inc --ckpt "$CK" --method tent --opt sgd --lr 0.00025
  echo "STAGE2 DONE $(date '+%F %T')"
  fm --datasets aid --methods dsbr_al dsbr_a99
  fm --datasets aid --methods dsbr_al dsbr_a99 --lr_mult 10
  echo "STAGE3 DONE $(date '+%F %T')"
  fm --datasets r45 --methods dsbr dsbr_al
  echo "STAGE4 DONE $(date '+%F %T')"
  inc --ckpt "$CK" --method dsbr --alpha 0.99 --opt adam --lr 0.00025
  inc --ckpt "$CK" --method tent --opt adam --lr 0.00025
  echo "ALL DONE $(date '+%F %T')"
} 2>&1 | grep --line-buffered -v -i "warn\|Loading weights\|access token" >> "$LOG"
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
