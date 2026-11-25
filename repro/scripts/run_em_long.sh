#!/bin/bash
# Long adaptation (2,000 steps) in the Tent protocol, AID-C severity 3, tent vs dsbr. Usage: run_em_long.sh <tag> <models...>
# pass 1 = the 4 priority corruptions, pass 2 = all 12 (finished pairs are skipped). tmux sessions: eml1..eml4. No ditload restart.
cd "$(dirname "$0")/.." || exit 1
TAG=$1; shift
L=experiments/2026-11-26_em_long_tent/logs/$TAG.log
P4="motion_blur_s3 data_gaps_s3 scale_image_s3 gaussian_noise_new_s3"
echo "=== pass 1 $(date '+%F %T') models $*" >> "$L"
~/REOBench/.venv/bin/python -u scripts/tta/cls_tta_long.py --models "$@" --streams $P4 >> "$L" 2>&1
echo "PASS1 DONE $(date '+%F %T')" >> "$L"
~/REOBench/.venv/bin/python -u scripts/tta/cls_tta_long.py --models "$@" >> "$L" 2>&1
echo "ALL DONE $(date '+%F %T')" >> "$L"
