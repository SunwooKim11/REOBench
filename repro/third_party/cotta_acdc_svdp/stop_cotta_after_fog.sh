#!/bin/bash
# CoTTA seg is ~20 s/image here: keep only the first condition of round 1 (fog, 400 images; paper 70.9 mIoU).
cd "$(dirname "$0")"
until grep -aq "| global" logs/cotta.log 2>/dev/null; do sleep 60; done
pkill -f "[t]ools/cotta.py"
echo "[$(date '+%F %T')] stopped CoTTA after fog: $(grep -a '| global' logs/cotta.log | head -1)"
