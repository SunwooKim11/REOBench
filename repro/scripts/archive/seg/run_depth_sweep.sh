#!/bin/bash
# Depth sweep of feature statistics alignment (fsa, no learning), MIM models, mild protocol.
#   classification (AID, PN heads): blocks 0,3,7,11,15,19,23, final norm, head input   -> experiments/2026-10-13_cls_tta
#   segmentation (Potsdam, B models): blocks 0,3,7,11,15,19,23, decode-head classifier input; clean + severities 3, 5
#                                     -> experiments/2026-10-20_seg_tta
# Input-level (ISA) and decoder-BN (bnadapt) points already exist. Classification and segmentation run in parallel.
#   setsid nohup scripts/archive/seg/run_depth_sweep.sh > experiments/2026-10-20_seg_tta/logs/run_depth_sweep.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../../.."
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
CL=experiments/2026-10-13_cls_tta/logs; SL=experiments/2026-10-20_seg_tta/logs
cls() {
  for m in satmae scalemae; do for L in blocks.0 blocks.3 blocks.7 blocks.11 blocks.15 blocks.19 blocks.23 norm head:in; do
    echo "[$(date '+%F %T')] cls $m $L"
    ~/REOBench/.venv/bin/python -u scripts/tta/cls_tta.py --model $m --head_cfg PN --method fsa --fsa_layer $L \
      > $CL/${m}_PN_fsa_${L/:/}.log 2>&1 || echo "  FAILED $m $L"
  done; done
}
seg() {
  for m in scalemae satmae; do for L in backbone.blocks.0 backbone.blocks.3 backbone.blocks.7 backbone.blocks.11 \
      backbone.blocks.15 backbone.blocks.19 backbone.blocks.23 decode_head.conv_seg:in; do
    echo "[$(date '+%F %T')] seg $m $L"
    TIMM_FUSED_ATTN=0 /opt/conda/envs/segtta/bin/python -u scripts/seg/seg_tta.py --model $m --method fsa --fsa_layer $L \
      --sevs 3 5 > $SL/tta_${m}_fsa_${L//[.:]/_}.log 2>&1 || echo "  FAILED $m $L"
  done; done
}
cls & seg & wait
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
echo "[$(date '+%F %T')] DONE"
