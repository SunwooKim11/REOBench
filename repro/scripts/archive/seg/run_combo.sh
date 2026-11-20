#!/bin/bash
# Combination of input alignment (batch ISA) with feature alignment, MIM models, mild protocol (after the depth sweep).
#   classification (AID, PN heads): ISA + fsa at blocks 7, 11, head input       -> experiments/2026-10-13_cls_tta
#   segmentation (Potsdam, B models): ISA + fsa at blocks 3, 7, 11; ISA + decoder BN (bnadapt); clean + severities 3, 5
#                                     -> experiments/2026-10-20_seg_tta
# Feature source statistics are taken after ISA on clean train batches. Classification and segmentation run in parallel.
#   tmux new -d -s combo 'scripts/archive/seg/run_combo.sh > experiments/2026-10-20_seg_tta/logs/run_combo.log 2>&1'
cd "$(dirname "$0")/../../.."
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
CL=experiments/2026-10-13_cls_tta/logs; SL=experiments/2026-10-20_seg_tta/logs
cls() {
  for m in satmae scalemae; do for L in blocks.7 blocks.11 head:in; do
    echo "[$(date '+%F %T')] cls $m isa + $L"
    ~/REOBench/.venv/bin/python -u scripts/tta/cls_tta.py --model $m --head_cfg PN --method fsa --fsa_layer $L --with_isa \
      > $CL/${m}_PN_fsa_${L/:/}_isa.log 2>&1 || echo "  FAILED $m $L"
  done; done
}
seg() {
  for m in scalemae satmae; do
    for L in backbone.blocks.3 backbone.blocks.7 backbone.blocks.11; do
      echo "[$(date '+%F %T')] seg $m isa + $L"
      TIMM_FUSED_ATTN=0 /opt/conda/envs/segtta/bin/python -u scripts/seg/seg_tta.py --model $m --method fsa --fsa_layer $L \
        --with_isa --sevs 3 5 > $SL/tta_${m}_fsa_${L//[.:]/_}_isa.log 2>&1 || echo "  FAILED $m $L"
    done
    echo "[$(date '+%F %T')] seg $m isa + bnadapt"
    TIMM_FUSED_ATTN=0 /opt/conda/envs/segtta/bin/python -u scripts/seg/seg_tta.py --model $m --method bnadapt --with_isa \
      --sevs 3 5 > $SL/tta_${m}_bnadapt_isa.log 2>&1 || echo "  FAILED $m bnadapt"
  done
}
cls & seg & wait
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
echo "[$(date '+%F %T')] DONE"
