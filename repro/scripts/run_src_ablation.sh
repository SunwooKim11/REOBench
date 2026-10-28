#!/bin/bash
# How much source information do ISA and feature alignment need? Classification (AID, PN heads), mild protocol.
#   _n32   : source statistics from 32 clean train images instead of 2,000 (FOA uses 32)
#   _isapre: source-free ISA, target = the pretraining Normalize constants shipped with the model (no AID train data)
#   tmux new -d -s srcabl 'scripts/run_src_ablation.sh > experiments/2026-10-13_cls_tta/logs/run_src_ablation.log 2>&1'
cd "$(dirname "$0")/.."
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
CL=experiments/2026-10-13_cls_tta/logs
PY="$HOME/REOBench/.venv/bin/python -u scripts/tta/cls_tta.py --head_cfg PN"
run() {  # model, log name, args...
  local m=$1 n=$2; shift 2
  echo "[$(date '+%F %T')] $m $n"
  $PY --model $m "$@" > $CL/${m}_PN_${n}.log 2>&1 || echo "  FAILED $m $n"
}
one() {
  local m=$1
  run $m isa_batch_n32            --method isa_batch --n_src 32
  run $m isa_batch_isapre         --method isa_batch --isa_target pretrain
  run $m fsa_blocks.7_n32         --method fsa --fsa_layer blocks.7 --n_src 32
  run $m fsa_headin_n32           --method fsa --fsa_layer head:in --n_src 32
  run $m fsa_blocks.7_isa_n32     --method fsa --fsa_layer blocks.7 --with_isa --n_src 32
  run $m fsa_blocks.7_isa_isapre  --method fsa --fsa_layer blocks.7 --with_isa --isa_target pretrain
}
one satmae & one scalemae & wait
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
echo "[$(date '+%F %T')] DONE"
