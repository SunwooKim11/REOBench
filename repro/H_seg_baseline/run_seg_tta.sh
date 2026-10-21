#!/bin/bash
# Potsdam segmentation TTA, mild protocol, setting B models. 2 jobs in parallel; CoTTA (14 teacher views, ~0.5 s/img)
# only on clean + severities 3 and 5.
#   setsid nohup ./run_seg_tta.sh > logs/run_seg_tta.log 2>&1 < /dev/null &
cd "$(dirname "$0")/seg"
export PYTHONPATH=. TIMM_FUSED_ATTN=0
(cd /home/irteam/dit_dummy && ./ditload kill) >/dev/null 2>&1
run() {  # model method [extra args...]
  m=$1; meth=$2; shift 2
  echo "[$(date '+%F %T')] start $m $meth $*"
  /opt/conda/envs/segtta/bin/python -u ../seg_tta.py --model $m --method $meth "$@" > ../logs/tta_${m}_${meth}$(echo "$*" | tr -d ' -').log 2>&1
  rc=$?
  echo "[$(date '+%F %T')] end   $m $meth $* rc=$rc"
}
export -f run
{
for m in scalemae satmae; do
  echo "$m none"; echo "$m bnadapt"; echo "$m isa"; echo "$m tent"; echo "$m tent --bn batch"; echo "$m sar"
done
for m in scalemae satmae; do echo "$m cotta --bs 4 --sevs 3 5"; done
} | xargs -P 2 -L 1 bash -c 'run "$@"' _
(cd /home/irteam/dit_dummy && ./ditload run-once) >/dev/null 2>&1
echo "[$(date '+%F %T')] DONE"
