#!/bin/bash
# ImageNet-C verification of D_tta/methods against the papers (see REFERENCES.md).
# Waits for D_tta/run_new.sh, stops the DiT dummy, runs 3 jobs in parallel, summarizes, restarts DiT.
#   setsid nohup ./run_verify.sh > logs/run_verify.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
mkdir -p logs
export PY=/home1/irteam/REOBench/.venv/bin/python
echo "[$(date '+%F %T')] run_verify start (pid $$)"
(cd /home/irteam/dit_dummy && ./ditload kill) 2>/dev/null
[ -n "$NO_DIT_RESTART" ] && echo "(DiT restart left to the caller)"

run() {  # args: setting method tag -- extra flags
  setting=$1; method=$2; tag=$3; shift 4  # tag "-" = none
  [ "$tag" = "-" ] && tag=""
  out=outputs/$setting/$method$tag.json
  if [ -f "$out" ]; then echo "skip $setting $method$tag"; return; fi
  echo "[$(date '+%F %T')] start $setting $method$tag"
  $PY -u imagenetc.py --setting $setting --method $method ${tag:+--tag $tag} "$@" > logs/${setting}_$method$tag.log 2>&1
  rc=$?  # capture before $(date) resets $?
  echo "[$(date '+%F %T')] end   $setting $method$tag rc=$rc"
}
export -f run

R50_MILD="--arch r50 --norm imagenet --prep crop224 --data full --stream reset"
VIT_LS="--arch vitb_augreg --norm imagenet --prep crop224 --data full --stream label_shift"
R50_CONT="--arch r50 --norm imagenet --prep res256crop224 --data rb5000 --stream continual"
VIT_CONT="--arch vitb_augreg --norm none --prep res256crop224 --data rb5000 --stream continual --bs 50"
VIT_MILD="--prep crop224 --data full --stream reset"

{
# source checks first (cheap), including candidate checkpoints for EATA-C's ViT (target avg 39.84, gauss 12.9)
echo "deyo_mild_r50 none - -- $R50_MILD"
echo "ls_vitb none - -- $VIT_LS"
echo "cont_r50 none - -- $R50_CONT"
echo "cont_vitb none - -- $VIT_CONT"
echo "eatac_src none _augreg2_half -- --arch vitb_augreg2 --norm half $VIT_MILD"
echo "eatac_src none _augreg_half -- --arch vitb_augreg --norm half $VIT_MILD"
echo "eatac_src none _augreg2_imnet -- --arch vitb_augreg2 --norm imagenet $VIT_MILD"
# continual (long, start early)
echo "cont_vitb vida - -- $VIT_CONT"
echo "cont_r50 cotta - -- $R50_CONT"
# DeYO Table 1 (R50-BN mild)
for m in tent eata sar deyo; do echo "deyo_mild_r50 $m - -- $R50_MILD"; done
# DeYO Table 5 / SAR Table 2 (ViT label shift)
for m in tent eata sar deyo; do echo "ls_vitb $m - -- $VIT_LS"; done
} | xargs -P 3 -L 1 bash -c 'run "$@"' _

$PY summarize.py > summary.txt 2>&1
cat summary.txt
if [ -z "$NO_DIT_RESTART" ]; then
  echo "[$(date '+%F %T')] restarting DiT dummy load"
  (cd /home/irteam/dit_dummy && ./ditload run-once)
fi
echo "[$(date '+%F %T')] run_verify DONE"
