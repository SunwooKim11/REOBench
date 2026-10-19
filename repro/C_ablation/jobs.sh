#!/bin/bash
# One-factor-at-a-time ablation (A -> B), 3 seeds. Prints one command per line.
# Trailing blanks are stripped: xargs -L treats them as line continuation.
declare -A SAT=( [A0]="" [norm]="--norm" [bnhead]="--bnhead" [optim]="--optim adamw_cos" [aug]="--aug"
                 [B]="--norm --bnhead --optim adamw_cos --aug" )
declare -A SCL=( [A0]="--bb_train" [norm]="--bb_train --norm" [bnhead]="--bb_train --bnhead"
                 [optim]="--bb_train --optim adamw_cos" [aug]="--bb_train --aug" [bbeval]=""
                 [B]="--norm --bnhead --optim adamw_cos --aug" )
for s in 0 1 2; do
  for n in A0 norm bnhead optim aug B; do echo "--model satmae --name $n --seed $s ${SAT[$n]}" | sed "s/ *$//"; done
  for n in A0 norm bnhead optim aug bbeval B; do echo "--model scalemae --name $n --seed $s ${SCL[$n]}" | sed "s/ *$//"; done
done
