#!/bin/bash
# A0 with the paper's scheduler (CosineAnnealingWarmRestarts stepped once per epoch), 3 seeds.
for s in 0 1 2; do
  echo "--model satmae --name A0ep --seed $s --optim adam_wr_epoch"
  echo "--model scalemae --name A0ep --seed $s --bb_train --optim adam_wr_epoch"
done
