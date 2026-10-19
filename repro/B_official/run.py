"""Setting B: official-protocol-aligned linear probing on AID (same recipe for both models).

  - Normalization from each model's pretraining:
      SatMAE   : fMoW mean/std (SatMAE util/datasets.py CustomDatasetFromImages)
      ScaleMAE : ImageNet mean/std (checkpoint args data_config / scale-mae dataloaders/fmow.py)
  - Frozen backbone in eval() (drop_path 0), CLS token + final norm
  - Head = BatchNorm1d(affine=False) + Linear, trunc_normal_(std=0.01)  (MAE / Scale-MAE main_linprobe.py)
  - Scale-MAE main_linprobe.py optimizer: AdamW(betas=(0.9, 0.95)), blr 1e-3 (lr = blr * bs / 256),
    wd 0.05 (not on bias), no warmup, half-cycle cosine stepped per iteration; batch 256, 100 epochs
    (The MAE LARS recipe was tried first: with bs 256 its lr=0.1 is ~64x smaller than at the
     16k batch it was designed for, and it barely trained - 47% val at epoch 15.)
  - Train aug: RandomResizedCrop(256, bicubic) + HorizontalFlip; eval: full 256x256 image
  - ScaleMAE input_res = 1.0 (official eval default: eval_base_resolution=1.0, no --eval_gsd)
  - bf16 autocast for the frozen backbone
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common  # noqa: E402
from timm.layers import trunc_normal_  # noqa: E402

NORM = {
    "satmae": ([0.4182007312774658, 0.4214799106121063, 0.3991275727748871],
               [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]),
    "scalemae": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["satmae", "scalemae"], required=True)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--warmup_epochs", type=int, default=0)
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--blr", type=float, default=1e-3)
p.add_argument("--weight_decay", type=float, default=0.05)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
args = p.parse_args()

torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)
dev = torch.device("cuda")
lr = args.blr * args.batch_size / 256

mean, std = NORM[args.model]
tf_train = T.Compose([T.RandomResizedCrop(common.IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
tf_eval = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
train_ds = datasets.ImageFolder(common.TRAIN_DIR, transform=tf_train)
test_ds = datasets.ImageFolder(common.TEST_DIR, transform=tf_eval)
assert train_ds.classes == common.class_names()
train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=16,
                      pin_memory=True, drop_last=True, persistent_workers=True)
test_dl = DataLoader(test_ds, batch_size=256, num_workers=16)

backbone = common.build_backbone(args.model, drop_path=0.0).to(dev).eval()
for p_ in backbone.parameters():
    p_.requires_grad = False
head = nn.Sequential(nn.BatchNorm1d(1024, affine=False, eps=1e-6), nn.Linear(1024, common.NUM_CLASSES)).to(dev)
trunc_normal_(head[1].weight, std=0.01)
nn.init.zeros_(head[1].bias)
opt = torch.optim.AdamW([{"params": [head[1].weight], "weight_decay": args.weight_decay},
                         {"params": [head[1].bias], "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))
crit = nn.CrossEntropyLoss()


def feats(x):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return common.features(backbone, args.model, x).float()


def set_lr(epoch_f):
    if epoch_f < args.warmup_epochs:
        v = lr * epoch_f / args.warmup_epochs
    else:
        v = lr * 0.5 * (1 + math.cos(math.pi * (epoch_f - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for g in opt.param_groups:
        g["lr"] = v
    return v


def predict(x):
    head.eval()
    with torch.no_grad():
        return head(feats(x))


t0 = time.time()
n_it = len(train_dl)
for ep in range(args.epochs):
    head.train()
    for i, (x, y) in enumerate(train_dl):
        cur_lr = set_lr(ep + i / n_it)
        f = feats(x.to(dev, non_blocking=True))
        loss = crit(head(f), y.to(dev, non_blocking=True))
        opt.zero_grad()
        loss.backward()
        opt.step()
    msg = f"epoch {ep + 1} loss {loss.item():.4f} lr {cur_lr:.2e} ({time.time() - t0:.0f}s)"
    if (ep + 1) % 5 == 0:
        c = t = 0
        for x, y in test_dl:
            c += (predict(x.to(dev)).argmax(1).cpu() == y).sum().item()
            t += len(y)
        msg += f" val_acc {100 * c / t:.2f}"
    print(msg, flush=True)
torch.save(head.state_dict(), f"{args.out}/{args.model}_head_ep{args.epochs}.pth")

print("evaluating clean + corrupted ...", flush=True)
res = common.evaluate_all(predict, tf_eval)
common.report(args.model, res, f"{args.out}/{args.model}_results.json")
print(f"done in {time.time() - t0:.0f}s")
