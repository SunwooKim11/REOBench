"""Setting C: one-factor-at-a-time ablation from A (paper protocol) toward B (official protocol).

Factors (each can be switched on independently):
  --norm      : per-model pretraining normalization (SatMAE fMoW / ScaleMAE ImageNet)
  --bnhead    : BatchNorm1d(affine=False) + Linear head, trunc_normal_(0.01)   [A: plain Linear]
  --optim     : adam_wr  = Adam 1e-3, bs16, CosineAnnealingWarmRestarts(T_0=5) per iteration  [A, released code]
                adam_wr_epoch = same, but scheduler stepped once per epoch -> 5-epoch cycle    [paper App. F.1]
                adamw_cos = AdamW(0.9,0.95) 1e-3, wd 0.05, bs256, half-cycle cosine per iteration [B]
  --aug       : RandomResizedCrop(256, bicubic) + HFlip during training
  --bb_train  : backbone in train() mode during training (drop_path 0.1 active)   [A ScaleMAE only]
All settings use a bf16-autocast frozen backbone; evaluation always uses an eval-mode backbone,
so clean/corrupted eval features are cached once per (model, norm) and reused.

  python run.py --prep --model satmae [--norm]      # build feature cache
  python run.py --model satmae --name A0 --seed 0   # train head + evaluate
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import common  # noqa: E402
from timm.layers import trunc_normal_  # noqa: E402

NORM = {
    "satmae": ([0.4182007312774658, 0.4214799106121063, 0.3991275727748871],
               [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]),
    "scalemae": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["satmae", "scalemae"], required=True)
p.add_argument("--prep", action="store_true")
p.add_argument("--name", default="")
p.add_argument("--norm", action="store_true")
p.add_argument("--bnhead", action="store_true")
p.add_argument("--optim", choices=["adam_wr", "adam_wr_epoch", "adamw_cos"], default="adam_wr")
p.add_argument("--aug", action="store_true")
p.add_argument("--bb_train", action="store_true")
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--save_head", action="store_true", help="also save head weights (used by D_tta)")
args = p.parse_args()

dev = torch.device("cuda")
CACHE = f"{HERE}/cache/{args.model}_{'norm' if args.norm else 'raw'}.pt"
OUT = f"{HERE}/outputs/{args.model}"
os.makedirs(os.path.dirname(CACHE), exist_ok=True)
os.makedirs(OUT, exist_ok=True)

base_tf = [T.ToTensor()] + ([T.Normalize(*NORM[args.model])] if args.norm else [])
tf_eval = T.Compose(base_tf)
tf_train = T.Compose(([T.RandomResizedCrop(common.IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
                       T.RandomHorizontalFlip()] if args.aug else []) + base_tf)

backbone = common.build_backbone(args.model, drop_path=0.1 if args.bb_train else 0.0).to(dev).eval()
for p_ in backbone.parameters():
    p_.requires_grad = False


def feats(x):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return common.features(backbone, args.model, x).float()


def extract(ds, bs=256):
    out = [[] for _ in range(5)]
    for batch in DataLoader(ds, batch_size=bs, num_workers=16):
        out[0].append(feats(batch[0].to(dev, non_blocking=True)).half().cpu())
        for i, b in enumerate(batch[1:], 1):
            out[i].append(b)
    return [torch.cat(o) for o in out if o]


if args.prep:
    t0 = time.time()
    c = {}
    c["train_f"], c["train_y"] = extract(datasets.ImageFolder(common.TRAIN_DIR, transform=tf_eval))
    c["test_f"], c["test_y"] = extract(datasets.ImageFolder(common.TEST_DIR, transform=tf_eval))
    c["cor_f"], c["cor_y"], c["cor_n"], c["cor_s"] = extract(common.CorruptedAID(tf_eval))
    torch.save(c, CACHE)
    print(f"cached {CACHE} in {time.time() - t0:.0f}s", {k: tuple(v.shape) for k, v in c.items()})
    sys.exit()

torch.manual_seed(args.seed)
cache = torch.load(CACHE)
t0 = time.time()

# ---- head ----
if args.bnhead:
    head = nn.Sequential(nn.BatchNorm1d(1024, affine=False, eps=1e-6), nn.Linear(1024, common.NUM_CLASSES))
    lin = head[1]
else:
    head = lin = nn.Linear(1024, common.NUM_CLASSES)
if args.bnhead or args.model == "scalemae":  # A-ScaleMAE uses trunc_normal_(0.01), A-SatMAE default init
    trunc_normal_(lin.weight, std=0.01)
if args.bnhead:
    nn.init.zeros_(lin.bias)
head = head.to(dev)

# ---- optimizer ----
if args.optim == "adam_wr":
    bs = 16
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    wr = CosineAnnealingWarmRestarts(opt, T_0=5)
    step_sched = lambda it, n_it, ep: wr.step()  # noqa: E731  per iteration, as released
elif args.optim == "adam_wr_epoch":
    bs = 16
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    wr = CosineAnnealingWarmRestarts(opt, T_0=5)

    def step_sched(it, n_it, ep):
        if it == n_it - 1:  # once per epoch: "initial cycle length set to 5 epochs"
            wr.step()
else:
    bs = 256
    lr = 1e-3 * bs / 256
    opt = torch.optim.AdamW([{"params": [lin.weight], "weight_decay": 0.05},
                             {"params": [lin.bias], "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))

    def step_sched(it, n_it, ep):
        e = ep + (it + 1) / n_it
        for g in opt.param_groups:
            g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * e / args.epochs))

# ---- data ----
online = args.aug or args.bb_train
if online:
    ds = datasets.ImageFolder(common.TRAIN_DIR, transform=tf_train)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=12, pin_memory=True,
                    drop_last=args.optim == "adamw_cos", persistent_workers=True)
else:
    dl = DataLoader(TensorDataset(cache["train_f"], cache["train_y"]), batch_size=bs, shuffle=True,
                    drop_last=args.optim == "adamw_cos")
crit = nn.CrossEntropyLoss()


def head_predict(f):
    head.eval()
    with torch.no_grad():
        return torch.cat([head(f[i:i + 4096].to(dev).float()).argmax(1).cpu() for i in range(0, len(f), 4096)])


log = []
for ep in range(args.epochs):
    head.train()
    if online:
        backbone.train(args.bb_train)
    n_it = len(dl)
    for it, (x, y) in enumerate(dl):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        f = feats(x) if online else x.float()
        loss = crit(head(f), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step_sched(it, n_it, ep)
    backbone.eval()
    if (ep + 1) % 10 == 0:
        acc = 100 * (head_predict(cache["test_f"]) == cache["test_y"]).float().mean().item()
        log.append((ep + 1, acc))
        print(f"epoch {ep + 1} loss {loss.item():.4f} val_acc {acc:.2f} ({time.time() - t0:.0f}s)", flush=True)

# ---- evaluation on cached eval-mode features ----
res = {"config": vars(args), "val_log": log,
       "clean": 100 * (head_predict(cache["test_f"]) == cache["test_y"]).float().mean().item()}
pred = head_predict(cache["cor_f"])
ok = (pred == cache["cor_y"]).float()
for ni, noise in enumerate(common.NOISES):
    sev = []
    for s in range(5):
        m = (cache["cor_n"] == ni) & (cache["cor_s"] == s)
        assert m.sum() == 2000
        sev.append(100 * ok[m].mean().item())
    res[noise] = {"severity": sev, "mean": sum(sev) / 5}
res["avg"] = sum(res[n]["mean"] for n in common.NOISES) / len(common.NOISES)
path = f"{OUT}/{args.name}_s{args.seed}.json"
if args.save_head:
    torch.save(head.state_dict(), path.replace(".json", "_head.pth"))
with open(path, "w") as f:
    json.dump(res, f, indent=2)
print(f"[{args.model} {args.name} s{args.seed}] clean {res['clean']:.2f} avg {res['avg']:.2f} "
      f"({time.time() - t0:.0f}s) -> {path}")
