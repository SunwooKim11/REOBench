"""Setting A: reproduce REOBench's released AID classification protocol as faithfully as possible.

Kept from the released scripts (including the questionable parts):
  - ToTensor() only, no normalization; 256x256 input; batch 16; 100 epochs
  - Adam lr 1e-3, CosineAnnealingWarmRestarts(T_0=5) stepped *every iteration*
  - SatMAE : frozen backbone in eval(), separate nn.Linear(1024, 30) head (SatMAE_finetune.py intent)
  - ScaleMAE: frozen backbone but model.train() -> drop_path 0.1 active, model.head with
              trunc_normal_(std=0.01), input_res = 1 (scaleMAE_finetune.py)
  - final-epoch weights are evaluated (no model selection)
Fixed only where the released code cannot run or is plainly wrong:
  - `del model.head` crash -> use forward_features() for SatMAE features
  - SatMAE checkpoint: Sentinel multispectral `pretrain-vit-large-e199.pth` -> fMoW-RGB `fmow_pretrain.pth`
  - SatMAE eval loads the trained head (released SatMAE_eval.py loads SatMAE++ weights)
Speed-only change: SatMAE backbone is frozen+eval+no aug, so its features are cached once
(mathematically identical to recomputing every epoch).
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common  # noqa: E402
from timm.layers import trunc_normal_  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["satmae", "scalemae"], required=True)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--ckpt", default=None, help="e.g. satmae_sentinel (checkpoint named in REOBench code)")
p.add_argument("--tag", default="")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
args = p.parse_args()

torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.makedirs(args.out, exist_ok=True)
dev = torch.device("cuda")
BS, LR, VAL_STEP = 16, 1e-3, 5

tf = ToTensor()
train_ds = datasets.ImageFolder(common.TRAIN_DIR, transform=tf)
test_ds = datasets.ImageFolder(common.TEST_DIR, transform=tf)
assert train_ds.classes == common.class_names()
train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=8, drop_last=False)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=8)

drop_path = 0.2 if args.model == "satmae" else 0.1
model = common.build_backbone(args.model, drop_path, args.ckpt).to(dev)
for n_, p_ in model.named_parameters():
    p_.requires_grad = n_ in ("head.weight", "head.bias")
crit = nn.CrossEntropyLoss()
t0 = time.time()

if args.model == "satmae":
    model.eval()

    @torch.no_grad()
    def extract(dl):
        fs, ys = [], []
        for x, y in dl:
            fs.append(model.forward_features(x.to(dev)).float().cpu())
            ys.append(y)
        return torch.cat(fs), torch.cat(ys)

    ftr, ytr = extract(DataLoader(train_ds, batch_size=64, num_workers=8))
    fte, yte = extract(test_dl)
    print(f"features cached {tuple(ftr.shape)} in {time.time() - t0:.0f}s", flush=True)
    head = nn.Linear(1024, common.NUM_CLASSES).to(dev)  # CLS_Head, default init
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    sched = CosineAnnealingWarmRestarts(opt, T_0=5)
    feat_dl = DataLoader(TensorDataset(ftr, ytr), batch_size=BS, shuffle=True)
    for ep in range(1, args.epochs + 1):
        head.train()
        for f, y in feat_dl:
            loss = crit(head(f.to(dev)), y.to(dev))
            loss.backward()
            opt.step()
            opt.zero_grad()
            sched.step()  # per iteration, as in the released code
        if ep % VAL_STEP == 0:
            head.eval()
            with torch.no_grad():
                acc = (head(fte.to(dev)).argmax(1).cpu() == yte).float().mean().item()
            print(f"epoch {ep} loss {loss.item():.4f} val_acc {100 * acc:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    torch.save(head.state_dict(), f"{args.out}/satmae{args.tag}_head_ep{args.epochs}.pth")

    @torch.no_grad()
    def predict(x):
        return head(model.forward_features(x))

else:  # scalemae
    trunc_normal_(model.head.weight, std=0.01)
    opt = torch.optim.Adam(model.parameters(), lr=LR)  # only head has requires_grad
    sched = CosineAnnealingWarmRestarts(opt, T_0=5)
    for ep in range(1, args.epochs + 1):
        model.train()  # drop_path active in the frozen backbone, as released
        for x, y in train_dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            res = torch.ones(len(x), device=dev)
            loss = crit(model(x, input_res=res), y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            sched.step()
        print(f"epoch {ep} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        if ep % VAL_STEP == 0:
            model.eval()
            c = t = 0
            with torch.no_grad():
                for x, y in test_dl:
                    x = x.to(dev)
                    c += (model(x, input_res=torch.ones(len(x), device=dev)).argmax(1).cpu() == y).sum().item()
                    t += len(y)
            print(f"epoch {ep} val_acc {100 * c / t:.2f}", flush=True)
    torch.save({k: v for k, v in model.state_dict().items() if k.startswith("head")},
               f"{args.out}/scalemae_head_ep{args.epochs}.pth")

    @torch.no_grad()
    def predict(x):
        return model(x, input_res=torch.ones(len(x), device=x.device))

model.eval()
print("evaluating clean + corrupted ...", flush=True)
res = common.evaluate_all(predict, tf)
common.report(args.model, res, f"{args.out}/{args.model}{args.tag}_results.json")
print(f"done in {time.time() - t0:.0f}s")
