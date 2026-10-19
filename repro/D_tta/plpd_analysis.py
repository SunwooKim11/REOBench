"""Test the DeYO "texture" hypothesis on AID with the source model (no adaptation).

DeYO selects samples whose prediction drops when the image is patch-shuffled:
  PLPD = p(y_hat | x) - p(y_hat | x'),  x' = patch_shuffle(x, patch_len)
and keeps samples with PLPD > tau (0.3 mild). Hypothesis: AID scene classes defined by texture keep
their prediction under shuffling (small PLPD), so DeYO filters most of them out.

Class groups are fixed here *before* looking at the results:
  texture: BareLand, Desert, Farmland, Forest, Meadow, Mountain
  shape  : Airport, BaseballField, Bridge, Church, Playground, Stadium, StorageTanks, Viaduct
  other  : remaining 16 classes

  python plpd_analysis.py --model satmae --head_cfg PN
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import common  # noqa: E402

TEXTURE = ["BareLand", "Desert", "Farmland", "Forest", "Meadow", "Mountain"]
SHAPE = ["Airport", "BaseballField", "Bridge", "Church", "Playground", "Stadium", "StorageTanks", "Viaduct"]
NORM = {
    "satmae": ([0.4182007312774658, 0.4214799106121063, 0.3991275727748871],
               [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]),
    "scalemae": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, choices=["satmae", "scalemae"])
p.add_argument("--head_cfg", required=True, choices=["P", "PN"])
p.add_argument("--patch_lens", nargs="*", type=int, default=[2, 4, 8])
p.add_argument("--tau", type=float, default=0.3)
args = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True
dev = torch.device("cuda")


def patch_shuffle(x, patch_len):  # same as methods/deyo.py (aug_type='patch'); 256 is divisible by 2/4/8
    x = rearrange(x, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=patch_len, ps2=patch_len)
    perm = torch.argsort(torch.rand(x.shape[0], x.shape[1]), dim=-1)
    x = x[torch.arange(x.shape[0]).unsqueeze(-1), perm]
    return rearrange(x, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=patch_len, ps2=patch_len)


backbone = common.build_backbone(args.model, drop_path=0.0).to(dev).eval()
head = nn.Linear(1024, common.NUM_CLASSES)
head.load_state_dict(torch.load(f"{HERE}/../C_ablation/outputs/{args.model}/{args.head_cfg}_s0_head.pth"))
head = head.to(dev).eval()
tf = T.Compose([T.ToTensor()] + ([T.Normalize(*NORM[args.model])] if args.head_cfg == "PN" else []))
ds = datasets.ImageFolder(common.TEST_DIR, transform=tf)
classes = ds.classes

torch.manual_seed(0)
rows = {pl: [] for pl in args.patch_lens}  # (label, correct, p_orig, p_shuf, pred_kept)
with torch.no_grad():
    for x, y in DataLoader(ds, batch_size=128, num_workers=8):
        x = x.to(dev)
        prob = head(common.features(backbone, args.model, x)).softmax(1)
        yhat = prob.argmax(1)
        p_o = prob.gather(1, yhat[:, None])[:, 0]
        for pl in args.patch_lens:
            prob_s = head(common.features(backbone, args.model, patch_shuffle(x, pl))).softmax(1)
            p_s = prob_s.gather(1, yhat[:, None])[:, 0]
            for i in range(len(y)):
                rows[pl].append((y[i].item(), (yhat[i].cpu() == y[i]).item(), p_o[i].item(), p_s[i].item(),
                                 (prob_s[i].argmax() == yhat[i]).item()))

out = {"config": vars(args), "per_class": {}, "groups": {}}
print(f"\n### {args.model} / {args.head_cfg}: PLPD = p(y_hat|x) - p(y_hat|shuffled x); 'sel' = PLPD > {args.tau}")
hdr = " ".join(f"PLPD@{pl} sel@{pl} keep@{pl}" for pl in args.patch_lens)
print(f"{'class':18s} {'grp':7s} {'acc':>5s}  {hdr}")
for ci, c in enumerate(classes):
    grp = "texture" if c in TEXTURE else "shape" if c in SHAPE else "other"
    d = {"group": grp}
    cells = []
    for pl in args.patch_lens:
        r = [t for t in rows[pl] if t[0] == ci]
        plpd = [po - ps for _, _, po, ps, _ in r]
        d[pl] = {"plpd": sum(plpd) / len(plpd), "sel": sum(v > args.tau for v in plpd) / len(plpd),
                 "keep_pred": sum(t[4] for t in r) / len(r)}
        d["acc"] = sum(t[1] for t in r) / len(r)
        cells.append(f"{d[pl]['plpd']:7.3f} {100 * d[pl]['sel']:5.0f}% {100 * d[pl]['keep_pred']:5.0f}%")
    out["per_class"][c] = d
    print(f"{c:18s} {grp:7s} {100 * d['acc']:5.1f}  " + "  ".join(cells))

print("\ngroup means (unweighted over classes)")
for grp in ["texture", "shape", "other"]:
    cs = [c for c in classes if out["per_class"][c]["group"] == grp]
    g = {}
    for pl in args.patch_lens:
        g[pl] = {k: sum(out["per_class"][c][pl][k] for c in cs) / len(cs) for k in ["plpd", "sel", "keep_pred"]}
    out["groups"][grp] = g
    print(f"{grp:8s} (n={len(cs):2d}) " + "  ".join(
        f"pl{pl}: PLPD {g[pl]['plpd']:.3f}, sel {100 * g[pl]['sel']:.0f}%, pred kept {100 * g[pl]['keep_pred']:.0f}%"
        for pl in args.patch_lens))
os.makedirs(f"{HERE}/outputs/plpd", exist_ok=True)
with open(f"{HERE}/outputs/plpd/{args.model}_{args.head_cfg}.json", "w") as f:
    json.dump(out, f, indent=2)
