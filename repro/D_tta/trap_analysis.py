"""Does DeYO's PLPD separate correct from wrong predictions on corrupted RS images? (source model, no adaptation)

DeYO keeps samples with low entropy AND high PLPD, assuming PLPD flags predictions that rely on shape
(CPR) rather than on spurious cues (TRAP). In RS, haze/brightness/cloud cues are global and survive patch
shuffling just like texture labels do, so PLPD may not tell right from wrong. Per stream we report:
  acc        : source accuracy
  AUROC_plpd : how well PLPD ranks correct above wrong predictions (0.5 = no signal)
  AUROC_ent  : same for -entropy (the signal TENT/SAR/EATA rely on)
  sel        : fraction kept by DeYO's filter (entropy < 0.5 ln C and PLPD > 0.3, mild setting)
  acc_sel    : accuracy among kept samples (precision of the filter) vs acc_ent: kept by entropy only

  python trap_analysis.py --model satmae --head_cfg PN
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import common  # noqa: E402
from einops import rearrange  # noqa: E402

# same definitions as plpd_analysis.py (that file runs on import, so they are repeated here)
TEXTURE = ["BareLand", "Desert", "Farmland", "Forest", "Meadow", "Mountain"]
SHAPE = ["Airport", "BaseballField", "Bridge", "Church", "Playground", "Stadium", "StorageTanks", "Viaduct"]
NORM = {
    "satmae": ([0.4182007312774658, 0.4214799106121063, 0.3991275727748871],
               [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]),
    "scalemae": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


def patch_shuffle(x, patch_len):  # = methods/deyo.py aug_type='patch'
    x = rearrange(x, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=patch_len, ps2=patch_len)
    perm = torch.argsort(torch.rand(x.shape[0], x.shape[1]), dim=-1)
    x = x[torch.arange(x.shape[0]).unsqueeze(-1), perm]
    return rearrange(x, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=patch_len, ps2=patch_len)

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, choices=["satmae", "scalemae"])
p.add_argument("--head_cfg", required=True, choices=["P", "PN"])
p.add_argument("--streams", nargs="*", default=["clean", "haze_update_new:3", "haze_update_new:5",
                                                 "brightness_contrast:3", "brightness_contrast:5", "clouds:5",
                                                 "gaussian_noise_new:5", "gaussian_blur:5", "motion_blur:3"])
p.add_argument("--patch_len", type=int, default=4)
args = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True
dev = torch.device("cuda")
C = common.NUM_CLASSES
TAU_ENT, TAU_PLPD = 0.5 * math.log(C), 0.3

backbone = common.build_backbone(args.model, drop_path=0.0).to(dev).eval()
head = nn.Linear(1024, C)
head.load_state_dict(torch.load(f"{HERE}/../C_ablation/outputs/{args.model}/{args.head_cfg}_s0_head.pth"))
head = head.to(dev).eval()
tf = T.Compose([T.ToTensor()] + ([T.Normalize(*NORM[args.model])] if args.head_cfg == "PN" else []))
cor = common.CorruptedAID(None)
classes = common.class_names()


class Files(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return tf(Image.open(self.items[i][0]).convert("RGB")), self.items[i][1]


def auroc(score, label):  # P(score_correct > score_wrong), ties count half
    pos = score[label == 1]
    neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = torch.cat([pos, neg])
    r = s.argsort().argsort().double() + 1
    return ((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))).item()


from torchvision import datasets  # noqa: E402
results = {}
print(f"### {args.model} / {args.head_cfg} (patch_len {args.patch_len})")
print(f"{'stream':24s} {'acc':>5s} {'AUROC_plpd':>10s} {'AUROC_ent':>9s} {'sel':>5s} {'acc_sel':>7s} {'acc_ent':>7s}"
      f"  {'sel_tex':>7s} {'sel_shp':>7s}")
torch.manual_seed(0)
for st in args.streams:
    if st == "clean":
        items = datasets.ImageFolder(common.TEST_DIR).samples
    else:
        noise, sev = st.split(":")
        ni = common.NOISES.index(noise)
        items = [(p_, c) for p_, c, n, s in cor.items if n == ni and s == int(sev) - 1]
    ys, cor_, ents, plpds = [], [], [], []
    with torch.no_grad():
        for x, y in DataLoader(Files(items), batch_size=128, num_workers=8):
            x = x.to(dev)
            prob = head(common.features(backbone, args.model, x)).softmax(1)
            yhat = prob.argmax(1)
            prob_s = head(common.features(backbone, args.model, patch_shuffle(x, args.patch_len))).softmax(1)
            plpds.append((prob.gather(1, yhat[:, None]) - prob_s.gather(1, yhat[:, None]))[:, 0].cpu())
            ents.append(-(prob * prob.clamp_min(1e-12).log()).sum(1).cpu())
            cor_.append((yhat.cpu() == y).long())
            ys.append(y)
    y, ok, ent, plpd = map(torch.cat, (ys, cor_, ents, plpds))
    keep_ent = ent < TAU_ENT
    keep = keep_ent & (plpd > TAU_PLPD)
    tex = torch.tensor([classes[i] in TEXTURE for i in y.tolist()])
    shp = torch.tensor([classes[i] in SHAPE for i in y.tolist()])
    r = {"acc": ok.float().mean().item(), "auroc_plpd": auroc(plpd, ok), "auroc_ent": auroc(-ent, ok),
         "sel": keep.float().mean().item(),
         "acc_sel": ok[keep].float().mean().item() if keep.any() else float("nan"),
         "acc_ent": ok[keep_ent].float().mean().item() if keep_ent.any() else float("nan"),
         "sel_texture": keep[tex].float().mean().item(), "sel_shape": keep[shp].float().mean().item()}
    results[st] = r
    print(f"{st:24s} {100 * r['acc']:5.1f} {r['auroc_plpd']:10.3f} {r['auroc_ent']:9.3f} {100 * r['sel']:4.0f}% "
          f"{100 * r['acc_sel']:6.1f}% {100 * r['acc_ent']:6.1f}%  {100 * r['sel_texture']:6.0f}% {100 * r['sel_shape']:6.0f}%",
          flush=True)
os.makedirs(f"{HERE}/outputs/trap", exist_ok=True)
with open(f"{HERE}/outputs/trap/{args.model}_{args.head_cfg}.json", "w") as f:
    json.dump(results, f, indent=2)
