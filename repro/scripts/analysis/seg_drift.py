"""Stage 1 for segmentation: drift decomposition and calibration of the existing Potsdam models (no adaptation).

Models: REOBench Potsdam B models (SatMAE B, ScaleMAE B; backbone fine-tuned, UperNet), scripts/seg/train.sh.
Streams: clean + 8 corruptions x 5 severities, 2,016 test patches each, batch 8, source model in eval mode.
Per stream:
  mIoU (5 classes; clutter and 255 ignored, = seg_tta.py), pixel ECE (15 bins over every valid pixel), signed pixel ECE
  (mean confidence - pixel accuracy), pixel AUROC of max-softmax (256 random valid pixels per image)
  per layer (backbone blocks 7 / 11 / 15 / 23 = the four UperNet inputs, and the classifier input
  decode_head.conv_seg:in), on 64 fixed random token / pixel positions per image (same positions in every stream),
  standardized with the clean-stream mean / std of that layer:
    rel_drift, frac_common, cka, cos  (definitions as scripts/analysis/feat_metrics.py; "common" = mean drift over all
    images and positions of the stream, i.e. a per-channel shift shared by the whole stream)

  TIMM_FUSED_ATTN=0 /opt/conda/envs/segtta/bin/python scripts/analysis/seg_drift.py --model scalemae   # from repro/
Output: experiments/2026-11-16_backbone_feats/seg/<model>_B.json
"""
import argparse
import json
import os
import sys

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import paths  # noqa: E402
os.chdir(paths.REOBENCH_SEG)
sys.path.insert(0, str(paths.REOBENCH_SEG))
from mmseg.models import build_segmentor  # noqa: E402

T = "/data/rs/reobench/Potsdam_extracted/test"
CORR = ["brightness_contrast", "clouds", "compression_artifacts", "gaussian_blur", "gaussian_noise", "haze",
        "motion_blur", "salt_and_pepper_noise"]
C, IGNORE, K = 5, 5, 64
LAYERS = ["backbone.blocks.7", "backbone.blocks.11", "backbone.blocks.15", "backbone.blocks.23", "decode_head.conv_seg:in"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["scalemae", "satmae"])
ap.add_argument("--sevs", nargs="*", type=int, default=[1, 2, 3, 4, 5])
args = ap.parse_args()
dev = torch.device("cuda")
cfg = mmcv.Config.fromfile(f"configs/repro/{args.model}_B_12ep.py")
model = build_segmentor(cfg.model)
load_checkpoint(model, f"{paths.SEG_BASELINE}/work_dirs/{args.model}_B_12ep/latest.pth", map_location="cpu", logger="silent")
model = model.to(dev).eval()
MEAN = torch.tensor(cfg.img_norm_cfg["mean"]).view(1, 3, 1, 1).to(dev)
STD = torch.tensor(cfg.img_norm_cfg["std"]).view(1, 3, 1, 1).to(dev)
META = [dict(img_shape=(512, 512, 3), ori_shape=(512, 512, 3), pad_shape=(512, 512, 3), flip=False)]
NAMES = sorted(os.listdir(f"{T}/clean"))
OUT = paths.EXP / "2026-11-16_backbone_feats" / "seg"
OUT.mkdir(parents=True, exist_ok=True)


class Files(Dataset):
    def __init__(self, d):
        self.d = d

    def __len__(self):
        return len(NAMES)

    def __getitem__(self, i):
        img = np.asarray(Image.open(f"{self.d}/{NAMES[i]}").convert("RGB"), dtype=np.float32)
        lab = np.asarray(Image.open(f"{T}/clean_ann/{NAMES[i]}").convert("L")).astype(np.int64)
        lab[lab == 255] = IGNORE
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(lab), i


cap = {}


def hook(name):
    def f(m, inp, out):
        h = inp[0] if name.endswith(":in") else out
        if h.dim() == 4:  # B, C, H, W -> B, HW, C
            h = h.flatten(2).transpose(1, 2)
        cap[name] = h
    return f


mods = dict(model.named_modules())
for L in LAYERS:
    mod = mods[L.replace(":in", "")]
    (mod.register_forward_pre_hook(lambda m, i, n=L: hook(n)(m, i, None)) if L.endswith(":in")
     else mod.register_forward_hook(hook(L)))
g = torch.Generator().manual_seed(0)
POS = {}  # layer -> (num_images, K) fixed positions, drawn at first sight of the layer's token count


@torch.no_grad()
def run_stream(d):
    inter = torch.zeros(C, device=dev); union = torch.zeros(C, device=dev)
    nb = torch.zeros(15, device=dev); sc = torch.zeros(15, device=dev); sk = torch.zeros(15, device=dev)
    confs, oks, toks = [], [], {L: [] for L in LAYERS}
    gp = torch.Generator(device="cpu").manual_seed(1)
    for x, y, idx in DataLoader(Files(d), batch_size=8, num_workers=8):
        x, y = x.to(dev), y.to(dev)
        logit = model.encode_decode((x - MEAN) / STD, META * len(x))
        p = logit.softmax(1)
        conf, pred = p.max(1)
        v = y != IGNORE
        for k in range(C):
            inter[k] += ((pred == k) & (y == k) & v).sum(); union[k] += (((pred == k) | (y == k)) & v).sum()
        cv, okv = conf[v], (pred[v] == y[v]).float()
        b = torch.clamp((cv * 15).long(), max=14)
        nb += torch.bincount(b, minlength=15); sc += torch.bincount(b, cv, minlength=15); sk += torch.bincount(b, okv, minlength=15)
        for i in range(len(x)):  # 256 random valid pixels per image for AUROC
            vi = v[i].flatten().nonzero().flatten()
            if len(vi):
                sel = vi[torch.randint(len(vi), (256,), generator=gp).to(dev)]
                confs.append(conf[i].flatten()[sel]); oks.append((pred[i] == y[i]).flatten()[sel].float())
        for L in LAYERS:
            h = cap[L]
            if L not in POS:
                POS[L] = torch.randint(h.shape[1], (len(NAMES), K), generator=g)
            pos = POS[L][idx].to(dev)
            toks[L].append(torch.gather(h, 1, pos[..., None].expand(-1, -1, h.shape[2])).reshape(-1, h.shape[2]).half())
    iou = inter / union.clamp_min(1)
    ece = ((sk - sc).abs().sum() / nb.sum()).item()
    conf, ok = torch.cat(confs), torch.cat(oks)
    _, inv, cnt = torch.unique(conf, return_inverse=True, return_counts=True)
    avg = (torch.cumsum(cnt, 0) - (cnt - 1) / 2.0).float()
    npos, nneg = ok.sum(), (1 - ok).sum()
    auroc = (((avg[inv] * ok).sum() - npos * (npos + 1) / 2) / (npos * nneg)).item()
    r = {"miou": 100 * iou.mean().item(), "pix_acc": 100 * (sk.sum() / nb.sum()).item(), "ece": ece,
         "signed_ece": ((sc.sum() - sk.sum()) / nb.sum()).item(), "auroc": auroc}
    return r, {L: torch.cat(v) for L, v in toks.items()}


def cka(x, y):
    x = x - x.mean(0); y = y - y.mean(0)
    return ((y.T @ x).norm() ** 2 / ((x.T @ x).norm() * (y.T @ y).norm())).item()


res = {"model": args.model, "streams": {}}
r, clean_tok = run_stream(f"{T}/clean")
res["streams"]["clean"] = r
stats = {L: (t.float().mean(0), t.float().std(0) + 1e-6) for L, t in clean_tok.items()}
print("clean", {k: round(v, 3) for k, v in r.items()}, flush=True)
for c in CORR:
    for s in args.sevs:
        r, tok = run_stream(f"{T}/{c}/severity_{s}")
        for L in LAYERS:
            mu, sd = stats[L]
            a, b = clean_tok[L].float(), tok[L].float()
            za, zb = (a - mu) / sd, (b - mu) / sd
            dlt = zb - za
            dn = (dlt ** 2).sum(1)
            short = L.replace("backbone.", "").replace("decode_head.conv_seg:in", "cls_in")
            r[f"{short}/rel_drift"] = (dn.mean() / (za ** 2).sum(1).mean()).item()
            r[f"{short}/frac_common"] = ((dlt.mean(0) ** 2).sum() / dn.mean()).item()
            r[f"{short}/cka"] = cka(a, b)
            r[f"{short}/cos"] = F.cosine_similarity(a, b, dim=1).mean().item()
        res["streams"][f"{c}_s{s}"] = r
        print(f"{c}_s{s}", {k: round(v, 3) for k, v in r.items() if "/" not in k or "frac_common" in k}, flush=True)
json.dump(res, open(OUT / f"{args.model}_B.json", "w"), indent=1)
