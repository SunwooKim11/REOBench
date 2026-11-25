"""Entropy minimisation run long in the TENT protocol: does TENT merge classes, and does DSBR prevent it? (fig4 / T9 of the
mid-term notes were made with a head-input affine on saved features, scripts/analysis/em_phases.py; this is the same
question with the images passed through the backbone and the norm layers updated.)

Model, parameter set, optimiser and learning rate = scripts/tta/cls_tta_fm.py (frozen backbone + stage-1 linear head; affine
of the norm layers except the top 25% of the blocks and the final norm; SGD momentum 0.9, lr 0.001 LayerNorm / 0.00025
BatchNorm backbones, BN layers keep their running statistics). One model copy per (stream, method), reset before each.
A stream = the 2,000 AID test images of one (corruption, severity). It is repeated: every epoch is a new random order,
batches of 64, until --steps updates are made. At the checkpoints (--ckpts) the current model predicts the whole stream
without updating:
  acc     accuracy (%)
  n_pred  number of classes predicted at least once (of 30)
  top     share of the most predicted class (%)
  ent     mean entropy
Methods: tent (src/reobench_tta/tta/tent.py), dsbr (cls_tta_fm.DSBR, decay 0.9); same parameters and optimiser, only the loss differs.
The images of a stream are decoded once (224 px, uint8 on the GPU), so the run is not limited by JPEG decoding.

  ~/REOBench/.venv/bin/python -u scripts/tta/cls_tta_long.py --models satmae --streams motion_blur_s3 data_gaps_s3   # from repro/, GPU
Output: experiments/2026-11-26_em_long_tent/<model>.json (finished (stream, method) pairs are skipped on restart)
"""
import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import cls_tta_fm as F  # noqa: E402
from reobench_tta import paths  # noqa: E402
from reobench_tta.tta import tent  # noqa: E402

OUT = paths.EXP / "2026-11-26_em_long_tent"
CKPTS = [0, 10, 20, 50, 100, 200, 500, 1000, 2000]
S3 = ["motion_blur_s3", "data_gaps_s3", "scale_image_s3", "gaussian_noise_new_s3", "salt_and_pepper_noise_s3", "gaussian_blur_s3",
      "brightness_contrast_s3", "clouds_s3", "haze_update_new_s3", "compression_artifacts_s3", "rotate_new_s3", "translate_image_new_s3"]
dev = torch.device("cuda")


class U8(Dataset):
    def __init__(self, items):
        self.items, self.tf = items, T.Compose([T.Resize((224, 224)), T.PILToTensor()])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.tf(Image.open(self.items[i][0]).convert("RGB")), self.items[i][1]


@torch.no_grad()
def evaluate(net, X, y, norm):
    lg = torch.cat([net(norm(X[i:i + 500])) for i in range(0, len(X), 500)]).float()
    p = lg.softmax(1)
    pred = lg.argmax(1)
    cnt = torch.bincount(pred, minlength=lg.shape[1])
    return {"acc": 100 * (pred == y).float().mean().item(), "n_pred": int((cnt > 0).sum()), "top": 100 * cnt.max().item() / len(y),
            "ent": -(p * p.clamp_min(1e-12).log()).sum(1).mean().item()}


def run(name, args):
    f = OUT / f"{name}.json"
    net0 = F.Net(name, "aid").to(dev).eval()
    is_bn = any(isinstance(m, nn.BatchNorm2d) for m in net0.modules())
    lr = 0.00025 if is_bn else 0.001
    res = json.load(open(f)) if f.exists() else {"model": name, "dataset": "aid", "lr": lr, "bs": args.bs, "ckpts": args.ckpts, "streams": {}}
    assert res["ckpts"] == args.ckpts
    mean, std = (torch.tensor(v, device=dev).view(1, 3, 1, 1) for v in net0.norm)
    norm = lambda u: (u.float() / 255 - mean) / std
    streams = dict(F.streams_of("aid"))
    for sname in args.streams:
        todo = [m for m in args.methods if m not in res["streams"].get(sname, {})]
        if not todo:
            continue
        dl = DataLoader(U8(streams[sname]), batch_size=250, num_workers=args.workers)
        X, y = map(torch.cat, zip(*[(x, yy) for x, yy in dl]))
        X, y = X.to(dev), y.to(dev)
        for method in todo:
            t0 = time.time()
            net = F.configure(copy.deepcopy(net0))
            opt = torch.optim.SGD(F.collect(net)[0], lr, momentum=0.9)
            a = tent.Tent(net, opt) if method == "tent" else F.DSBR(net, opt, "raw", 0.9)
            g = torch.Generator(device="cpu").manual_seed(args.seed)
            row = {k: [] for k in ("acc", "n_pred", "top", "ent")}
            order, pos = torch.randperm(len(X), generator=g), 0
            for t in range(max(args.ckpts) + 1):
                if t in args.ckpts:
                    for k, v in evaluate(net, X, y, norm).items():
                        row[k].append(v)
                if t == max(args.ckpts):
                    break
                if pos + args.bs > len(order):
                    order, pos = torch.randperm(len(X), generator=g), 0
                a(norm(X[order[pos:pos + args.bs].to(dev)]))
                pos += args.bs
            res["streams"].setdefault(sname, {})[method] = row
            json.dump(res, open(f, "w"))
            print(f"{name} {sname} {method}: acc " + " ".join(f"{v:.1f}" for v in row["acc"]) + " | n_pred " + " ".join(str(v) for v in row["n_pred"])
                  + f" ({time.time() - t0:.0f}s)", flush=True)
            del net, a, opt
        del X, y
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--streams", nargs="+", default=S3)
    ap.add_argument("--methods", nargs="+", default=["tent", "dsbr"])
    ap.add_argument("--ckpts", nargs="+", type=int, default=CKPTS)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    OUT.mkdir(parents=True, exist_ok=True)
    for m in args.models:
        run(m, args)
    print("DONE")
