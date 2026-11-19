"""Reproduce Weighted Flips (Press et al., "The Entropy Enigma", ICML 2024) with the official code on ImageNet-C.

Official code: third_party/entropy_enigma (oripress/EntropyEnigma @ 04ac397, byte-identical). This driver repeats its
main() step by step with the functions imported from it (RDumb, get_unlabeled_subset, check_flips, count_flips) and the
flips-to-accuracy coefficients hard-coded there. Differences from main.py, none of which changes the estimate:
  - ResNet-50 weights from the local torchvision V1 file (= models.resnet50(pretrained=True))
  - the accuracy pass also records the confidences (for AC / DoC / ATC); it iterates the same shuffled loader once, as
    get_acc does, so the random stream is the same
  - the adaptation loop stops after the flip check at iteration 990 (main.py runs to the end of the epoch, unused)
  - DataLoader workers 10 instead of 4
Per split: target accuracy of the source model, weighted / unweighted flips over 990 RDumb iterations on the first 1,000
images, the three estimates f(WF), f2(UF), f3(UF), and the confidence baselines with ImageNet val as the source set:
  AC = mean max-softmax; DoC = acc_val - (AC_val - AC); ATC = share of scores above the threshold t with
  share_val(score > t) = acc_val, for score = max-softmax (MC) and negative entropy (NE). No temperature scaling.
Splits: val (Resize 256 + CenterCrop 224), ImageNet-C 15 corruptions x 5 severities (the paper's "IN-C {75}"), and the 4
holdout corruptions x 5 (speckle_noise, gaussian_blur, spatter, saturate; the paper fits f on these + val).

  ~/REOBench/.venv/bin/python -u scripts/verify/weighted_flips_repro.py [--splits ...]     # from repro/, GPU
Output: experiments/2026-11-19_weighted_flips_repro/<split>.json, val_scores.pt
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as trn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.join(HERE, "..", "..", "third_party", "entropy_enigma"))
from reobench_tta import paths  # noqa: E402
import main as ee  # noqa: E402  (official main.py)
from rdumb import RDumb  # noqa: E402

OUT = paths.EXP / "2026-11-19_weighted_flips_repro"
INC, INVAL = "/data/tta/ImageNet_Family/ImageNet_C", "/data/tta/ImageNet_Family/ImageNet_v1"
W = "/data/rs/pretrained_weights/imagenet/resnet50-0676ba61.pth"
MAIN = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost",
        "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]
HOLD = ["speckle_noise", "gaussian_blur", "spatter", "saturate"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
COEF = {"wf_quad": [3.55553868e-04, -3.21445023e-01, 7.56618580e+01], "uf_lin": [-0.10318227, 83.55158891],
        "uf_quad": [-1.01165493e-05, -9.59406999e-02, 8.25756446e+01]}  # = main.py


def run(split, args):
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    if split == "val":
        path, tf = INVAL, trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(MEAN, STD)])
    else:
        c, s = split.rsplit("_", 1)
        path, tf = f"{INC}/{c}/{s}", trn.Compose([trn.ToTensor(), trn.Normalize(MEAN, STD)])
    loader = torch.utils.data.DataLoader(dset.ImageFolder(path, transform=tf), batch_size=args.bs, shuffle=True,
                                         num_workers=args.workers, pin_memory=True, drop_last=True)
    device = torch.device("cuda")
    model = models.resnet50()
    model.load_state_dict(torch.load(W, map_location="cpu"))
    model = model.to(device).eval()
    t0 = time.time()
    conf, nent, ok = [], [], []
    with torch.no_grad():  # = ee.get_acc, plus the scores
        for x, y in loader:
            out = model(x.cuda(non_blocking=True))
            p = out.softmax(1)
            conf.append(p.max(1)[0].cpu()); nent.append((p * out.log_softmax(1)).sum(1).cpu()); ok.append((out.argmax(1).cpu() == y))
    conf, nent, ok = torch.cat(conf), torch.cat(nent), torch.cat(ok).float()
    acc = 100 * ok.mean().item()
    model.train()  # get_acc leaves the model in train mode
    t1 = time.time()
    rd = RDumb(model)
    stored_images, stored_labels = ee.get_unlabeled_subset(rd.model, loader)
    iterations, df, done = 0, None, False
    while not done:
        for images, labels in loader:
            if iterations == 0 or iterations == 990:
                df = ee.check_flips(rd.model, stored_images, stored_labels, iterations, device, df)
                if iterations == 990:
                    done = True
                    break
            _ = rd(images.cuda(non_blocking=True))
            iterations += 1
    wf, uf = float(ee.count_flips(df)), float(ee.count_flips(df, weighted=False))
    res = {"split": split, "n": len(ok), "acc": acc, "wf": wf, "uf": uf, "ac": 100 * conf.mean().item(),
           "est": {"wf_quad": float(np.poly1d(COEF["wf_quad"])(wf)), "uf_lin": float(np.poly1d(COEF["uf_lin"])(uf)),
                   "uf_quad": float(np.poly1d(COEF["uf_quad"])(uf))},
           "rdumb_used": [rd.num_samples_update_1, rd.num_samples_update_2], "sec": [t1 - t0, time.time() - t1]}
    if split == "val":
        torch.save({"conf": conf, "nent": nent, "acc": acc}, OUT / "val_scores.pt")
    v = torch.load(OUT / "val_scores.pt")
    for k, sc, vs in (("mc", conf, v["conf"]), ("ne", nent, v["nent"])):
        t = torch.quantile(vs.double(), 1 - v["acc"] / 100).item()
        res[f"atc_{k}"] = 100 * (sc.double() > t).float().mean().item()
    res["doc"] = v["acc"] - (100 * v["conf"].mean().item() - res["ac"])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["val"] + [f"{c}_{s}" for c in MAIN + HOLD for s in range(1, 6)])
    ap.add_argument("--seed", type=int, default=42)  # = main.py
    ap.add_argument("--bs", type=int, default=64)    # = main.py
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for sp in args.splits:
        f = OUT / f"{sp}.json"
        if f.exists():
            continue
        r = run(sp, args)
        json.dump(r, open(f, "w"), indent=1)
        print(f"{sp:22s} acc {r['acc']:5.2f} | WF {r['wf']:6.1f} UF {r['uf']:5.0f} | est wf_quad {r['est']['wf_quad']:5.2f} uf_lin {r['est']['uf_lin']:5.2f} "
              f"uf_quad {r['est']['uf_quad']:5.2f} | AC {r['ac']:5.2f} DoC {r['doc']:5.2f} ATC-MC {r['atc_mc']:5.2f} ATC-NE {r['atc_ne']:5.2f} "
              f"| {r['sec'][0]:.0f}+{r['sec'][1]:.0f}s", flush=True)
    print("DONE")
