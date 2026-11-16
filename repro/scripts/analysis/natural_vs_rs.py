"""Question (b): does the *same* corruption (same function, same parameters) act differently on natural and RS images?

Corruptions: the 7 ImageNet-C types with a REOBench counterpart (scripts/tools/make_imagenetc_style.py), severities 1-5.
  imagenet : 5 val images per class (5,000; fixed seed), clean = Resize 256 + CenterCrop 224 (ImageNet-C geometry),
             corrupted = the stored ImageNet-C files of the same images
  aid      : all 2,000 test images, 224 px, ImageNet-C parameters (/data/rs/imagenetc_style/aid)
  r45      : 2,000 test images (stratified, fixed seed) of /data/rs/imagenetc_style/r45
Per (dataset, corruption, severity):
  pixel   : PSNR and MS-SSIM (5 scales, Gaussian window 11 / sigma 1.5, per channel; ImageNet-C's design target was
            MS-SSIM = 1 - s/10, github.com/hendrycks/robustness/issues/12) between clean and corrupted, 0-1 range
  feature : per backbone, rel_drift, frac_common, cka, cos (feat_metrics.py definitions; standardized with the clean
            features of that dataset)
  accuracy: AID / RESISC45 with the stage-1 linear heads (/data/rs/feats/<model>/head_*.pt); ImageNet only for the
            supervised anchors (their own ImageNet head)

  ~/REOBench/.venv/bin/python scripts/analysis/natural_vs_rs.py [--models ...]   # from repro/, GPU
Output: experiments/2026-11-16_backbone_feats/natural_vs_rs/{pixel.json, <model>.json, summary.txt}
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reobench_tta import backbones, paths  # noqa: E402
from extract_feats import MODELS  # noqa: E402
from feat_metrics import cka  # noqa: E402

CORRS = ["gaussian_noise", "impulse_noise", "gaussian_blur", "motion_blur", "jpeg_compression", "brightness", "contrast"]
IN_VAL, IN_C, ST = "/data/tta/ImageNet_Family/ImageNet_v1", "/data/tta/ImageNet_Family/ImageNet_C", "/data/rs/imagenetc_style"
OUT = paths.EXP / "2026-11-16_backbone_feats" / "natural_vs_rs"
dev = torch.device("cuda")


def sets():
    g = torch.Generator().manual_seed(0)
    wn = sorted(d for d in os.listdir(IN_VAL) if os.path.isdir(f"{IN_VAL}/{d}"))  # skip the val tar next to the class folders
    inet = []
    for y, w in enumerate(wn):
        fs = sorted(f for f in os.listdir(f"{IN_VAL}/{w}") if f.endswith(".JPEG"))
        inet += [(f"{w}/{fs[i]}", y) for i in torch.randperm(len(fs), generator=g)[:5].tolist()]
    aid_cls = sorted(os.listdir(f"{ST}/aid/clean"))
    aid = [(f"{c}/{f}", y) for y, c in enumerate(aid_cls) for f in sorted(os.listdir(f"{ST}/aid/clean/{c}"))]
    r45_cls = sorted(os.listdir("/data/rs/resisc45/NWPU-RESISC45"))
    r45_all = [(f"{c}/{f}", r45_cls.index(c)) for c in sorted(os.listdir(f"{ST}/r45/clean")) for f in sorted(os.listdir(f"{ST}/r45/clean/{c}"))]
    r45 = []
    for y in range(45):
        ix = [i for i, (_, yy) in enumerate(r45_all) if yy == y]
        r45 += [r45_all[ix[i]] for i in torch.randperm(len(ix), generator=g)[: round(2000 * len(ix) / len(r45_all))].tolist()]
    return {"imagenet": inet, "aid": aid, "r45": r45}


def path(ds, rel, c, s):
    if ds == "imagenet":
        return f"{IN_VAL}/{rel}" if c == "clean" else f"{IN_C}/{c}/{s}/{rel}"
    return f"{ST}/{ds}/clean/{rel}" if c == "clean" else f"{ST}/{ds}/{c}/{s}/{rel}"


class Files(Dataset):
    def __init__(self, ds, items, c, s):
        self.ds, self.items, self.c, self.s = ds, items, c, s
        clean_in = ds == "imagenet" and c == "clean"
        self.tf = T.Compose(([T.Resize(256), T.CenterCrop(224)] if clean_in else [T.Resize((224, 224))]) + [T.ToTensor()])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.tf(Image.open(path(self.ds, self.items[i][0], self.c, self.s)).convert("RGB"))


def load_all(ds, items, c, s):
    return torch.cat([x for x in DataLoader(Files(ds, items, c, s), batch_size=256, num_workers=32)])


def _gauss(sigma=1.5, size=11):
    x = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2)); g /= g.sum()
    return (g[:, None] * g[None]).to(dev)[None, None].repeat(3, 1, 1, 1)


W = None


def ms_ssim(a, b):  # a, b: B x 3 x H x W in [0, 1]; Wang et al. 2003 weights
    global W
    W = _gauss() if W is None else W
    wts = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=dev)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for k in range(5):
        mu_a, mu_b = F.conv2d(a, W, groups=3), F.conv2d(b, W, groups=3)
        saa = F.conv2d(a * a, W, groups=3) - mu_a ** 2
        sbb = F.conv2d(b * b, W, groups=3) - mu_b ** 2
        sab = F.conv2d(a * b, W, groups=3) - mu_a * mu_b
        cs = ((2 * sab + C2) / (saa + sbb + C2)).mean((1, 2, 3))
        if k == 4:
            vals.append(((2 * mu_a * mu_b + C1) / (mu_a ** 2 + mu_b ** 2 + C1) * (2 * sab + C2) / (saa + sbb + C2)).mean((1, 2, 3)))
        else:
            vals.append(cs)
            a, b = F.avg_pool2d(a, 2), F.avg_pool2d(b, 2)
    v = torch.stack(vals, 1).clamp_min(1e-6)
    return (v ** wts).prod(1)


def drift(fc, fh):
    mu, sd = fc.mean(0), fc.std(0) + 1e-6
    zc, zh = (fc - mu) / sd, (fh - mu) / sd
    d = zh - zc
    dn = (d ** 2).sum(1)
    return {"rel_drift": (dn.mean() / (zc ** 2).sum(1).mean()).item(),
            "frac_common": ((d.mean(0) ** 2).sum() / dn.mean()).item(),
            "cka": cka(fc, fh), "cos": F.cosine_similarity(fc, fh, dim=1).mean().item()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets()
    print({k: len(v) for k, v in S.items()}, flush=True)
    conds = [("clean", 0)] + [(c, s) for c in CORRS for s in range(1, 6)]
    # images once (uint8 in RAM: 3 sets x 36 conditions x ~2-5k x 150 KB)
    IM = {ds: {f"{c}_{s}": (load_all(ds, it, c, s) * 255).round().to(torch.uint8) for c, s in conds} for ds, it in S.items()}
    print("images loaded", flush=True)
    pf = OUT / "pixel.json"
    if not pf.exists():
        pix = {}
        for ds in S:
            cl = IM[ds]["clean_0"]
            for c, s in conds[1:]:
                ps, ms = [], []
                for i in range(0, len(cl), 256):
                    a = cl[i:i + 256].to(dev).float() / 255; b = IM[ds][f"{c}_{s}"][i:i + 256].to(dev).float() / 255
                    ps.append(10 * torch.log10(1 / ((a - b) ** 2).mean((1, 2, 3)).clamp_min(1e-10))); ms.append(ms_ssim(a, b))
                pix[f"{ds}/{c}_{s}"] = {"psnr": torch.cat(ps).mean().item(), "ms_ssim": torch.cat(ms).mean().item()}
        json.dump(pix, open(pf, "w"), indent=1)
    for name in args.models:
        f = OUT / f"{name}.json"
        if f.exists():
            continue
        model, norm, _ = backbones.build(name)
        model = model.to(dev)
        mu, sd = torch.tensor(norm[0], device=dev).view(1, 3, 1, 1), torch.tensor(norm[1], device=dev).view(1, 3, 1, 1)

        @torch.no_grad()
        def fe(x):
            return torch.cat([model((x[i:i + 256].to(dev).float() / 255 - mu) / sd).float() for i in range(0, len(x), 256)])
        heads = {ds: torch.load(f"/data/rs/feats/{name}/head_{ds}.pt") for ds in ("aid", "r45")}
        res = {"model": name}
        for ds, it in S.items():
            y = torch.tensor([t[1] for t in it], device=dev)
            if ds == "imagenet":
                cls = (lambda z: model.head(z)) if hasattr(model, "classify") else None
            else:
                h = heads[ds]
                cls = lambda z, h=h: ((z - h["mu"].to(dev)) / h["sd"].to(dev)) @ h["w"].to(dev) + h["b"].to(dev)
            fc = fe(IM[ds]["clean_0"])
            res[f"{ds}/clean"] = {"acc": (cls(fc).argmax(1) == y).float().mean().item() * 100 if cls else None}
            for c, s in conds[1:]:
                fh = fe(IM[ds][f"{c}_{s}"])
                r = drift(fc, fh)
                r["acc"] = (cls(fh).argmax(1) == y).float().mean().item() * 100 if cls else None
                res[f"{ds}/{c}_{s}"] = r
        json.dump(res, open(f, "w"), indent=1)
        print(name, {ds: round(sum(res[f"{ds}/{c}_{s}"]["frac_common"] for c, s in conds[1:]) / 35, 3) for ds in S}, flush=True)
        del model; torch.cuda.empty_cache()
