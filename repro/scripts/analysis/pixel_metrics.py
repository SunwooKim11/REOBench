"""Image-space distance between each corrupted image and its clean original: PSNR, MS-SSIM, LPIPS (AlexNet, VGG).

Question: does a learned perceptual distance (LPIPS, Zhang et al. CVPR 2018) order the corruptions by how much they
hurt the classifiers better than the signal-processing ones (PSNR, MS-SSIM)?
Sets (all 224 x 224, values in [0, 1]; LPIPS gets [-1, 1] as its reference code expects):
  natural_vs_rs : the ImageNet-C-style sets of natural_vs_rs.py (ImageNet 5,000 / AID 2,000 / RESISC45 1,996 images,
                  7 corruptions x 5 severities), same image lists and loading
  reobench      : AID-C (2,000 test images) and RESISC45-C (6,300) x 12 REOBench corruptions x 5 severities against the
                  clean test image, both resized whole to 224 as in extract_feats.py
LPIPS: vendored lpips 0.1.4 (third_party/lpips, weights v0.1), spatially averaged; backbone ImageNet weights from
torchvision, cached under /data/rs/pretrained_weights/torch_hub.

  ~/REOBench/.venv/bin/python scripts/analysis/pixel_metrics.py     # from repro/, GPU
Output: experiments/2026-11-16_backbone_feats/pixel_metrics/{natural_vs_rs.json, reobench.json}
"""
import json
import os
import sys

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.join(HERE, "..", "..", "third_party", "lpips"))
sys.path.insert(0, HERE)
torch.hub.set_dir("/data/rs/pretrained_weights/torch_hub")
import lpips  # noqa: E402
from reobench_tta import common, paths  # noqa: E402
import natural_vs_rs as nv  # noqa: E402
import extract_feats as ef  # noqa: E402

OUT = paths.EXP / "2026-11-16_backbone_feats" / "pixel_metrics"
dev = torch.device("cuda")
NETS = {n: lpips.LPIPS(net=n, verbose=False).to(dev).eval() for n in ("alex", "vgg")}


class Paths(Dataset):
    def __init__(self, ps):
        self.ps, self.tf = ps, T.Compose([T.Resize((224, 224)), T.ToTensor()])

    def __len__(self):
        return len(self.ps)

    def __getitem__(self, i):
        return self.tf(Image.open(self.ps[i]).convert("RGB"))


def load(ps):
    return torch.cat([x for x in DataLoader(Paths(ps), batch_size=256, num_workers=32)])


@torch.no_grad()
def dist(a, b):  # a, b: N x 3 x 224 x 224 in [0, 1] (CPU) -> per-image metrics
    out = {"psnr": [], "ms_ssim": [], "lpips_alex": [], "lpips_vgg": []}
    for i in range(0, len(a), 128):
        x, y = a[i:i + 128].to(dev), b[i:i + 128].to(dev)
        out["psnr"].append(10 * torch.log10(1 / ((x - y) ** 2).mean((1, 2, 3)).clamp_min(1e-10)))
        out["ms_ssim"].append(nv.ms_ssim(x, y))
        for n, net in NETS.items():
            out[f"lpips_{n}"].append(net(2 * x - 1, 2 * y - 1).flatten())
    return {k: torch.cat(v).mean().item() for k, v in out.items()}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "natural_vs_rs.json"
    if not f.exists():
        res = {}
        for ds, items in nv.sets().items():
            clean = nv.load_all(ds, items, "clean", 0)
            for c in nv.CORRS:
                for s in range(1, 6):
                    res[f"{ds}/{c}_{s}"] = dist(clean, nv.load_all(ds, items, c, s))
                print(ds, c, {k: round(v, 3) for k, v in res[f"{ds}/{c}_5"].items()}, flush=True)
        json.dump(res, open(f, "w"), indent=1)
    f = OUT / "reobench.json"
    if not f.exists():
        res = {}
        for ds, test, corr in (("aid", ef.aid_items("test"), ef.aid_c_items()), ("r45", ef.r45_items("test"), ef.r45_c_items())):
            clean = load([p for p, *_ in test])
            for n, cname in enumerate(common.NOISES):
                for s in range(1, 6):
                    it = [(p, i) for p, _, nn, ss, i in corr if nn == n and ss == s]
                    res[f"{ds}/{cname}_{s}"] = dist(clean[[i for _, i in it]], load([p for p, _ in it]))
                print(ds, cname, {k: round(v, 3) for k, v in res[f"{ds}/{cname}_5"].items()}, flush=True)
        json.dump(res, open(f, "w"), indent=1)
    print("DONE")
