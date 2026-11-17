"""Stage 1 of the backbone-pair study: frozen linear-probe features of every backbone on AID(-C) and RESISC45(-C).

Each image is resized (whole image, no crop) to 224 and normalized with the backbone's own constants; the feature is
the one its official linear evaluation uses (src/reobench_tta/backbones.py). Everything later (linear heads, accuracy,
ECE, drift decomposition, head-input alignment) is computed from these files: scripts/analysis/feat_metrics.py.

Sets (corrupted sets keep "src" = row of the clean test image they were made from):
  aid_train 7,999 / aid_test 2,000 / aid_c 120,000 (12 corruptions x 5 severities)
  r45_train 18,900 / r45_val 6,300 / r45_test 6,300 / r45_c 378,000 (torchgeo splits, 45 classes; RESISC45-C from
  scripts/tools/make_corrupted_cls.py)
  es_{r45,ucm}_{train,val,test}: EarthShift 19 shared classes (RESISC45 7,915 / 2,726 / 2,659, UC Merced 1,142 / 379 / 379;
  not in the default --sets, pass them explicitly)

  ~/REOBench/.venv/bin/python -u scripts/analysis/extract_feats.py [--models ...] [--sets ...]   # from repro/
Output: /data/rs/feats/<model>/<set>.pt = {feats fp16 (N, D), y, corr, sev, src}; existing files are skipped.
"""
import argparse
import os
import sys
import time

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import backbones, common  # noqa: E402

OUT = "/data/rs/feats"
R45 = "/data/rs/resisc45"
MODELS = ["satmae", "scalemae", "mae_in", "moco_in", "moco_fmow", "gassl_fmow", "dino_in_s16", "satdino_s16",
          "dino_in_s8", "satdino_s8", "dino_in_b16", "satdino_b16", "dinov3_web_l16", "dinov3_sat_l16",
          "sup_in_r50", "sup_in_vitb16", "sup_in_vitl16"]
SETS = ["aid_train", "aid_test", "aid_c", "r45_train", "r45_val", "r45_test", "r45_c"]


def aid_items(split):
    from torchvision import datasets
    return [(p, y, -1, 0, i) for i, (p, y) in enumerate(datasets.ImageFolder(f"{common.DATA_ROOT}/AID_clean_renamed/{split}").samples)]


def aid_c_items():
    clean = {os.path.basename(p): i for p, _, _, _, i in aid_items("test")}
    return [(p, c, n, s + 1, clean[os.path.basename(p)]) for p, c, n, s in common.CorruptedAID(None).items]


R45_CLASSES = sorted(d for d in os.listdir(f"{R45}/NWPU-RESISC45") if os.path.isdir(f"{R45}/NWPU-RESISC45/{d}"))


def r45_items(split):
    names = [l.strip() for l in open(f"{R45}/resisc45-{split}.txt") if l.strip()]
    cls = lambda f: f.rsplit("_", 1)[0]
    return [(f"{R45}/NWPU-RESISC45/{cls(f)}/{f}", R45_CLASSES.index(cls(f)), -1, 0, i) for i, f in enumerate(names)]


def r45_c_items():
    test = r45_items("test")
    out = []
    for n, c in enumerate(common.NOISES):
        for s in range(1, 6):
            for p, y, _, _, i in test:
                cn = R45_CLASSES[y]
                out.append((f"{R45}/RESISC45-C/{cn}/{c}/severity_{s}/{y}/{os.path.splitext(os.path.basename(p))[0]}.jpg",
                            y, n, s, i))
    return out


UCM = "/data/rs/ucmerced"


def es_items(ds, split):  # EarthShift 19-class RESISC45 <-> UC Merced splits (scripts/tools/make_rs_splits.py)
    root, img = (R45, "NWPU-RESISC45") if ds == "r45" else (UCM, "UCMerced_LandUse/Images")
    lines = [l.split() for l in open(f"{root}/earthshift19/{split}.txt") if l.strip()]
    return [(f"{root}/{img}/{f}", int(y), -1, 0, i) for i, (f, y) in enumerate(lines)]


ITEMS = {"aid_train": lambda: aid_items("train"), "aid_test": lambda: aid_items("test"), "aid_c": aid_c_items,
         "r45_train": lambda: r45_items("train"), "r45_val": lambda: r45_items("val"), "r45_test": lambda: r45_items("test"),
         "r45_c": r45_c_items,
         **{f"es_{d}_{sp}": (lambda d=d, sp=sp: es_items(d, sp)) for d in ("r45", "ucm") for sp in ("train", "val", "test")}}


class Files(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.tf(Image.open(self.items[i][0]).convert("RGB")), i


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    items = {s: ITEMS[s]() for s in args.sets}
    for s, it in items.items():
        assert all(os.path.exists(p) for p, *_ in it[:: max(1, len(it) // 500)]), s
        print(f"{s}: {len(it)} images", flush=True)
    for name in args.models:
        todo = [s for s in args.sets if not os.path.exists(f"{OUT}/{name}/{s}.pt")]
        if not todo:
            continue
        model, norm, dim = backbones.build(name)
        model = model.cuda()
        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(*norm)])
        os.makedirs(f"{OUT}/{name}", exist_ok=True)
        for s in todo:
            t0 = time.time()
            it = items[s]
            feats = torch.empty(len(it), dim, dtype=torch.float16)
            with torch.no_grad():
                for x, idx in DataLoader(Files(it, tf), batch_size=args.bs, num_workers=args.workers, pin_memory=True):
                    feats[idx] = model(x.cuda(non_blocking=True)).half().cpu()
            col = lambda k: torch.tensor([r[k] for r in it])
            torch.save({"feats": feats, "y": col(1), "corr": col(2), "sev": col(3), "src": col(4),
                        "model": name, "norm": norm, "set": s}, f"{OUT}/{name}/{s}.pt.part")
            os.replace(f"{OUT}/{name}/{s}.pt.part", f"{OUT}/{name}/{s}.pt")
            print(f"[{time.strftime('%F %T')}] {name} {s}: {len(it)} x {dim} in {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
