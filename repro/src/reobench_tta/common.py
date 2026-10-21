"""Shared pieces for the REOBench AID classification reproduction (settings A and B).

Model definitions are imported unchanged from REOBench/Classification (verified identical
to the official SatMAE / Scale-MAE files).
"""
import json
import os
import sys
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .paths import REOBENCH

sys.path.insert(0, str(REOBENCH / "Classification"))  # released models_vit.py / scale_models_vit.py

import models_vit  # noqa: E402  (SatMAE ViT)
import scale_models_vit  # noqa: E402  (Scale-MAE ViT)

DATA_ROOT = "/data/rs/reobench/AID"
TRAIN_DIR = f"{DATA_ROOT}/AID_clean_renamed/train"
TEST_DIR = f"{DATA_ROOT}/AID_clean_renamed/test"
CORRUPT_DIR = f"{DATA_ROOT}/corrupted_renamed"

CKPT = {
    "satmae": "/data/rs/pretrained_weights/satmae_fmow_non_temporal_rgb.pth",  # fmow_pretrain.pth, 800ep
    "scalemae": "/data/rs/pretrained_weights/scalemae-vitlarge-800.pth",
    # Checkpoint actually named in REOBench SatMAE_finetune.py: SatMAE fMoW-Sentinel group-channel ViT-L
    "satmae_sentinel": "/data/rs/pretrained_weights/pretrain-vit-large-e199.pth",
}

NUM_CLASSES = 30
IMG_SIZE = 256

# Same order as REOBench eval scripts
NOISES = [
    "gaussian_noise_new", "salt_and_pepper_noise", "gaussian_blur", "motion_blur",
    "brightness_contrast", "clouds", "haze_update_new", "data_gaps", "compression_artifacts",
    "rotate_new", "scale_image", "translate_image_new",
]

# REOBench paper Table 1 (AID, top-1 %)
PAPER = {
    "satmae": {"clean": 72.05, "brightness_contrast": 44.82, "clouds": 59.58, "compression_artifacts": 67.26,
               "data_gaps": 46.49, "gaussian_blur": 71.33, "gaussian_noise_new": 71.25, "haze_update_new": 28.31,
               "motion_blur": 63.85, "rotate_new": 69.15, "salt_and_pepper_noise": 70.45, "scale_image": 59.74,
               "translate_image_new": 66.12, "avg": 59.86},
    "scalemae": {"clean": 75.75, "brightness_contrast": 51.80, "clouds": 72.65, "compression_artifacts": 39.60,
                 "data_gaps": 43.69, "gaussian_blur": 31.65, "gaussian_noise_new": 46.31, "haze_update_new": 55.24,
                 "motion_blur": 17.49, "rotate_new": 66.15, "salt_and_pepper_noise": 47.27, "scale_image": 61.58,
                 "translate_image_new": 69.84, "avg": 50.27},
}


def class_names():
    return datasets.ImageFolder(TRAIN_DIR).classes


def build_backbone(name, drop_path, ckpt=None):
    """Build ViT-L/16 @256 and load the pretrained encoder exactly like REOBench does
    (shape-mismatched keys are dropped; pos_embed therefore falls back to the model's own
    16x16 sincos, which is also what the official SatMAE finetune code ends up doing)."""
    ck = torch.load(CKPT[ckpt or name], map_location="cpu", weights_only=False)["model"]
    if name == "satmae":
        model = models_vit.vit_large_patch16(patch_size=16, img_size=IMG_SIZE, in_chans=3,
                                             num_classes=NUM_CLASSES, drop_path_rate=drop_path, global_pool=False)
    else:
        model = scale_models_vit.vit_large_patch16(img_size=IMG_SIZE, num_classes=NUM_CLASSES,
                                                   drop_path_rate=drop_path, global_pool=False)
    sd = model.state_dict()
    for k in ["pos_embed", "patch_embed.proj.weight", "patch_embed.proj.bias", "head.weight", "head.bias"]:
        if k in ck and ck[k].shape != sd[k].shape:
            del ck[k]
    msg = model.load_state_dict(ck, strict=False)
    if ckpt == "satmae_sentinel":  # reproduce REOBench's silent strict=False load as-is
        print(f"[{ckpt}] MISSING (left random): {msg.missing_keys}")
        return model
    # Fail loudly instead of the silent strict=False of the original scripts
    assert set(msg.missing_keys) == {"pos_embed", "head.weight", "head.bias"}, msg.missing_keys
    print(f"[{name}] loaded; missing={msg.missing_keys}, unexpected={len(msg.unexpected_keys)} (decoder etc.)")
    return model


def features(model, name, x):
    """CLS-token feature after final norm (global_pool=False), shape (B, 1024)."""
    if name == "satmae":
        return model.forward_features(x)
    res = torch.ones(len(x), device=x.device)  # Scale-MAE input_res (official eval default = 1.0)
    return model.forward_features(x, input_res=res)


class CorruptedAID(Dataset):
    """All 2000 test images x 12 noises x 5 severities in one dataset.
    Label = alphabetical class index (matches ImageFolder and the numbered sub-folders)."""

    def __init__(self, transform, noises=NOISES):
        self.transform = transform
        self.items = []
        for ci, cname in enumerate(class_names()):
            for ni, noise in enumerate(noises):
                for sev in range(1, 6):
                    d = f"{CORRUPT_DIR}/{cname}/{noise}/severity_{sev}/{ci}"
                    assert os.path.isdir(d), d
                    for f in sorted(os.listdir(d)):
                        self.items.append((f"{d}/{f}", ci, ni, sev - 1))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, c, n, s = self.items[i]
        return self.transform(Image.open(p).convert("RGB")), c, n, s


@torch.no_grad()
def evaluate_all(predict, transform, batch_size=256, workers=16):
    """predict: images(B,3,H,W on cuda) -> logits. Returns dict with clean + per noise/severity acc (%).

    Accuracy per (noise, severity) is correct/2000 over all classes, which equals the
    REOBench sum_c acc_c * n_c/2000 aggregation."""
    out = {}
    clean = datasets.ImageFolder(TEST_DIR, transform=transform)
    corr = tot = 0
    for x, y in DataLoader(clean, batch_size=batch_size, num_workers=workers):
        pred = predict(x.cuda(non_blocking=True)).argmax(1).cpu()
        corr += (pred == y).sum().item()
        tot += len(y)
    out["clean"] = 100 * corr / tot

    ds = CorruptedAID(transform)
    c = defaultdict(int)
    t = defaultdict(int)
    for x, y, n, s in DataLoader(ds, batch_size=batch_size, num_workers=workers):
        pred = predict(x.cuda(non_blocking=True)).argmax(1).cpu()
        for yi, pi, ni, si in zip(y.tolist(), pred.tolist(), n.tolist(), s.tolist()):
            c[(ni, si)] += int(yi == pi)
            t[(ni, si)] += 1
    for ni, noise in enumerate(NOISES):
        sev = [100 * c[(ni, s)] / t[(ni, s)] for s in range(5)]
        assert all(t[(ni, s)] == 2000 for s in range(5))
        out[noise] = {"severity": sev, "mean": sum(sev) / 5}
    out["avg"] = sum(out[n]["mean"] for n in NOISES) / len(NOISES)
    return out


def report(name, res, path):
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    p = PAPER[name]
    lines = [f"{'metric':24s} {'ours':>7s} {'paper':>7s}   sev1..5",
             f"{'clean':24s} {res['clean']:7.2f} {p['clean']:7.2f}"]
    for n in NOISES:
        sev = " ".join(f"{v:5.1f}" for v in res[n]["severity"])
        lines.append(f"{n:24s} {res[n]['mean']:7.2f} {p[n]:7.2f}   {sev}")
    lines.append(f"{'avg (12 noises)':24s} {res['avg']:7.2f} {p['avg']:7.2f}")
    rtp = 100 * (res["clean"] - res["avg"]) / res["clean"]
    prtp = 100 * (p["clean"] - p["avg"]) / p["clean"]
    lines.append(f"{'R_TP (%)':24s} {rtp:7.2f} {prtp:7.2f}")
    txt = "\n".join(lines)
    print(txt)
    with open(path.replace(".json", ".txt"), "w") as f:
        f.write(txt + "\n")
