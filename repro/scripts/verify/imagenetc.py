"""Verify our TTA implementations (D_tta/methods, the exact code used on REOBench) on ImageNet-C against the papers.

The method code is shared with D_tta/tta.py; only the backbone, data and stream protocol come from each paper's
official repo. Reference numbers: docs/verify/REFERENCES.md. Output: outputs/{setting}/{method}{tag}.json

Protocols
  --stream reset      : one stream per corruption, model reset before each (DeYO/SAR "mild"; EATA-C "single-domain")
  --stream label_shift: reset per corruption, DeYO repo's pre-computed online label-shift order (ir 500000, no shuffle)
  --stream continual  : 15 corruptions in a row, reset only before the first (CoTTA / ViDA)
  --data full         : all 50,000 images per corruption (DeYO/SAR/EATA)
  --data rb5000       : robustbench imagenet_test_image_ids.txt (5,000, file order) (CoTTA/ViDA)
  --prep crop224      : CenterCrop(224) on the 224px ImageNet-C files (DeYO repo te_transforms_imageC)
  --prep res256crop224: Resize(256)+CenterCrop(224) (robustbench 'Res256Crop224', CoTTA/ViDA)
Normalization (--norm) is applied inside the model so that CoTTA/ViDA augmentations act on [0,1] pixels; for the
other methods this is the same arithmetic as normalizing in the transform.

  python imagenetc.py --arch vitb_augreg --norm imagenet --stream label_shift --method sar
"""
import argparse
import json
import math
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))  # assets/ next to this script
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # repro/src
from reobench_tta import paths  # noqa: E402
from reobench_tta.tta import cotta, deyo, eata, sam, sar, tent, vida  # noqa: E402

IMNET_C = "/data/tta/ImageNet_Family/ImageNet_C"
IMNET_VAL = "/data/tta/ImageNet_Family/ImageNet_v1"  # clean val, class folders (EATA Fisher)
RB_IDS = f"{HERE}/assets/imagenet_test_image_ids.txt"  # copied from robustbench (CoTTA/ViDA repos)
LS_DIR = f"{HERE}/assets"  # DeYO repo dataset/*.npy label-shift orders
VIDA_CKPT = "/data/rs/pretrained_weights/vida_imagenet_vitb_source.pt"
CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
               'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
               'jpeg_compression']
NORMS = {"imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), "half": ([0.5] * 3, [0.5] * 3),
         "none": ([0.0] * 3, [1.0] * 3)}

p = argparse.ArgumentParser()
p.add_argument("--arch", required=True, choices=["r50", "vitb_augreg", "vitb_augreg2"])
p.add_argument("--norm", required=True, choices=list(NORMS))
p.add_argument("--prep", default="crop224", choices=["crop224", "res256crop224"])
p.add_argument("--data", default="full", choices=["full", "rb5000"])
p.add_argument("--stream", default="reset", choices=["reset", "label_shift", "continual"])
p.add_argument("--order", nargs="*", default=CORRUPTIONS)
p.add_argument("--method", required=True, choices=["none", "tent", "eata", "sar", "deyo", "cotta", "vida"])
p.add_argument("--bs", type=int, default=64)
p.add_argument("--lr", type=float, default=None)
p.add_argument("--seed", type=int, default=2024, help="DeYO repo default seed (also picks the label-shift file)")
p.add_argument("--plpd", type=float, default=None, help="DeYO tau_PLPD (repo: 0.3 mild, 0.2 wild)")
p.add_argument("--cotta_ap", type=float, default=0.1)
p.add_argument("--vida_fix_hka", action="store_true")
p.add_argument("--vida_ckpt", default=VIDA_CKPT, help="'none' = random adapter init")
p.add_argument("--setting", required=True, help="output subfolder name, e.g. deyo_mild_r50")
p.add_argument("--tag", default="")
p.add_argument("--workers", type=int, default=16)
p.add_argument("--limit", type=int, default=None, help="smoke test: images per corruption")
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.10 default (DeYO/SAR/CoTTA/ViDA environments)
torch.backends.cudnn.allow_tf32 = True
dev = torch.device("cuda")
C = 1000
IS_VIT = args.arch.startswith("vit")
cotta.IMG_SIZE = 224
OUT = f"{paths.VERIFY_IMAGENETC}/outputs/{args.setting}"
os.makedirs(OUT, exist_ok=True)

# ---- learning rates of the official repos ----
if args.lr is not None:
    LR = args.lr
elif args.method == "cotta":
    LR = 0.01                      # CoTTA cfgs/10orders/cotta (R50); EATA-C uses 0.005 for ViT
elif args.method == "vida":
    LR = None                      # two groups, below
else:
    LR = 0.001 / 64 * args.bs if IS_VIT else 0.00025  # DeYO main.py (bs >= 32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        if args.arch == "r50":     # torchvision IMAGENET1K_V1 = resnet50-19c8e357 (SAR/DeYO) = robustbench Standard_R50
            import torchvision
            self.backbone = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        else:
            import timm
            tag = {"vitb_augreg": "augreg_in21k_ft_in1k", "vitb_augreg2": "augreg2_in21k_ft_in1k"}[args.arch]
            self.backbone = timm.create_model(f"vit_base_patch16_224.{tag}", pretrained=True)
        self.register_buffer("mu", torch.tensor(NORMS[args.norm][0]).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor(NORMS[args.norm][1]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mu) / self.sd)


PREP = {"crop224": T.Compose([T.CenterCrop(224), T.ToTensor()]),
        "res256crop224": T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])}[args.prep]


class Files(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        return self.tf(datasets.folder.default_loader(path)), y


def corruption_items(c):
    root = f"{IMNET_C}/{c}/5"
    if args.data == "full":
        items = datasets.ImageFolder(root).samples
        if args.stream == "label_shift":  # DeYO main.py: val_dataset.set_specific_subset(np.load(...))
            f = (f"{LS_DIR}/total_100000_ir_500000_class_order_shuffle_yes.npy" if args.seed == 2021 else
                 f"{LS_DIR}/seed{args.seed}_total_100000_ir_500000_class_order_shuffle_yes.npy")
            items = [items[i] for i in np.load(f).astype(int).tolist()]
    else:  # robustbench make_custom_dataset: ids file order, labels from sorted class folders
        classes = sorted(os.listdir(root))
        cidx = {k: i for i, k in enumerate(classes)}
        items = [(f"{root}/{line.strip()}", cidx[line.split("/")[0]]) for line in open(RB_IDS) if line.strip()]
    return items[:args.limit] if args.limit else items


net = Net().to(dev)
if args.method == "vida":
    vida.configure_model(net, r=1, r2=128, ckpt=None if args.vida_ckpt == "none" else args.vida_ckpt,
                         key_map=lambda k: "backbone." + k[len("module."):], allow_missing=("mu", "sd"))
from copy import deepcopy  # noqa: E402
pristine = deepcopy(net)  # configure_model() drops BN running stats, so reset = fresh copy, not load_state_dict

fishers = None
if args.method == "eata":  # DeYO main.py L420-446: 2000 random clean val images, pseudo-label CE, mean sq. grads
    fisher_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])  # DeYO te_transforms
    val_items = datasets.ImageFolder(IMNET_VAL).samples
    random.shuffle(val_items)
    fl = DataLoader(Files(val_items[:2000], fisher_tf), batch_size=args.bs, shuffle=True, num_workers=args.workers)
    eata.configure_model(net)
    params, _ = eata.collect_params(net)
    ewc_optimizer = torch.optim.SGD(params, 0.001)
    fishers = {}
    loss_fn = nn.CrossEntropyLoss().cuda()
    for iter_, (images, _) in enumerate(fl, start=1):
        outputs = net(images.to(dev))
        _, targets = outputs.max(1)
        loss_fn(outputs, targets).backward()
        for name, param in net.named_parameters():
            if param.grad is not None:
                fisher = param.grad.data.clone().detach() ** 2
                if iter_ > 1:
                    fisher = fisher + fishers[name][0]
                if iter_ == len(fl):
                    fisher = fisher / iter_
                fishers.update({name: [fisher, param.data.clone().detach()]})
        ewc_optimizer.zero_grad()
    del ewc_optimizer
    print(f"EATA fisher over {len(fishers)} tensors")

plpd = args.plpd if args.plpd is not None else (0.2 if args.stream == "label_shift" else 0.3)
dargs = SimpleNamespace(wandb_log=0, filter_ent=1, filter_plpd=1, reweight_ent=1, reweight_plpd=1,
                        aug_type="patch", patch_len=4, plpd_threshold=plpd,
                        occlusion_size=112, row_start=56, column_start=56)
stats = {}


def make_adapter():
    global net
    net = deepcopy(pristine)
    if args.method == "none":
        net.eval()
        return lambda x, i: net(x)
    if args.method == "eata":
        eata.configure_model(net)
        params, _ = eata.collect_params(net)
        a = eata.EATA(SimpleNamespace(), net, torch.optim.SGD(params, LR, momentum=0.9), fishers, 2000.0,
                      e_margin=0.4 * math.log(C), d_margin=0.05)
        return lambda x, i: a(x)
    if args.method == "cotta":
        cotta.configure_model(net)
        params, _ = cotta.collect_params(net)
        a = cotta.CoTTA(net, torch.optim.SGD(params, lr=LR, momentum=0.9, dampening=0, weight_decay=0.0,
                                             nesterov=True), ap=args.cotta_ap)
        stats["obj"] = a
        return lambda x, i: a(x)
    if args.method == "vida":  # cfgs/vit/vida.yaml: LR 5e-7, ViDALR 2e-7, MT 0.999, MT_ViDA 0.8; unc_thr 0.2
        net.train()
        mp, vp = vida.collect_params(net)
        opt = vida.setup_optimizer_vida(mp, vp, args.lr or 5e-7, 2e-7)
        a = vida.ViDA(net, opt, ema=0.999, ema_vida=0.8, unc_thr=0.2, fix_hka=args.vida_fix_hka)
        stats["obj"] = a
        return lambda x, i: a(x)
    mod = {"tent": tent, "sar": sar, "deyo": deyo}[args.method]
    mod.configure_model(net)
    params, names = mod.collect_params(net)
    print(f"adapting {len(names)} tensors: {names[:2]} ... {names[-2:]}")
    if args.method == "tent":
        a = tent.Tent(net, torch.optim.SGD(params, LR, momentum=0.9))
        return lambda x, i: a(x)
    if args.method == "sar":
        a = sar.SAR(net, sam.SAM(params, torch.optim.SGD, lr=LR, momentum=0.9), margin_e0=0.4 * math.log(C))
        return lambda x, i: a(x)
    a = deyo.DeYO(net, dargs, torch.optim.SGD(params, LR, momentum=0.9),
                  deyo_margin=0.5 * math.log(C), margin_e0=0.4 * math.log(C))

    def run(x, i):
        out, n_ent, n_final = a(x, i, flag=True)
        stats["sel"] = stats.get("sel", 0) + n_final
        return out
    return run


res = {"config": vars(args), "lr": LR, "acc": {}}
t0 = time.time()
adapt = None
for ci, c in enumerate(args.order):
    items = corruption_items(c)
    shuffle = args.stream == "reset"  # DeYO if_shuffle=True; label_shift and robustbench loaders: file order
    g = torch.Generator().manual_seed(args.seed + ci)
    dl = DataLoader(Files(items, PREP), batch_size=args.bs, shuffle=shuffle, generator=g,
                    num_workers=args.workers, pin_memory=True)
    if args.stream != "continual" or adapt is None:
        stats.pop("sel", None)
        adapt = make_adapter()
    correct = n = 0
    for i, (x, y) in enumerate(dl):
        x = x.to(dev, non_blocking=True)
        if args.method == "none":
            with torch.no_grad():
                out = adapt(x, i)
        else:
            out = adapt(x, i)
        correct += (out.argmax(1).cpu() == y).sum().item()
        n += len(y)
    acc = 100 * correct / n
    res["acc"][c] = acc
    extra = f" deyo_sel {100 * stats['sel'] / n:.0f}%" if "sel" in stats else ""
    if args.method == "vida":
        u = stats["obj"].unc_log
        extra = f" unc mean {np.mean(u):.3f} (>=thr {100 * np.mean(np.array(u) >= 0.2):.0f}%)"
    print(f"[{ci + 1}/{len(args.order)}] {c}: acc {acc:.2f} err {100 - acc:.2f}{extra} ({time.time() - t0:.0f}s)",
          flush=True)

res["avg_acc"] = sum(res["acc"].values()) / len(res["acc"])
res["avg_err"] = 100 - res["avg_acc"]
path = f"{OUT}/{args.method}{args.tag}.json"
if args.limit is None:
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
print(f"avg acc {res['avg_acc']:.2f} err {res['avg_err']:.2f} -> {path} ({time.time() - t0:.0f}s)")
