"""Test-time adaptation (TENT / SAR / DeYO) on REOBench AID corruptions — mild scenario.

Protocol (DeYO App. E.3 = SAR protocol, mild scenario of TENT):
  - one stream per (corruption, severity) [+ clean], 2,000 images each, shuffled, batch 64
  - model reset to the source model before every stream; online accuracy (prediction before update)
  - adapt LayerNorm affine params except the top 25% blocks and final norm (methods/params.py)
  - SGD momentum 0.9, lr = 0.001 / 64 * bs (ViT); SAR wraps it in SAM (rho 0.05)
  - thresholds scaled by ln C with C = 30: SAR E0 = 0.4 ln C, DeYO tau_Ent = 0.5 ln C, Ent0 = 0.4 ln C,
    DeYO tau_PLPD = 0.3 (mild), patch_len 4
  - fp32 with TF32 matmul (= PyTorch 1.10 default used by the DeYO/SAR environment)
Source model: frozen REOBench backbone (drop_path = 0) + linear head trained by C_ablation
  (--head_cfg P = paper protocol A0ep, PN = P + input normalization).

  python tta.py --model satmae --head_cfg P --method tent
"""
import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import common  # noqa: E402
from methods import cotta, deyo, eata, isa, lscd, sam, sar, tent  # noqa: E402
from PIL import Image  # noqa: E402

NORM = {
    "satmae": ([0.4182007312774658, 0.4214799106121063, 0.3991275727748871],
               [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]),
    "scalemae": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["satmae", "scalemae"], required=True)
p.add_argument("--head_cfg", choices=["P", "PN"], required=True)
p.add_argument("--head_seed", type=int, default=0)
p.add_argument("--method", required=True,
               choices=["none", "tent", "sar", "deyo", "eata", "cotta", "lscd", "isa_batch", "isa_image"])
p.add_argument("--bs", type=int, default=64)
p.add_argument("--lr", type=float, default=None, help="override protocol lr (0.001/64*bs); adds _lr<val> to output")
p.add_argument("--seed", type=int, default=0, help="stream order seed")
p.add_argument("--noises", nargs="*", default=common.NOISES)
p.add_argument("--sevs", nargs="*", type=int, default=[1, 2, 3, 4, 5])
p.add_argument("--no_clean", action="store_true")
p.add_argument("--cotta_ap", type=float, default=None,
               help="CoTTA augmentation threshold p_th; default = 5%% quantile of source confidence. "
                    "Official ImageNet code / EATA-C ViT: 0.1. Adds _ap<val> to output")
p.add_argument("--sar_reset", type=float, default=None,
               help="SAR model-recovery threshold e_m; default = DeYO-repo copy 0.005, official SAR / paper 0.2. "
                    "Adds _rst<val> to output")
p.add_argument("--cotta_params", choices=["all", "ln"], default="all",
               help="CoTTA update set: all weights incl. head (official) or the Tent/SAR LayerNorm set. Adds _ln")
p.add_argument("--tag", default="")
p.add_argument("--workers", type=int, default=12)
args = p.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = torch.device("cuda")
C = common.NUM_CLASSES
DEFAULT_LR = 0.01 if args.method == "cotta" else 0.001 / 64 * args.bs  # CoTTA: official ImageNet cfg (SGD 0.01, bs64)
LR = args.lr if args.lr is not None else DEFAULT_LR
if args.lr is not None:
    args.tag += f"_lr{args.lr:g}"
if args.cotta_ap is not None:
    args.tag += f"_ap{args.cotta_ap:g}"
if args.sar_reset is not None:
    args.tag += f"_rst{args.sar_reset:g}"
if args.cotta_params != "all":
    args.tag += f"_{args.cotta_params}"
OUT = f"{HERE}/outputs/{args.model}/{args.head_cfg}"
os.makedirs(OUT, exist_ok=True)

# CoTTA augments and ISA re-standardizes in pixel space, so for them PN normalization moves inside Net
# (same arithmetic); all other methods keep normalization in the data transform as before.
PIXEL_SPACE = args.method in ("cotta", "isa_batch", "isa_image")
tf = T.Compose([T.ToTensor()] + ([T.Normalize(*NORM[args.model])] if args.head_cfg == "PN" and not PIXEL_SPACE else []))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = common.build_backbone(args.model, drop_path=0.0)  # no stochastic depth during TTA
        self.head = nn.Linear(1024, C)
        hp = f"{HERE}/../C_ablation/outputs/{args.model}/{args.head_cfg}_s{args.head_seed}_head.pth"
        self.head.load_state_dict(torch.load(hp, map_location="cpu"))
        print(f"head: {hp}")
        self.in_norm = args.head_cfg == "PN" and PIXEL_SPACE
        if self.in_norm:
            self.register_buffer("mu", torch.tensor(NORM[args.model][0]).view(1, 3, 1, 1))
            self.register_buffer("sd", torch.tensor(NORM[args.model][1]).view(1, 3, 1, 1))

    def forward(self, x):
        if self.in_norm:
            x = (x - self.mu) / self.sd
        return self.head(common.features(self.backbone, args.model, x))


class Files(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        return tf(Image.open(path).convert("RGB")), y


# ---- streams ----
streams = []
if not args.no_clean:
    from torchvision import datasets
    streams.append(("clean", 0, [(p_, y) for p_, y in datasets.ImageFolder(common.TEST_DIR).samples]))
cor = common.CorruptedAID(None)
for noise in args.noises:
    ni = common.NOISES.index(noise)
    for s in args.sevs:
        streams.append((noise, s, [(p_, c) for p_, c, n, sv in cor.items if n == ni and sv == s - 1]))
assert all(len(it) == 2000 for _, _, it in streams), [len(it) for _, _, it in streams]

net = Net().to(dev)
init_state = {k: v.clone() for k, v in net.state_dict().items()}
dargs = SimpleNamespace(wandb_log=0, filter_ent=1, filter_plpd=1, reweight_ent=1, reweight_plpd=1,
                        aug_type="patch", patch_len=4, plpd_threshold=0.3,
                        occlusion_size=112, row_start=56, column_start=56)
if args.method in ("tent", "sar", "deyo"):
    _, names = tent.collect_params(net)
    print(f"adapting {len(names)} tensors, first {names[:2]} ... last {names[-2:]}")

# source-side quantities needed by EATA (Fisher), CoTTA (augmentation threshold) and ISA (pixel stats).
# All use 2,000 random clean AID *training* images (never the test split).
if args.method in ("eata", "cotta", "isa_batch", "isa_image"):
    from torchvision import datasets
    src_items = datasets.ImageFolder(common.TRAIN_DIR).samples
    src_idx = torch.randperm(len(src_items), generator=torch.Generator().manual_seed(0))[:2000].tolist()
    src_loader = DataLoader(Files([src_items[i] for i in src_idx]), batch_size=args.bs, shuffle=False,
                            num_workers=args.workers)
if args.method == "eata":
    # = DeYO repo main.py L420-446: pseudo-label CE, accumulate squared grads per batch, average
    eata.configure_model(net)
    params, names = eata.collect_params(net)
    ewc_optimizer = torch.optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    for iter_, (images, _) in enumerate(src_loader, start=1):
        outputs = net(images.to(dev))
        _, targets = outputs.max(1)
        loss = train_loss_fn(outputs, targets)
        loss.backward()
        for name, param in net.named_parameters():
            if param.grad is not None:
                if iter_ > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if iter_ == len(src_loader):
                    fisher = fisher / iter_
                fishers.update({name: [fisher, param.data.clone().detach()]})
        ewc_optimizer.zero_grad()
    del ewc_optimizer
    print(f"EATA fisher over {len(fishers)} tensors from {len(src_idx)} clean train images; adapting all LayerNorms")
if args.method == "cotta":
    # augmentation threshold p_th = 5% quantile of source confidence on source images (our reading of the
    # CoTTA supplementary, not in our PDF; official code hard-codes 0.1 for ImageNet / 0.92 for CIFAR10)
    net.eval()
    with torch.no_grad():
        conf = torch.cat([net(x.to(dev)).softmax(1).max(1)[0].cpu() for x, _ in src_loader])
    COTTA_AP = torch.quantile(conf, 0.05).item() if args.cotta_ap is None else args.cotta_ap
    print(f"CoTTA p_th = {COTTA_AP:.4f} ({'5% quantile of source confidence' if args.cotta_ap is None else 'fixed'})")
if args.method in ("isa_batch", "isa_image"):
    s1 = torch.zeros(3); s2 = torch.zeros(3); npx = 0
    for x, _ in src_loader:  # pixel space ([0,1], before any normalization)
        s1 += x.sum((0, 2, 3)); s2 += (x ** 2).sum((0, 2, 3)); npx += x.numel() // 3
    ISA_MEAN = s1 / npx
    ISA_STD = (s2 / npx - ISA_MEAN ** 2).sqrt()
    print(f"ISA train pixel mean {ISA_MEAN.tolist()} std {ISA_STD.tolist()}")


def make_adapter():
    cotta_ref.clear()  # free the previous stream's teacher/anchor copies before building a new one
    torch.cuda.empty_cache()
    net.load_state_dict(init_state)  # reset to source model
    if args.method == "none":
        net.eval()
        return lambda x, i: net(x)
    if args.method in ("isa_batch", "isa_image"):
        net.eval()
        a = isa.ISA(net, ISA_MEAN, ISA_STD, level=args.method.split("_")[1]).to(dev)
        return lambda x, i: a(x)
    if args.method == "eata":  # DeYO repo main.py L449-456 (fishers fixed, computed once above)
        eata.configure_model(net)
        params, _ = eata.collect_params(net)
        a = eata.EATA(SimpleNamespace(), net, torch.optim.SGD(params, LR, momentum=0.9), fishers, 2000.0,
                      e_margin=0.4 * math.log(C), d_margin=0.05)
        return lambda x, i: a(x)
    if args.method == "cotta":  # official imagenet/cotta.py: all weights/biases trainable, teacher output returned
        cotta.configure_model(net)
        if args.cotta_params == "ln":  # diagnostic: same LayerNorm affine set as Tent/SAR/DeYO (head and weights frozen)
            from methods.params import collect_norm_params
            keep = {id(p_) for p_ in collect_norm_params(net)[0]}
            for p_ in net.parameters():
                p_.requires_grad_(id(p_) in keep)
        params, names = cotta.collect_params(net)
        print(f"CoTTA updates {sum(p_.numel() for p_ in params) / 1e6:.2f}M params in {len(names)} tensors")
        a = cotta.CoTTA(net, torch.optim.SGD(params, lr=LR, momentum=0.9, dampening=0, weight_decay=0.0,
                                             nesterov=True), ap=COTTA_AP)
        cotta_ref[:] = [a]  # keep only the current one (appending leaked 3 ViT-L copies per stream -> OOM)
        return lambda x, i: a(x)
    if args.method == "lscd":
        lscd.configure_model(net)
        params, _ = lscd.collect_params(net)
        a = lscd.LSCD(net, torch.optim.SGD(params, LR, momentum=0.9))
        return lambda x, i: a(x)
    mod = {"tent": tent, "sar": sar, "deyo": deyo}[args.method]
    mod.configure_model(net)
    params, _ = mod.collect_params(net)
    if args.method == "tent":
        a = tent.Tent(net, torch.optim.SGD(params, LR, momentum=0.9))
        return lambda x, i: a(x)
    if args.method == "sar":
        rst = {} if args.sar_reset is None else {"reset_constant_em": args.sar_reset}
        a = sar.SAR(net, sam.SAM(params, torch.optim.SGD, lr=LR, momentum=0.9), margin_e0=0.4 * math.log(C), **rst)
        return lambda x, i: a(x)
    a = deyo.DeYO(net, dargs, torch.optim.SGD(params, LR, momentum=0.9),
                  deyo_margin=0.5 * math.log(C), margin_e0=0.4 * math.log(C))

    def run(x, i):
        out, n_ent, n_final = a(x, i, flag=True)
        stats["sel_ent"] += n_ent
        stats["sel_final"] += n_final
        return out
    return run


res = {"config": vars(args), "lr": LR}
cotta_ref = []
t0 = time.time()
for si, (noise, sev, items) in enumerate(streams):
    g = torch.Generator().manual_seed(args.seed * 1000 + si)
    dl = DataLoader(Files(items), batch_size=args.bs, shuffle=True, generator=g, num_workers=args.workers,
                    pin_memory=True)
    stats = {"sel_ent": 0, "sel_final": 0}
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
    extra = f" deyo_selected {100 * stats['sel_final'] / n:.0f}%" if args.method == "deyo" else ""
    if args.method == "cotta":
        extra = f" cotta_aug_batches {getattr(cotta_ref[-1], 'n_aug_batches', 0)}/{i + 1}"
    print(f"[{si + 1}/{len(streams)}] {noise} s{sev}: {acc:.2f}{extra} ({time.time() - t0:.0f}s)", flush=True)
    if noise == "clean":
        res["clean"] = acc
    else:
        res.setdefault(noise, {"severity": [None] * 5})["severity"][sev - 1] = acc
        if args.method == "deyo":
            res[noise].setdefault("deyo_selected", [None] * 5)[sev - 1] = stats["sel_final"] / n

done = [k for k in common.NOISES if k in res and None not in res[k]["severity"]]
for k in done:
    res[k]["mean"] = sum(res[k]["severity"]) / 5
if len(done) == len(common.NOISES):
    res["avg"] = sum(res[k]["mean"] for k in done) / len(done)
# re-resolve at save time: the experiment folder may be renamed while a run is going (temp <-> repro)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", args.model, args.head_cfg)
os.makedirs(OUT, exist_ok=True)
path = f"{OUT}/{args.method}_s{args.seed}{args.tag}.json"
with open(path, "w") as f:
    json.dump(res, f, indent=2)
print(f"saved {path} ({time.time() - t0:.0f}s)", {k: res[k] for k in ("clean", "avg") if k in res})
