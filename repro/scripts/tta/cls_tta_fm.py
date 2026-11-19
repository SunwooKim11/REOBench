"""Test-time adaptation of the 17 stage-1 backbones (images through the backbone, norm layers updated) on AID-C / RESISC45-C.

Source model = frozen backbone (src/reobench_tta/backbones.py, 224 px, its own normalisation) + the stage-1 linear head
(/data/rs/feats/<model>/head_<ds>.pt: feature standardised with clean-train mean / std, then W, b). Same inputs as
scripts/analysis/extract_feats.py, so `none` reproduces the stage-1 accuracy.

Protocol = scripts/tta/cls_tta.py (mild scenario of TENT / SAR / DeYO):
  one stream per (corruption, severity) + clean, shuffled (seed), batches of 64, model reset before every stream, online
  accuracy (prediction of a batch before the update made with it); SGD momentum 0.9, lr 0.001 (LayerNorm backbones) or
  0.00025 (BatchNorm backbones), times --lr_mult; E0 = 0.4 ln C.
BatchNorm backbones (ResNet-50): the BN layers keep their running statistics (eval mode) and only the affine is updated,
  i.e. the same treatment as LayerNorm. Reason: the running statistics are those of the pretraining data (ImageNet / fMoW)
  and the linear head was fitted on features computed with them; switching every BN to batch statistics changes the clean
  features (sup_in_r50 AID clean 90.8 -> 62.1 in the 3-stream check). The textbook TENT with batch statistics is kept as
  the extra method tent_bnstat (BatchNorm backbones only).
Updated parameters (tent, sar and the flip variants): affine of the norm layers except the top 25% of the blocks and the
final norm (ViT), except layer4 (ResNet-50) -- the SAR / DeYO rule, generalised over the block naming of each family.
EATA: Fisher from 2,000 clean train images, fisher_alpha 2000 (official values).
  eata         = the same parameter set as above (final norm and top blocks excluded)
  eata_allnorm = every norm layer, as in the official code. It diverges on 5 backbones (mae_in, dino_in_b16, dinov3_web_l16,
                 dinov3_sat_l16, sup_in_vitl16 -> chance accuracy even on clean): the Fisher of a few final-norm elements is
                 12.6 - 5,050, far above the stability limit of the quadratic penalty under SGD with momentum,
                 lr * 2 * alpha * F < 2 (1 + 0.9), i.e. F < 0.95 at lr 0.001. Those elements feed feature channels whose
                 clean-train std is close to 0 (down to 2e-6), which the head-input standardisation divides by.

Methods
  none, none_al : no learning; none_al = head input re-standardised per batch (align_ms), computed in the same pass
  tent, sar, eata, eata_allnorm : src/reobench_tta/tta (official code)
  entf    : entropy minimisation on the samples with entropy < E0 only (the reliability filter alone: no SAM, no weights)
  flipf   : entropy minimisation on the samples whose prediction is the same with and without head-input alignment
  bothf   : entf and flipf
  tent_al, flipf_al : prediction and loss on the aligned head input (alignment + backbone adaptation)
  tent_bnstat : TENT with batch statistics in every BN (BatchNorm backbones only)
  dsbr, dsbr_a99, dsbr_a0 : DSBR (Nielen et al., arXiv 2606.02339, Algorithm 1; no public code, implemented from the paper):
            entropy of every sample weighted by 1 / (C * P[its predicted class]), P = EMA (decay 0.9 / 0.99 / 0 = the batch
            alone) of the predicted-class frequencies, started at 1 / C at the beginning of every stream. No filter.
            Same parameter set, optimiser and lr as tent here (the paper uses Adam with the lr of TENT), so that the only
            difference from tent is the loss.
  dsbr_al : dsbr (decay 0.9) on the aligned head input
--lr_mult k adds _x<k> to the method names (none is not rerun).
Per stream: acc, ECE, signed ECE, NLL, AUROC (feat_metrics.stream_metrics); for the filter variants the share of samples used;
top_share = share of the most frequent predicted class (prediction bias; written for runs made from 2026-11-19 on).

  ~/REOBench/.venv/bin/python -u scripts/tta/cls_tta_fm.py [--models ...] [--datasets aid r45] [--methods ...]   # from repro/, GPU
Output: experiments/2026-11-17_fm_tta/<ds>/<model>/<method>.json (existing files are skipped)
"""
import argparse
import copy
import json
import math
import os
import re
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from reobench_tta import backbones, paths  # noqa: E402
from reobench_tta.tta import eata, sam, sar, tent  # noqa: E402
import extract_feats as ef  # noqa: E402
import feat_metrics as fm  # noqa: E402

OUT = paths.EXP / "2026-11-17_fm_tta"
METHODS = ["none", "tent", "entf", "flipf", "bothf", "tent_al", "flipf_al", "sar", "eata", "tent_bnstat"]  # none also writes none_al
SMALL_FIRST = ["dino_in_s16", "satdino_s16", "moco_in", "moco_fmow", "gassl_fmow", "sup_in_r50", "dino_in_b16", "satdino_b16",
               "sup_in_vitb16", "dino_in_s8", "satdino_s8", "mae_in", "satmae", "scalemae", "sup_in_vitl16", "dinov3_web_l16",
               "dinov3_sat_l16"]
dev = torch.device("cuda")


class Net(nn.Module):
    def __init__(self, name, ds):
        super().__init__()
        self.backbone, self.norm, _ = backbones.build(name)
        h = torch.load(f"{fm.FEATS}/{name}/head_{ds}.pt", map_location="cpu")
        for k in ("w", "b", "mu", "sd"):
            self.register_buffer(k, h[k].float())

    def z(self, x):  # head input, standardised with the clean-train statistics
        return (self.backbone(x) - self.mu) / self.sd

    def head(self, z):
        return z @ self.w + self.b

    def forward(self, x):
        return self.head(self.z(x))


def collect(model):
    """Norm affine params except the top 25% of the blocks and the final norm (ViT) / layer4 (ResNet-50)."""
    norms = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm))]
    idx = lambda n: re.search(r"(?:blocks\.|layer\.|encoder_layer_)(\d+)", n)
    nb = 1 + max((int(idx(n).group(1)) for n, _ in norms if idx(n)), default=-1)
    first_skipped = nb - nb // 4
    params, names = [], []
    for n, m in norms:
        b = idx(n)
        if (b and int(b.group(1)) >= first_skipped) or "layer4" in n or (not b and isinstance(m, nn.LayerNorm)):
            continue
        for k, p in m.named_parameters():
            if k in ("weight", "bias"):
                params.append(p); names.append(f"{n}.{k}")
    return params, names


def configure(net, bn_batch_stats=False):
    """Eval mode, gradients only on the norm layers. LayerNorm backbones give identical outputs in train and eval mode."""
    net.eval()
    net.requires_grad_(False)
    for m in net.modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            m.requires_grad_(True)
        if bn_batch_stats and isinstance(m, nn.BatchNorm2d):  # = tent.configure_model
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return net


def batch_align(z):
    return (z - z.mean(0)) / (z.std(0, unbiased=False) + 1e-5)


class Filtered:
    """Entropy minimisation with a sample filter; inp = raw | al, sel = all | entf | flipf | bothf."""

    def __init__(self, net, opt, inp, sel, e0):
        self.net, self.opt, self.inp, self.sel, self.e0, self.used = net, opt, inp, sel, e0, 0

    @torch.enable_grad()
    def __call__(self, x):
        z = self.net.z(x)
        lr_, la_ = self.net.head(z), self.net.head(batch_align(z))
        logits = la_ if self.inp == "al" else lr_
        ent = tent.softmax_entropy(logits)
        m = torch.ones_like(ent, dtype=torch.bool)
        if self.sel in ("entf", "bothf"):
            m &= ent.detach() < self.e0
        if self.sel in ("flipf", "bothf"):
            m &= lr_.argmax(1) == la_.argmax(1)
        self.used += int(m.sum())
        if m.any():
            ent[m].mean().backward()
            self.opt.step()
        self.opt.zero_grad()
        return logits.detach()


class DSBR:
    """Algorithm 1 of DSBR: L = mean_i H(p_i) / (C * P[yhat_i]); P updated with the batch before the loss; weights are constants."""

    def __init__(self, net, opt, inp, alpha):
        self.net, self.opt, self.inp, self.alpha = net, opt, inp, alpha
        C = net.w.shape[1]
        self.P = torch.full((C,), 1.0 / C, device=net.w.device)

    @torch.enable_grad()
    def __call__(self, x):
        z = self.net.z(x)
        logits = self.net.head(batch_align(z) if self.inp == "al" else z)
        pred = logits.detach().argmax(1)
        self.P = self.alpha * self.P + (1 - self.alpha) * torch.bincount(pred, minlength=len(self.P)).float() / len(pred)
        (tent.softmax_entropy(logits) / (len(self.P) * self.P[pred])).mean().backward()
        self.opt.step()
        self.opt.zero_grad()
        return logits.detach()


class Files(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.tf(Image.open(self.items[i][0]).convert("RGB")), self.items[i][1]


def streams_of(ds):
    """[(name, [(path, y), ...])]: clean test + 12 corruptions x 5 severities."""
    from reobench_tta import common
    test, corr = (ef.aid_items("test"), ef.aid_c_items()) if ds == "aid" else (ef.r45_items("test"), ef.r45_c_items())
    out = [("clean", [(p, y) for p, y, *_ in test])]
    for n, cname in enumerate(common.NOISES):
        for s in range(1, 6):
            out.append((f"{cname}_s{s}", [(p, y) for p, y, nn_, ss, _ in corr if nn_ == n and ss == s]))
    assert all(len(it) == len(test) for _, it in out)
    return out


def fisher_of(net0, loader):  # = scripts/tta/cls_tta.py (DeYO repo main.py L420-446): pseudo-label CE, squared grads, averaged
    net = configure(copy.deepcopy(net0))
    params, _ = eata.collect_params(net)
    opt = torch.optim.SGD(params, 0.001)
    fishers, ce = {}, nn.CrossEntropyLoss()
    for it, (x, _) in enumerate(loader, start=1):
        out = net(x.to(dev))
        ce(out, out.max(1)[1]).backward()
        for name, p in net.named_parameters():
            if p.grad is not None:
                f = p.grad.data.clone() ** 2 + (fishers[name][0] if it > 1 else 0)
                fishers[name] = [f / it if it == len(loader) else f, p.data.clone()]
        opt.zero_grad()
    return fishers


def run(name, ds, methods, args):
    out_dir = OUT / ds / name
    out_dir.mkdir(parents=True, exist_ok=True)
    net0 = Net(name, ds).to(dev).eval()
    C = net0.w.shape[1]
    e0 = 0.4 * math.log(C)
    is_bn = any(isinstance(m, nn.BatchNorm2d) for m in net0.modules())
    lr = (0.00025 if is_bn else 0.001) * args.lr_mult
    sfx = "" if args.lr_mult == 1 else f"_x{args.lr_mult:g}"
    methods = [m for m in methods if (is_bn or m != "tent_bnstat") and (not sfx or m != "none")]
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(*net0.norm)])
    todo = [m for m in methods if not (out_dir / f"{m}{sfx}.json").exists()]
    if not todo:
        return
    streams = streams_of(ds)
    items, batches, sid = [], [], []
    for si, (_, it) in enumerate(streams):
        perm = torch.randperm(len(it), generator=torch.Generator().manual_seed(args.seed * 1000 + si)) + len(items)
        items += it
        for i in range(0, len(it), args.bs):
            batches.append(perm[i:i + args.bs].tolist()); sid.append(si)
    _, names = collect(net0)
    print(f"== {name} {ds}: C {C}, lr {lr:g}, {'BN' if is_bn else 'LN'}, adapting {len(names)} tensors "
          f"({names[0]} ... {names[-1]}), {len(streams)} streams x {len(streams[0][1])} images", flush=True)
    fishers = None
    for method in todo:
        t0 = time.time()
        if method in ("eata", "eata_allnorm") and fishers is None:
            tr = ef.aid_items("train") if ds == "aid" else ef.r45_items("train")
            pick = torch.randperm(len(tr), generator=torch.Generator().manual_seed(0))[:2000].tolist()
            fishers = fisher_of(net0, DataLoader(Files([tr[i][:2] for i in pick], tf), batch_size=args.bs, num_workers=args.workers))
            sub = set(names)
            print(f"  fisher max: all norms {max(v[0].max().item() for v in fishers.values()):.3g}, adapted set "
                  f"{max(v[0].max().item() for k, v in fishers.items() if k in sub):.3g} (stability limit {1.9 / (2000 * lr):.3g})", flush=True)

        def make():
            net = copy.deepcopy(net0)
            if method == "none":
                return net, None
            configure(net, bn_batch_stats=method == "tent_bnstat")
            if method in ("eata", "eata_allnorm"):
                ps, ns = eata.collect_params(net) if method == "eata_allnorm" else collect(net)
                fz = fishers if method == "eata_allnorm" else {k: v for k, v in fishers.items() if k in set(ns)}
                a = eata.EATA(SimpleNamespace(), net, torch.optim.SGD(ps, lr, momentum=0.9), fz, 2000.0, e_margin=e0, d_margin=0.05)
                return net, a
            params, _ = collect(net)
            if method in ("tent", "tent_bnstat"):
                return net, tent.Tent(net, torch.optim.SGD(params, lr, momentum=0.9))
            if method == "sar":
                return net, sar.SAR(net, sam.SAM(params, torch.optim.SGD, lr=lr, momentum=0.9), margin_e0=e0)
            inp = "al" if method.endswith("_al") else "raw"
            if method.startswith("dsbr"):
                alpha = {"dsbr_a99": 0.99, "dsbr_a0": 0.0}.get(method, 0.9)
                return net, DSBR(net, torch.optim.SGD(params, lr, momentum=0.9), inp, alpha)
            sel = {"tent_al": "all"}.get(method, method.replace("_al", ""))
            return net, Filtered(net, torch.optim.SGD(params, lr, momentum=0.9), inp, sel, e0)

        dl = DataLoader(Files(items, tf), batch_sampler=batches, num_workers=args.workers, pin_memory=True)
        logits = {k: [[] for _ in streams] for k in (["none", "none_al"] if method == "none" else [method])}
        ys, used = [[] for _ in streams], [0] * len(streams)
        cur, net, a = -1, None, None
        for bi, (x, y) in enumerate(dl):
            si = sid[bi]
            if si != cur:
                if isinstance(a, Filtered):
                    used[cur] = a.used
                del net, a
                net, a = make()
                cur = si
            x = x.to(dev, non_blocking=True)
            if method == "none":
                with torch.no_grad():
                    z = net.z(x)
                    logits["none"][si].append(net.head(z).float().cpu()); logits["none_al"][si].append(net.head(batch_align(z)).float().cpu())
            else:
                logits[method][si].append(a(x).detach().float().cpu())
            ys[si].append(y)
        if isinstance(a, Filtered):
            used[cur] = a.used
        del net, a
        torch.cuda.empty_cache()
        for k, lg in logits.items():
            res = {"model": name, "dataset": ds, "method": k, "lr": lr, "bs": args.bs, "seed": args.seed, "streams": {}}
            for si, (sname, it) in enumerate(streams):
                L = torch.cat(lg[si])
                r = fm.stream_metrics(L, torch.cat(ys[si]))
                res["streams"][sname] = {q: r[q] for q in ("acc", "ece", "signed_ece", "nll", "auroc")}
                res["streams"][sname]["top_share"] = torch.bincount(L.argmax(1), minlength=C).max().item() / len(L)
                if k in ("entf", "flipf", "bothf", "flipf_al", "tent_al"):
                    res["streams"][sname]["used"] = used[si] / len(it)
            mean = lambda q: sum(v[q] for s_, v in res["streams"].items() if s_ != "clean") / 60
            res["clean"], res["avg"], res["avg_signed_ece"] = res["streams"]["clean"]["acc"], mean("acc"), mean("signed_ece")
            json.dump(res, open(out_dir / f"{k}{sfx}.json", "w"), indent=1)
            print(f"  {name} {ds} {k + sfx:12s} clean {res['clean']:5.1f} corrupted {res['avg']:5.1f} sECE {res['avg_signed_ece']:+.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=SMALL_FIRST)
    ap.add_argument("--datasets", nargs="*", default=["aid"])
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr_mult", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    fm.dev = torch.device("cpu")
    for ds in args.datasets:
        for m in args.models:
            run(m, ds, args.methods, args)
    print("DONE")
