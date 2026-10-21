"""Diagnose TTA collapse: per-batch accuracy, #selected, entropy, gradient norm and parameter drift.

  python diag.py --model scalemae --head_cfg P --method sar --stream clean
"""
import argparse
import math
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--head_cfg", required=True)
p.add_argument("--method", required=True, choices=["tent", "sar", "deyo"])
p.add_argument("--stream", default="clean", help="'clean' or '<noise>:<severity>'")
p.add_argument("--batches", type=int, default=12)
a = p.parse_args()
noises, sevs = (["gaussian_noise_new"], [1]) if a.stream == "clean" else ([a.stream.split(":")[0]], [int(a.stream.split(":")[1])])
sys.argv = ["tta.py", "--model", a.model, "--head_cfg", a.head_cfg, "--method", a.method, "--noises", *noises,
            "--sevs", *map(str, sevs), "--workers", "8"] + ([] if a.stream == "clean" else ["--no_clean"])

# reuse tta.py setup without running its main loop
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tta", "cls_tta.py")).read().split("res = {")[0]
exec(compile(src, "tta.py", "exec"))

from reobench_tta.tta.tent import softmax_entropy  # noqa: E402

name, sev, items = streams[0]
dl = DataLoader(Files(items), batch_size=args.bs, shuffle=True, generator=torch.Generator().manual_seed(0),
                num_workers=8)
stats = {"sel_ent": 0, "sel_final": 0}
adapt = make_adapter()
params, pnames = tent.collect_params(net)
p0 = [q.detach().clone() for q in params]
E0 = 0.4 * math.log(C)
print(f"stream {name} s{sev}, method {args.method}, lr {LR}, E0 {E0:.2f}")
print(f"{'it':>3s} {'acc':>5s} {'ent_mean':>8s} {'#ent<E0':>7s} {'grad_norm(TENT loss)':>20s} {'drift':>8s} {'max|g|param':>30s}")
for i, (x, y) in enumerate(dl):
    if i >= a.batches:
        break
    x = x.to(dev)
    # measure gradient of the plain entropy loss at the current params (diagnostic only)
    net.zero_grad()
    with torch.enable_grad():
        out0 = net(x)
        ent = softmax_entropy(out0)
        ent.mean().backward()
    gn = torch.sqrt(sum((q.grad ** 2).sum() for q in params if q.grad is not None)).item()
    big = max(((q.grad.abs().max().item(), n) for q, n in zip(params, pnames) if q.grad is not None))
    net.zero_grad()
    out = adapt(x, i)
    acc = (out.argmax(1).cpu() == y).float().mean().item() * 100
    drift = torch.sqrt(sum(((q - q0) ** 2).sum() for q, q0 in zip(params, p0))).item()
    print(f"{i:3d} {acc:5.1f} {ent.mean().item():8.4f} {(ent < E0).sum().item():7d} {gn:20.3e} {drift:8.4f} "
          f"{big[0]:.2e} {big[1].replace('backbone.', '')}")
