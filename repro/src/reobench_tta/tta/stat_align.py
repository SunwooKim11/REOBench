"""Feature statistics alignment (FSA) at a chosen depth — the "depth sweep" tool. No parameter is learned.

At one position of the network, the activations of the current test batch are re-standardized per channel to the
source statistics of the same position:
    h' = (h - mu_test) / sigma_test * sigma_src + mu_src        (mu, sigma per channel)
mu_test / sigma_test come from the test batch itself (all images and all tokens/pixels of the batch), mu_src /
sigma_src from clean source (training) images, computed once with `source_stats`.

Positions are given as module names; the module's output is aligned through a forward hook ("name:in" aligns the
module's *input* instead, e.g. "head:in" = the feature fed to a linear classifier). For a ViT block the
output is (B, N, D) and the channel is D (statistics over B x N tokens). For a (B, C, H, W) output the channel is
C (statistics over B x H x W). Input-level alignment is ISA (isa.py); decoder-BN alignment is "bnadapt".
"""
import torch


def _flat(h):
    """-> (M, C) view with channels last."""
    if h.dim() == 3:                       # B, N, D tokens
        return h.reshape(-1, h.shape[-1])
    if h.dim() == 4:                       # B, C, H, W maps
        return h.permute(0, 2, 3, 1).reshape(-1, h.shape[1])
    if h.dim() == 2:                       # B, D vectors
        return h
    raise ValueError(f"unsupported activation shape {tuple(h.shape)}")


def _apply(h, mu_s, sd_s, eps=1e-6):
    f = _flat(h).float()
    mu_t, sd_t = f.mean(0), f.std(0).clamp_min(eps)
    if h.dim() == 4:
        shp = (1, -1, 1, 1)
    else:
        shp = (1,) * (h.dim() - 1) + (-1,)
    out = (h.float() - mu_t.view(shp)) / sd_t.view(shp) * sd_s.view(shp) + mu_s.view(shp)
    return out.to(h.dtype)


@torch.no_grad()
def source_stats(model, module_names, batches, forward):
    """Per-channel mean/std of each named module's output over clean source batches.
    forward(model, x) runs the model on a batch x; batches yields input tensors (already on the right device)."""
    mods = dict(model.named_modules())
    s1, s2, n, hooks = {}, {}, {}, []

    def mk(name):
        def hook(_m, _i, out):
            f = _flat(out[0] if isinstance(out, tuple) else out).double()
            s1[name] = s1.get(name, 0) + f.sum(0)
            s2[name] = s2.get(name, 0) + (f ** 2).sum(0)
            n[name] = n.get(name, 0) + f.shape[0]
        return hook

    for nm in module_names:
        base, is_in = nm.split(":")[0], nm.endswith(":in")
        if is_in:
            hooks.append(mods[base].register_forward_pre_hook(lambda m, i, nm=nm: mk(nm)(m, None, i[0])))
        else:
            hooks.append(mods[base].register_forward_hook(mk(nm)))
    try:
        for x in batches:
            forward(model, x)
    finally:
        for h in hooks:
            h.remove()
    stats = {}
    for nm in module_names:
        mu = s1[nm] / n[nm]
        sd = (s2[nm] / n[nm] - mu ** 2).clamp_min(0).sqrt()
        stats[nm] = (mu.float(), sd.float())
    return stats


class StatAlign:
    """Attach alignment hooks to the named modules (normally one, for a depth sweep). Use as a context manager or
    call .remove(). Statistics are moved to the activation's device on first use."""

    def __init__(self, model, stats):
        mods = dict(model.named_modules())
        self.hooks = []
        for nm, (mu, sd) in stats.items():
            base = nm.split(":")[0]
            if nm.endswith(":in"):
                f = self._mk(mu, sd)
                self.hooks.append(mods[base].register_forward_pre_hook(lambda m, i, f=f: (f(m, None, i[0]),) + tuple(i[1:])))
            else:
                self.hooks.append(mods[base].register_forward_hook(self._mk(mu, sd)))

    @staticmethod
    def _mk(mu, sd):
        cache = {}

        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.device not in cache:
                cache[h.device] = (mu.to(h.device), sd.to(h.device))
            h2 = _apply(h, *cache[h.device])
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.remove()
