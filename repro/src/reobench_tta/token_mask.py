"""Forward a frozen ViT backbone (backbones.build) with some patch tokens excluded.

    feats = token_mask.forward(name, model, x, keep)     # keep: (B, gh, gw) bool, False = excluded patch

"Excluded" = no token attends to that patch token in any block (key mask). For the CLS feature this is the same as
dropping the tokens from the sequence, but it works with a different number of excluded tokens per image. The DINOv3
feature [CLS, patch mean] averages the kept patches only. ResNets are not supported (no tokens).

How the mask reaches the attention of each family:
  timm ViT (mae_in, dino_in_*), REOBench MIM (satmae, scalemae; timm blocks) : F.scaled_dot_product_attention is wrapped
      while forward() runs (timm fused attention; TIMM_FUSED_ATTN must not be 0)
  SatDINO (own Attention with an explicit softmax)  : Attention.forward replaced by a masked copy
  torchvision ViT (nn.MultiheadAttention)           : key_padding_mask added by a forward pre-hook
  DINOv3 (transformers)                             : additive attention_mask passed through the model kwargs
Token order: [CLS, patches] for all but SatDINO [CLS, patches, GSD register] and DINOv3 [CLS, registers, patches].
check(): the feature must not change when the pixels of excluded patches change, and an all-True mask must give model(x).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbones

_KEEP = None  # (B, tokens) bool of the batch being forwarded
_sdpa = F.scaled_dot_product_attention


def _masked_sdpa(q, k, v, attn_mask=None, **kw):
    if _KEEP is not None:
        assert attn_mask is None and q.shape[0] == _KEEP.shape[0] and q.shape[-2] == _KEEP.shape[1], (q.shape, _KEEP.shape)
        attn_mask = _KEEP[:, None, None, :]
    return _sdpa(q, k, v, attn_mask=attn_mask, **kw)


def _satdino_attention(self, x):  # vision_transformer_satdino.Attention.forward + key mask
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    if _KEEP is not None:
        attn = attn.masked_fill(~_KEEP[:, None, None, :], float("-inf"))
    attn = self.attn_drop(attn.softmax(dim=-1))
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(x)), attn


def _mha_hook(module, args, kwargs):
    return args, ({**kwargs, "key_padding_mask": ~_KEEP} if _KEEP is not None else kwargs)


def supported(name):
    return backbones.SPECS[name][0] not in ("resnet_moco", "resnet_plain", "tv_resnet")


def patch(name):
    return 8 if name.endswith("_s8") else 16


def forward(name, model, x, keep):
    """x: (B, 3, H, W) normalized, keep: (B, H / patch, W / patch) bool. Returns the linear-probe feature (B, dim)."""
    global _KEEP
    fam = backbones.SPECS[name][0]
    B = len(x)
    keep = keep.reshape(B, -1)
    assert keep.shape[1] == (x.shape[2] // patch(name)) * (x.shape[3] // patch(name)) and keep.any(1).all()
    one = lambda n: torch.ones(B, n, dtype=torch.bool, device=x.device)
    if fam == "dinov3":
        full = torch.cat([one(model.skip), keep], 1)
        bias = torch.zeros(B, 1, 1, full.shape[1], device=x.device).masked_fill(~full[:, None, None, :], torch.finfo(torch.float32).min)
        h = model.net(pixel_values=x, attention_mask=bias).last_hidden_state
        hp = h[:, model.skip:]
        return torch.cat([h[:, 0], (hp * keep[..., None]).sum(1) / keep.sum(1, keepdim=True)], 1)
    _KEEP = torch.cat([one(1), keep, one(1)] if fam == "satdino" else [one(1), keep], 1)
    hooks, restore = [], None
    try:
        if fam in ("vit_timm", "reobench_mim"):
            F.scaled_dot_product_attention = _masked_sdpa
        elif fam == "satdino":
            cls = type(model.blocks[0].attn)
            restore = (cls, cls.forward)
            cls.forward = _satdino_attention
        elif fam == "tv_vit":
            hooks = [m.register_forward_pre_hook(_mha_hook, with_kwargs=True) for m in model.modules() if isinstance(m, nn.MultiheadAttention)]
        else:
            raise ValueError(f"{name}: no tokens to exclude")
        return model(x)
    finally:
        _KEEP = None
        F.scaled_dot_product_attention = _sdpa
        if restore:
            restore[0].forward = restore[1]
        for h in hooks:
            h.remove()


@torch.no_grad()
def check(name, model, device="cuda", drop=0.3, seed=0):
    """Returns (|masked(all kept) - plain|, |masked(x) - masked(x with other pixels in the excluded patches)|,
    |plain(x) - plain(x changed)|) as max abs differences; the first two must be ~0 and the third must not."""
    g = torch.Generator().manual_seed(seed)
    p = patch(name)
    n = 224 // p
    x = torch.randn(4, 3, 224, 224, generator=g).to(device)
    keep = (torch.rand(4, n, n, generator=g) > drop).to(device)
    up = keep.repeat_interleave(p, 1).repeat_interleave(p, 2)[:, None]
    x2 = torch.where(up, x, torch.randn(4, 3, 224, 224, generator=g).to(device))
    d = lambda a, b: (a - b).abs().max().item()
    plain = model(x)
    return (d(forward(name, model, x, torch.ones_like(keep)), plain), d(forward(name, model, x, keep), forward(name, model, x2, keep)),
            d(plain, model(x2)))
