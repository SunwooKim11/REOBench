"""Frozen backbones for the fMoW-vs-ImageNet pretraining comparison (docs/feedback/2026-11-16_direction.md §3).

Each pair shares architecture and pretraining method; only the pretraining data differs. Supervised ImageNet models are
reference anchors without an fMoW pair. Checkpoints: scripts/tools/download_backbones.sh -> /data/rs/pretrained_weights.

    model, norm, dim = backbones.build("mae_in")      # norm = (mean, std) to put in T.Normalize
    feats = model(x)                                  # (B, dim): CLS after the final norm (ViT) / global avg pool (ResNet)

The ViT-L MIM pair at 256 px (SatMAE, ScaleMAE as in REOBench) stays in common.build_backbone; here everything is the
model's own pretraining resolution (224, timm / torchvision / SatDINO code). Loading fails loudly on unexpected key gaps.
"""
import sys

import timm
import torch
import torch.nn as nn
import torchvision

from .paths import THIRD

W = "/data/rs/pretrained_weights"
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
FMOW_SATDINO = ((0.433, 0.436, 0.407), (0.198, 0.193, 0.195))  # SatDINO normalization_values["fmow"] (b033d3f)

# name: (family, pretraining data, method, checkpoint, arch)
SPECS = {
    "mae_in": ("vit_timm", "ImageNet-1k", "MAE", f"{W}/imagenet/mae_pretrain_vit_large.pth", "vit_large_patch16_224"),
    "moco_in": ("resnet_moco", "ImageNet-1k", "MoCo-v2 200ep", f"{W}/imagenet/moco_v2_200ep_pretrain.pth.tar", None),
    "moco_fmow": ("resnet_moco", "fMoW-RGB", "MoCo-v2 200ep (GASSL baseline)", f"{W}/gassl/moco_fmow_200ep.pth.tar", None),
    "gassl_fmow": ("resnet_plain", "fMoW-RGB", "MoCo-v2+Geo+TP (GASSL)", f"{W}/gassl/resnet50_fmow_rgb_gassl-da43d987.pth", None),
    "dino_in_s16": ("vit_timm", "ImageNet-1k", "DINO", f"{W}/imagenet/dino_deitsmall16_pretrain.pth", "vit_small_patch16_224"),
    "dino_in_s8": ("vit_timm", "ImageNet-1k", "DINO", f"{W}/imagenet/dino_deitsmall8_pretrain.pth", "vit_small_patch8_224"),
    "dino_in_b16": ("vit_timm", "ImageNet-1k", "DINO", f"{W}/imagenet/dino_vitbase16_pretrain.pth", "vit_base_patch16_224"),
    "satdino_s16": ("satdino", "fMoW-RGB", "SatDINO", f"{W}/satdino/satdino-vit_small-16.pth", ("vit_small", 16)),
    "satdino_s8": ("satdino", "fMoW-RGB", "SatDINO", f"{W}/satdino/satdino-vit_small-8.pth", ("vit_small", 8)),
    "satdino_b16": ("satdino", "fMoW-RGB", "SatDINO", f"{W}/satdino/satdino-vit_base-16.pth", ("vit_base", 16)),
    "sup_in_r50": ("tv_resnet", "ImageNet-1k", "supervised (torchvision V1)", f"{W}/imagenet/resnet50-0676ba61.pth", None),
    "sup_in_vitb16": ("tv_vit", "ImageNet-1k", "supervised (torchvision V1)", f"{W}/imagenet/vit_b_16-c867db91.pth", "vit_b_16"),
    "sup_in_vitl16": ("tv_vit", "ImageNet-1k", "supervised (torchvision V1)", f"{W}/imagenet/vit_l_16-852ce7e3.pth", "vit_l_16"),
}
PAIRS = [("mae_in", "satmae"), ("moco_in", "moco_fmow"), ("moco_in", "gassl_fmow"),
         ("dino_in_s16", "satdino_s16"), ("dino_in_s8", "satdino_s8"), ("dino_in_b16", "satdino_b16")]


def _load(model, sd, allowed_missing=(), allowed_unexpected=()):
    msg = model.load_state_dict(sd, strict=False)
    bad_m = [k for k in msg.missing_keys if not k.startswith(tuple(allowed_missing))]
    bad_u = [k for k in msg.unexpected_keys if not k.startswith(tuple(allowed_unexpected))]
    assert not bad_m and not bad_u, f"missing {bad_m[:5]} unexpected {bad_u[:5]}"
    return msg


class _Supervised(nn.Module):
    """Feature extractor that keeps the ImageNet classifier available as .classify (for checking the weights)."""

    def __init__(self, net, head_attr):
        super().__init__()
        self.net, self.head = net, getattr(net, head_attr)
        setattr(net, head_attr, nn.Identity())

    def forward(self, x):
        return self.net(x)

    def classify(self, x):
        return self.head(self.net(x))


def build(name, pretrained=True):
    """Return (model in eval mode, (mean, std), feature dim). pretrained=False gives the same arch with random weights."""
    fam, _, _, ckpt, arch = SPECS[name]
    norm = FMOW_SATDINO if fam == "satdino" else IMAGENET
    ck = torch.load(ckpt, map_location="cpu", weights_only=False) if pretrained else None
    if fam == "vit_timm":
        m = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="token")
        if pretrained:
            _load(m, ck.get("model", ck), allowed_unexpected=("mask_token", "decoder", "head"))
    elif fam == "satdino":
        sys.path.insert(0, str(THIRD / "satdino"))
        from satdino import vision_transformer_satdino as vs  # Apache-2.0, strakaj/SatDINO @ b033d3f
        m = getattr(vs, arch[0])(patch_size=arch[1])
        if pretrained:
            _load(m, ck["teacher"])  # strict; forward returns CLS after norm (GSD register is an extra token, no GSD input)
    elif fam in ("resnet_moco", "resnet_plain"):
        m = torchvision.models.resnet50()
        m.fc = nn.Identity()
        if pretrained:
            sd = ck["state_dict"] if fam == "resnet_moco" else ck
            if fam == "resnet_moco":
                sd = {k[len("module.encoder_q."):]: v for k, v in sd.items()
                      if k.startswith("module.encoder_q.") and not k.startswith("module.encoder_q.fc")}
            _load(m, sd)
    elif fam == "tv_resnet":
        net = torchvision.models.resnet50()
        if pretrained:
            _load(net, ck)
        m = _Supervised(net, "fc")
    elif fam == "tv_vit":
        net = getattr(torchvision.models, arch)()
        if pretrained:
            _load(net, ck)
        m = _Supervised(net, "heads")
    m.eval()
    with torch.no_grad():
        dim = m(torch.zeros(1, 3, 224, 224)).shape[1]
    return m, norm, dim
