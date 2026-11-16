"""Frozen backbones for the fMoW-vs-ImageNet pretraining comparison (docs/feedback/2026-11-16_direction.md §3).

Each pair shares architecture and pretraining method; only the pretraining data differs. Supervised ImageNet models are
reference anchors without an fMoW pair. Checkpoints: scripts/tools/download_backbones.sh -> /data/rs/pretrained_weights.

    model, norm, dim = backbones.build("mae_in")      # norm = (mean, std) to put in T.Normalize
    feats = model(x)                                  # (B, dim): the feature each method's own linear evaluation uses

Linear-probe feature per family (official protocol of the model; identical inside a pair):
  MAE, SatMAE, DINO / SatDINO, supervised ViT : CLS after the final norm (SatDINO's released heads are CLS-only, so the
                                                DINO pair uses that instead of DINO's ImageNet setting of 4 blocks / avgpool)
  DINOv3                                      : [CLS, patch mean] (2 x dim)
  ResNet-50                                   : global average pool

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
FMOW_SATMAE = ((0.4182007312774658, 0.4214799106121063, 0.3991275727748871),
               (0.28774282336235046, 0.27541765570640564, 0.2764017581939697))  # = cls_tta.py NORM["satmae"] (PN heads)
FMOW_SATDINO = ((0.433, 0.436, 0.407), (0.198, 0.193, 0.195))  # SatDINO normalization_values["fmow"] (b033d3f)

# name: (family, pretraining data, method, checkpoint, arch)
SPECS = {
    # REOBench MIM pair members at 224 (pretraining size: the checkpoint pos_embed 14x14 is used as is; at 256 REOBench drops
    # it). ScaleMAE gets input_res = 1 as in its official eval. Model code = REOBench/Classification (common.py).
    "satmae": ("reobench_mim", "fMoW-RGB", "SatMAE (MAE)", "/data/rs/pretrained_weights/satmae_fmow_non_temporal_rgb.pth", "satmae"),
    "scalemae": ("reobench_mim", "fMoW-RGB", "Scale-MAE (MAE + GSD PE)", "/data/rs/pretrained_weights/scalemae-vitlarge-800.pth", "scalemae"),
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
    # DINOv3 ViT-L/16, both distilled from a ViT-7B teacher of their own data (HF facebook/*, gated, DINOv3 license)
    "dinov3_web_l16": ("dinov3", "LVD-1689M (web)", "DINOv3 (distilled)", f"{W}/dinov3-vitl16-pretrain-lvd1689m", None),
    "dinov3_sat_l16": ("dinov3", "SAT-493M (Maxar)", "DINOv3 (distilled)", f"{W}/dinov3-vitl16-pretrain-sat493m", None),
    "sup_in_r50": ("tv_resnet", "ImageNet-1k", "supervised (torchvision V1)", f"{W}/imagenet/resnet50-0676ba61.pth", None),
    "sup_in_vitb16": ("tv_vit", "ImageNet-1k", "supervised (torchvision V1)", f"{W}/imagenet/vit_b_16-c867db91.pth", "vit_b_16"),
    "sup_in_vitl16": ("tv_vit", "ImageNet-1k", "supervised (torchvision V1)", f"{W}/imagenet/vit_l_16-852ce7e3.pth", "vit_l_16"),
}
# main pairs (only the pretraining data differs). gassl_fmow (MoCo-v2+Geo+TP) is kept as an optional extra: it adds
# geo-location / temporal-positive objectives, so it is not a clean pair of moco_in (2026-11-16 decision).
PAIRS = [("mae_in", "satmae"), ("moco_in", "moco_fmow"),
         ("dino_in_s16", "satdino_s16"), ("dino_in_s8", "satdino_s8"), ("dino_in_b16", "satdino_b16"),
         ("dinov3_web_l16", "dinov3_sat_l16")]


def _load(model, sd, allowed_missing=(), allowed_unexpected=()):
    msg = model.load_state_dict(sd, strict=False)
    bad_m = [k for k in msg.missing_keys if not k.startswith(tuple(allowed_missing))]
    bad_u = [k for k in msg.unexpected_keys if not k.startswith(tuple(allowed_unexpected))]
    assert not bad_m and not bad_u, f"missing {bad_m[:5]} unexpected {bad_u[:5]}"
    return msg


class _ReobenchMIM(nn.Module):
    def __init__(self, net, name):
        super().__init__()
        self.net, self.name = net, name

    def forward(self, x):
        from . import common
        return common.features(self.net, self.name, x)


class _DINOv3(nn.Module):
    """transformers DINOv3ViTModel -> [CLS, mean of patch tokens] after the final norm (2 x dim), the input of the official
    DINOv3 linear classifier (facebookresearch/dinov3 hub/classifiers.py). Tokens = CLS, register tokens, patches."""

    def __init__(self, net):
        super().__init__()
        self.net, self.skip = net, 1 + net.config.num_register_tokens

    def forward(self, x):
        h = self.net(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, self.skip:].mean(1)], dim=1)


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
    norm = FMOW_SATDINO if fam == "satdino" else FMOW_SATMAE if arch == "satmae" else IMAGENET
    if fam == "dinov3":  # a HF folder, not a single file; normalization from its preprocessor_config.json
        import json
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        pc = json.load(open(f"{ckpt}/preprocessor_config.json"))
        norm = (tuple(pc["image_mean"]), tuple(pc["image_std"]))  # web: ImageNet, sat: (0.430, 0.411, 0.296) / (0.213, 0.156, 0.143)
        net = DINOv3ViTModel.from_pretrained(ckpt) if pretrained else DINOv3ViTModel(DINOv3ViTConfig.from_pretrained(ckpt))
        m = _DINOv3(net)
        ck = None
    else:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False) if pretrained else None
    if fam == "dinov3":
        pass
    elif fam == "reobench_mim":
        from . import common
        mk = common.models_vit if arch == "satmae" else common.scale_models_vit
        kw = dict(patch_size=16, in_chans=3) if arch == "satmae" else {}
        net = mk.vit_large_patch16(img_size=224, num_classes=0, drop_path_rate=0.0, global_pool=False, **kw)
        if pretrained:
            _load(net, ck["model"], allowed_missing=("head",),
                  allowed_unexpected=("mask_token", "decoder", "fpn", "fcn_high", "fcn_low"))  # MAE / Scale-MAE decoders
        m = _ReobenchMIM(net, arch)
    elif fam == "vit_timm":
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
