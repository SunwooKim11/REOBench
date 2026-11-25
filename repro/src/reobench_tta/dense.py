"""Patch-level feature maps of the frozen backbones (backbones.build), for dense prediction with a linear head.

    model = dense.prepare(name, model)                 # once (changes ResNets in place)
    fmap = dense.feature_map(name, model, x)           # x: (B, 3, 224, 224) normalized -> (B, D, g, g)

ViTs: patch tokens after the final norm (g = 224 / patch: 14, or 28 for the /8 models); CLS and register tokens dropped.
  DINOv3 here is the patch tokens only (D = 1024), not the [CLS, patch mean] vector of the classification feature.
ResNet-50: layer4 output with layer4 dilated instead of strided (output stride 16, g = 14, D = 2048; the usual
  segmentation conversion, pretrained weights unchanged), so every backbone but the /8 ViTs has the same grid.
"""
from . import backbones, token_mask

RESNET = ("resnet_moco", "resnet_plain", "tv_resnet")


def _resnet(model):
    return model.net if isinstance(model, backbones._Supervised) else model


def prepare(name, model):
    if backbones.SPECS[name][0] in RESNET:
        l4 = _resnet(model).layer4
        l4[0].conv2.stride = (1, 1)
        l4[0].downsample[0].stride = (1, 1)
        for blk in list(l4)[1:]:
            blk.conv2.dilation, blk.conv2.padding = (2, 2), (2, 2)
    return model


def _grab(module, run):
    out = []
    h = module.register_forward_hook(lambda m, i, o: out.append(o))
    try:
        run()
    finally:
        h.remove()
    return out[0]


def feature_map(name, model, x):
    fam = backbones.SPECS[name][0]
    B, _, H, W = x.shape
    if fam in RESNET:
        return _grab(_resnet(model).layer4, lambda: model(x))
    p = token_mask.patch(name)
    if fam == "vit_timm":
        t = model.forward_features(x)[:, 1:]
    elif fam == "satdino":
        t = model(x, return_all=True)[:, 1:-1]
    elif fam == "dinov3":
        t = model.net(pixel_values=x).last_hidden_state[:, model.skip:]
    elif fam == "reobench_mim":
        t = _grab(model.net.norm, lambda: model(x))[:, 1:]
    else:  # tv_vit
        t = _grab(model.net.encoder.ln, lambda: model(x))[:, 1:]
    assert t.shape[1] == (H // p) * (W // p), (name, t.shape)
    return t.transpose(1, 2).reshape(B, -1, H // p, W // p)
