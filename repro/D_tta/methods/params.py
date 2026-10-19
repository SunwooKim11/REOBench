"""[REOBench] Shared parameter selection for TENT / SAR / DeYO.

The DeYO/SAR repos hard-code ViT-B (12 blocks): they skip 'blocks.9/10/11' and the final 'norm'.
Applied to ViT-L (24 blocks) by string match, that would skip middle blocks 9-11 and adapt the top
blocks 12-23 instead. Here the same rule is generalized: skip the top 25% of blocks
(ViT-B: 9-11, ViT-L: 18-23) and the backbone's final norm; adapt LayerNorm affine params elsewhere.
ResNets (ImageNet-C verification only): skip 'layer4', as in the original code.
"""
import re

import torch.nn as nn


def collect_norm_params(model):
    blocks = getattr(model.backbone, "blocks", None)
    first_skipped = len(blocks) - len(blocks) // 4 if blocks is not None else None
    params, names = [], []
    for nm, m in model.named_modules():
        b = re.search(r"blocks\.(\d+)(\.|$)", nm)
        if b and first_skipped is not None and int(b.group(1)) >= first_skipped:
            continue
        if "layer4" in nm:
            continue
        if nm == "backbone.norm":
            continue
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np_, p in m.named_parameters():
                if np_ in ["weight", "bias"]:
                    params.append(p)
                    names.append(f"{nm}.{np_}")
    return params, names
