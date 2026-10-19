"""Self-check for the REOBench issue #4 reply: reproduces each factual claim from the released code itself.

Runs the checkpoint-loading blocks of Classification/SatMAE_finetune.py (L26-44) and scaleMAE_finetune.py (L26-52)
verbatim (only the checkpoint path changed), on CPU, and prints what the reply quotes.

  ~/REOBench/.venv/bin/python ~/REOBench/repro/issue_check.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/REOBench/Classification"))  # released models_vit / util

W = "/data/rs/pretrained_weights"

print("=== claim 1: pretrain-vit-large-e199.pth is the multispectral (Sentinel) SatMAE ===")
ck = torch.load(f"{W}/pretrain-vit-large-e199.pth", map_location="cpu", weights_only=False)
print("args.dataset_type :", getattr(ck.get("args"), "dataset_type", None), "| epoch:", ck.get("epoch"))
print("patch_embed keys  :", [(k, tuple(v.shape)) for k, v in ck["model"].items() if k.startswith("patch_embed")][:3], "...")

print("\n=== claim 2: with the released SatMAE loading code, patch_embed stays random for e199 ===")
import models_vit  # noqa: E402  (REOBench/Classification)
from util.pos_embed import interpolate_pos_embed  # noqa: E402
for name in ["pretrain-vit-large-e199.pth", "satmae_fmow_non_temporal_rgb.pth"]:  # code's ckpt vs fmow_pretrain.pth
    # ---- SatMAE_finetune.py L26-44 ----
    model = models_vit.__dict__["vit_large_patch16"](
        patch_size=16, img_size=256, in_chans=3,
        num_classes=30, drop_path_rate=0.2, global_pool=False
    )
    checkpoint = torch.load(f"{W}/{name}", map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"{name:36s} Missing keys after loading: {sorted(msg.missing_keys)}")
print("(pos_embed missing is harmless: models_vit.py L24-26 fills it with fixed 2D sin-cos, as in MAE pretraining)")

print("\n=== claim 3: Scale-MAE checkpoint loads cleanly (only head + pos_embed missing) ===")
import scale_models_vit  # noqa: E402
from util.pos_embed_scale import interpolate_pos_embed as interp_scale  # noqa: E402
# ---- scaleMAE_finetune.py L26-52 ----
model = scale_models_vit.__dict__["vit_large_patch16"](img_size=256, num_classes=30, drop_path_rate=0.1,
                                                       global_pool=False)
checkpoint_model = torch.load(f"{W}/scalemae-vitlarge-800.pth", map_location="cpu", weights_only=False)['model']
state_dict = model.state_dict()
for k in ["head.weight", "head.bias"]:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        del checkpoint_model[k]
if "pos_embed" in checkpoint_model and checkpoint_model["pos_embed"].shape != state_dict["pos_embed"].shape:
    del checkpoint_model["pos_embed"]
interp_scale(model, checkpoint_model)
msg = model.load_state_dict(checkpoint_model, strict=False)
print("Missing keys:", sorted(msg.missing_keys))
assert set(msg.missing_keys) == {"head.weight", "head.bias", "pos_embed"}
print("assert at L52 passes")
