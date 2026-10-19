"""Study aid: run ONE real Potsdam patch through the REOBench segmentation model and print every tensor shape.

Follows the code path a training step and a test step take (see STUDY_GUIDE.md for the reading order):
  image -> ViT-L backbone (patch tokens of blocks 3,5,7,11) -> fpn1..4 (4 scales)
        -> UPerHead (PPM on the coarsest scale + FPN top-down + fuse) -> 1x1 conv -> logits (1/4 res)
        -> upsample to the label size -> cross-entropy (ignore 5)        [training]
        -> upsample to the image size -> argmax per pixel                [test]
CPU by default so it never competes with a running job (about a minute; use --cuda if the GPU is free).

  cd ~/REOBench/repro/H_seg_baseline/seg
  PYTHONPATH=. TIMM_FUSED_ATTN=0 /opt/conda/envs/segtta/bin/python ../trace_shapes.py [--ckpt ../work_dirs/scalemae_A_12ep/latest.pth]
"""
import argparse

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from mmseg.models import build_segmentor

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/repro/scalemae_A_12ep.py")
p.add_argument("--ckpt", default=None, help="trained segmentor; default = backbone-only init (random heads)")
p.add_argument("--image", default="/data/rs/reobench/Potsdam_extracted/test/clean/2_13_0_0_512_512.png")
p.add_argument("--cuda", action="store_true")
args = p.parse_args()
dev = torch.device("cuda" if args.cuda else "cpu")
show = lambda name, t: print(f"{name:48s} {tuple(t.shape)}")

# ---- 1. model -------------------------------------------------------------------------------------------
cfg = mmcv.Config.fromfile(args.config)
model = build_segmentor(cfg.model)
model.init_weights()
if args.ckpt:
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["state_dict"])
model = model.to(dev).eval()
bb, head, aux = model.backbone, model.decode_head, model.auxiliary_head
print(f"\nbackbone {type(bb).__name__}: {len(bb.blocks)} blocks, dim {bb.embed_dim}; decode {type(head).__name__}; aux {type(aux).__name__}")
print(f"params: backbone {sum(p.numel() for p in bb.parameters())/1e6:.0f}M, decode {sum(p.numel() for p in head.parameters())/1e6:.0f}M, "
      f"aux {sum(p.numel() for p in aux.parameters())/1e6:.1f}M\n")

# ---- 2. input (same normalization as the config's pipeline) ---------------------------------------------
n = cfg.img_norm_cfg
img = np.asarray(Image.open(args.image).convert("RGB")).astype(np.float32)          # H, W, 3 in 0..255
x = torch.from_numpy((img - np.array(n["mean"])) / np.array(n["std"])).permute(2, 0, 1)[None].float().to(dev)
label = np.asarray(Image.open(args.image.replace("/test/clean/", "/test/clean_ann/")).convert("L"))
show("input image (normalized)", x)

with torch.no_grad():
    # ---- 3. backbone: mmseg/models/backbones/scale_MAE_vit_large.py forward_features --------------------
    tok, (Hp, Wp) = bb.patch_embed(x)
    show("patch_embed: 16x16 patches -> tokens", tok)
    tok = tok + bb.pos_embed[:, 1:, :]                                                # CLS position dropped
    feats = []
    for i, blk in enumerate(bb.blocks):
        tok = blk(tok)
        if i in [3, 5, 7, 11]:
            feats.append(tok)
            show(f"  block {i:2d} output (kept)", tok)
    print(f"  ... blocks 12-{len(bb.blocks) - 1} are computed but not used")
    maps = [f.permute(0, 2, 1).reshape(1, -1, Hp, Wp) for f in feats]
    show("tokens -> 2D map (B, C, H/16, W/16)", maps[0])
    pyr = [op(m) for op, m in zip([bb.fpn1, bb.fpn2, bb.fpn3, bb.fpn4], maps)]
    for i, f in enumerate(pyr):
        show(f"fpn{i + 1} -> scale 1/{512 // f.shape[-1]}", f)

    # ---- 4. UPerHead: mmseg/models/decode_heads/uper_head.py forward -------------------------------------
    inputs = head._transform_inputs(list(pyr))
    lat = [conv(inputs[i]) for i, conv in enumerate(head.lateral_convs)]
    lat.append(head.psp_forward(inputs))                                              # PPM on the 1/32 map
    for i, l in enumerate(lat):
        show(f"lateral {i} ({'PPM' if i == 3 else '1x1 conv'})", l)
    for i in range(len(lat) - 1, 0, -1):                                              # top-down FPN
        lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[2:], mode="bilinear", align_corners=False)
    outs = [head.fpn_convs[i](lat[i]) for i in range(len(lat) - 1)] + [lat[-1]]
    outs = [outs[0]] + [F.interpolate(o, size=outs[0].shape[2:], mode="bilinear", align_corners=False) for o in outs[1:]]
    fused = head.fpn_bottleneck(torch.cat(outs, 1))
    show("concat 4 levels at 1/4 -> fpn_bottleneck", fused)
    logits = head.cls_seg(fused)
    show("cls_seg (dropout + 1x1 conv) = logits", logits)
    show("aux FCNHead logits (from fpn3, train only)", aux(list(pyr)))

    # ---- 5. what training and test do with the logits ---------------------------------------------------
    up = F.interpolate(logits, size=label.shape, mode="bilinear", align_corners=False)
    show("upsampled to label size", up)
    y = torch.from_numpy(label.astype(np.int64)).to(dev).clone(); y[y == 255] = 5
    ce = F.cross_entropy(up, y[None], ignore_index=5)
    print(f"{'train: CE(logits, label), ignore 5':48s} {ce.item():.3f}   (decode weight 1.0 + aux weight 0.4)")
    pred = up.argmax(1)[0].cpu().numpy()
    valid = (label != 5) & (label != 255)
    print(f"{'test: argmax per pixel -> pixel accuracy':48s} {100 * (pred[valid] == label[valid]).mean():.1f}%")
    ref = model.encode_decode(x, [dict(img_shape=(512, 512, 3), ori_shape=(512, 512, 3), pad_shape=(512, 512, 3), flip=False)])
    print(f"{'check vs model.encode_decode (max abs diff)':48s} {(ref - up).abs().max().item():.2e}")
