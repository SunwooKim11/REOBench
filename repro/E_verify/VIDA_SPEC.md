# ViDA (ICLR 2024) port spec — from the official repo (Yangsenqiao/vida @ c26b0db, `imagenet/`)

Source checkpoint (ViT-B, from the repo README Drive link): `/data/rs/pretrained_weights/vida_imagenet_vitb_source.pt`
- full model, keys prefixed `module.`; backbone bit-identical to timm `vit_base_patch16_224.augreg_in21k_ft_in1k`
  (timm >= 0.9 defaults to `augreg2`, which is different — request the tag explicitly)
- adapters only were warm-started: `vida_down*` std ~1.0 (not the code's 1/r^2 init), `vida_up*` ~2e-5
- no training script in the repo; paper App. C.4: random init is close on ACDC (60.5 vs 61.4 mIoU)

## Adapters (`inject_vida.py`)
- every `Attention` module: `qkv` and `proj` Linear -> `ViDAInjectedLinear` (MLP, norms, head untouched)
- `y = Wx + b + s1 * up1(down1(x)) + s2 * up2(down2(x))`, low rank r=1, high rank r2=128, no bias/activation
- init: down ~ N(0, 1/r^2), up = 0; ranks absolute (keep 1/128 for ViT-L)
- names with `vida_` = adapter params; backbone weights live in `*.linear_vida.*` (does not match `vida_`)

## HKA (`vida.py:103-131`)
- U = 0.1 * mean(var over 10 teacher logits on CoTTA `get_tta_transforms()` augs), one scalar per batch
- U >= 0.2: (high, low) = (1+U, 1-U); else low = 1+U, high = 1-U
- **bug**: `set_scale` uses `if hasattr(scale1) ... elif hasattr(scale2)` -> only scale1 (low) is ever set, scale2 stays 1.0.
  The reported 43.4 comes from this code -> reproduce the bug for verification (flag `--vida_fix_hka` for the paper version)
- augmented predictions only feed U, not the pseudo-label

## Loop (`vida.py:132-172`, `imagenetc.py`)
- teacher(x) -> p_t (returned prediction, before the update); student(x) -> p_s
- loss = symmetric CE: -0.5 sum(p_t log p_s) - 0.5 sum(p_s log p_t)
- ALL params trained: SGD nesterov momentum 0.9, wd 0; non-`vida_` lr 5e-7, `vida_` lr 2e-7; bs 50, 1 step
- EMA: `vida_` alpha 0.8, others 0.999; stochastic restore off on ImageNet (on for CIFAR, rst 0.1)
- continual: reset only before the first domain; order gauss, shot, impulse, defocus, glass, motion, zoom, snow, frost,
  fog, brightness, contrast, elastic, pixelate, jpeg; severity 5; 5000 imgs/domain from robustbench
  `imagenet_test_image_ids.txt`, file order, no shuffle
- preprocessing: Resize 256, CenterCrop 224, ToTensor, **no mean/std normalization** (augreg expects 0.5/0.5)
- target (error %, ViT-B): paper 43.4 (repo log 43.37); Source 55.8, Tent 51.0, CoTTA 54.8

## Paper vs code
| item | paper | code |
|---|---|---|
| HKA | both scales | only low-rank (bug) |
| uncertainty | MC dropout, per-sample | 10 augs, variance of logits x0.1, per batch |
| loss | CE | symmetric CE |
| params | adapters | all (backbone 5e-7, adapters 2e-7) |
| optimizer | Adam | SGD nesterov |
| EMA | 0.999 | 0.999 / 0.8 (adapters) |

## ViT-L MAE port notes
- normalization inside the model so the augs' clip(0,1) stays valid
- 0.1 factor and threshold 0.2 were tuned to ViT-B logits: log U per batch on new backbones
- no ViT-L source checkpoint: warm-up adapters on source train (frozen backbone, CE) or random init
- adapter params ViT-L: ~19.0M (ViT-B 7.13M)
