#!/bin/bash
# Matched backbone pairs for the "fMoW vs ImageNet pretraining" comparison (docs/feedback/2026-11-16_direction.md §3).
# Same architecture and pretraining method inside each pair; only the pretraining data differs.
#
#   pair            fMoW side (already have / get here)                ImageNet side (get here)
#   MIM ViT-L/16    SatMAE, ScaleMAE (have, /data/rs/pretrained_weights) MAE ViT-L/16 (facebookresearch/mae)
#   MoCo-v2 R50     GASSL MoCo-v2+Geo+TP (torchgeo HF mirror, cc-by-4.0)  MoCo-v2 200ep (facebookresearch/moco)
#                   plain MoCo-v2 fMoW = Zenodo only (see below)
#   DINO ViT-S/B    SatDINO ViT-S/16, S/8, B/16 (HF strakajk, Apache-2.0) DINO ViT-S/16, S/8, B/16 (facebookresearch/dino)
#
# All URLs checked with HEAD on 2026-11-16 (sizes below). Total about 4.3 GB.
# Zenodo (GASSL official, record 7379715) answers 403 "unusual traffic" from this server, so the plain MoCo-v2 fMoW
# checkpoint (the exact pair of ImageNet MoCo-v2) has to be fetched elsewhere and copied in:
#   https://zenodo.org/record/7379715/files/moco.pth.tar?download=1  ->  $OUT/gassl/moco_fmow_200ep.pth.tar
# The GASSL repo has no license file.
#
#   ~/.local/tmux-env/bin/tmux new -s bb 'bash ~/REOBench/repro/scripts/tools/download_backbones.sh 2>&1 | tee /data/rs/pretrained_weights/_download_backbones.log'
set -e
OUT=/data/rs/pretrained_weights
get() {  # url, destination (skips files that already exist)
  if [ -s "$2" ]; then echo "exists $2"; return; fi
  mkdir -p "$(dirname "$2")"
  echo "[$(date '+%F %T')] $1"
  wget -q --show-progress -O "$2.part" "$1" && mv "$2.part" "$2"
}

# MIM ViT-L/16, ImageNet-1k (1.21 GB)
get https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth $OUT/imagenet/mae_pretrain_vit_large.pth

# MoCo-v2 ResNet-50, ImageNet-1k 200 ep (112 MB; GASSL compares against this setting)
get https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar \
    $OUT/imagenet/moco_v2_200ep_pretrain.pth.tar
# GASSL MoCo-v2+Geo+TP ResNet-50, fMoW 200 ep (94 MB, encoder only, keys without module.encoder_q.)
get https://huggingface.co/torchgeo/resnet50_fmow_rgb_gassl/resolve/main/resnet50_fmow_rgb_gassl-da43d987.pth \
    $OUT/gassl/resnet50_fmow_rgb_gassl-da43d987.pth

# DINO ViT, ImageNet-1k (87 / 87 / 343 MB)
get https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth $OUT/imagenet/dino_deitsmall16_pretrain.pth
get https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth $OUT/imagenet/dino_deitsmall8_pretrain.pth
get https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth $OUT/imagenet/dino_vitbase16_pretrain.pth
# SatDINO, fMoW-RGB (87 / 87 / 343 MB; checkpoint keys 'teacher' (backbone) and 'linear_head' (63-class fMoW head))
for a in vit_small-16 vit_small-8 vit_base-16; do
  get https://huggingface.co/strakajk/satdino-$a/resolve/main/satdino-$a.pth $OUT/satdino/satdino-$a.pth
done

# Supervised ImageNet-1k reference anchors (no fMoW pair; link to EarthShift R50-imgnet / ViT-imgnet and Beyond Accuracy
# Nat-supervised). Pinned torchvision IMAGENET1K_V1 files, loaded from disk with torchvision.models.<arch>() + load_state_dict
# so the exact weights do not depend on the torchvision version. Normalization = ImageNet mean/std.
#   resnet50 V1 (76.13%, classic recipe: random-resized-crop + flip only; V2 80.86% uses TrivialAugment/mixup/cutmix, not used)
#   vit_b_16 V1 (81.07%), vit_l_16 V1 (79.66%) (torchvision ViT recipe: IN-1k from scratch with heavy augmentation)
get https://download.pytorch.org/models/resnet50-0676ba61.pth $OUT/imagenet/resnet50-0676ba61.pth
get https://download.pytorch.org/models/vit_b_16-c867db91.pth $OUT/imagenet/vit_b_16-c867db91.pth
get https://download.pytorch.org/models/vit_l_16-852ce7e3.pth $OUT/imagenet/vit_l_16-852ce7e3.pth

echo "[$(date '+%F %T')] DONE"
ls -la $OUT/imagenet $OUT/gassl $OUT/satdino
