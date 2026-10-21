# [REOBench repro] Setting B (fixed) for SatMAE: A + three fixes.
#   - RGB checkpoint fmow_pretrain.pth (A: multispectral Sentinel pretrain-vit-large-e199.pth -> random patch_embed)
#   - normalization on the 0-255 scale: fMoW mean/std x 255 (A: 0-1 values applied to 0-255 pixels -> inputs up to ~900)
#   - feature pyramid from ViT-L blocks [7, 11, 15, 23] and 24-layer layer decay (A: ViT-B [3, 5, 7, 11], 12 layers)
_base_ = ['./satmae_A_12ep.py']
_mean = [0.4182007312774658 * 255, 0.4214799106121063 * 255, 0.3991275727748871 * 255]
_std = [0.28774282336235046 * 255, 0.27541765570640564 * 255, 0.2764017581939697 * 255]
img_norm_cfg = dict(mean=_mean, std=_std, to_rgb=True)
model = dict(backbone=dict(out_indices=(7, 11, 15, 23),
                           ckpt='/data/rs/pretrained_weights/satmae_fmow_non_temporal_rgb.pth'))
optimizer = dict(paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.9))
# pipelines must be redefined: the inherited ones already hold the A normalization values
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=None, img_ratios=[1.0], flip=False,
         transforms=[dict(type='Resize', keep_ratio=True), dict(type='RandomFlip'),
                     dict(type='Normalize', **img_norm_cfg), dict(type='ImageToTensor', keys=['img']),
                     dict(type='Collect', keys=['img'])]),
]
data = dict(train=dict(pipeline=train_pipeline), val=dict(pipeline=test_pipeline), test=dict(pipeline=test_pipeline))
