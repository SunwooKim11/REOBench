# [REOBench repro] Setting B (fixed) for Scale-MAE: A + ViT-L feature blocks and 24-layer layer decay.
#   - feature pyramid from blocks [7, 11, 15, 23] (ViT-L; ViTDet/MAE practice) instead of [3, 5, 7, 11] (ViT-B/RVSA)
#   - LayerDecayOptimizerConstructor num_layers 24 instead of 12
#   (normalization: already ImageNet mean/std on 0-255 pixels; checkpoint: scalemae-vitlarge-800.pth, as in A)
_base_ = ['./scalemae_A_12ep.py']
model = dict(backbone=dict(out_indices=(7, 11, 15, 23)))
optimizer = dict(paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.9))
