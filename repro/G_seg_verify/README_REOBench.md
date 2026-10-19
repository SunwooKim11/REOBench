# Cityscapes→ACDC verification of segmentation TTA (Tent, CoTTA)

**Why this code.** CoTTA's own ACDC code is shared only through OneDrive, which needs a browser. The SVDP repo
(Anonymous-012/SVDP, by the ViDA authors' group) contains a copy of it. That copy includes `single_gpu_cotta` and
`single_gpu_tent`, the CoTTA-modified mmseg (`aug_test` returns probs/preds), and the CoTTA comment
"Batchsize=1 now, was 8 during cityscapes training". Hybrid-TTA ships an identical `single_gpu_tent`.

**Environment.** `conda env segtta`: python 3.10, torch 2.0.1+cu118, mmcv-full 1.7.2. The H200 needs CUDA ≥ 11.8;
the repo pins torch 1.12 / cu116.

**Data and weights.**
- ACDC: `/data/tta/cityscape-ACDC`, `rgb_anon/<cond>/train` (400 images per condition) with `gt/<cond>/train`.
- Weights: `segformer.b5.1024x1024.city.160k.pth`, from the SVDP README Drive link.

**Patches (all marked `[REOBench]`).**
1. `mmseg/__init__.py`: MMCV_MAX raised from 1.7.1 to 1.7.2.
2. `local_configs/_base_/datasets/acdc_1024x1024_repeat.py`: `data_root` changed.
3. `encoder_decoder.simple_test`: restored the standard mmseg return. SVDP returned `(pred, prob)`, which breaks
   `single_gpu_test` and `single_gpu_tent`. The SVDP version is kept as `simple_test_with_prob`.
4. `mmseg/apis/test.py`:
   - The forced image-dump `out_dir` is removed from `single_gpu_test` and `single_gpu_cotta`.
   - `single_gpu_cotta` used `args.ema_rate`, which is undefined there. It now uses 0.999, CoTTA's EMA.
5. `tools/tent.py`: SVDP kept only fog; the four conditions are restored, as in Hybrid-TTA's `tools/tent.py`.
6. `tools/{tent,cotta}.py`: an optional `ROUNDS` environment variable. The default is 10, as in the paper.

**Note: segmentation "TENT" in this code is not entropy minimization.** It is cross-entropy on the model's own
argmax prediction (hard pseudo-labels), with Adam 6e-5/8 on norm parameters. The CTTA segmentation papers (CoTTA,
ViDA, SVDP, Hybrid-TTA) all report this variant as "TENT".

**Reference: CoTTA Table 5, mIoU.**

| | Fog | Night | Rain | Snow |
|---|---|---|---|---|
| Source | 69.1 | 40.3 | 59.7 | 57.8 |
| TENT-continual, round 1 | 69.0 | 40.2 | 60.1 | 57.3 |
| CoTTA, round 1 | 70.9 | 41.2 | 62.4 | 59.7 |

The mean over all 10 rounds is 56.7 for Source, 52.3 for TENT and 58.6 for CoTTA.
