Thanks for the reply! I re-checked against the code, and two points are still unclear to me.

### 1. SatMAE: the checkpoint in the code is the multispectral model

https://github.com/lx709/REOBench/blob/0417584c68b2abad198f105502c5f90e7d247e46/Classification/SatMAE_finetune.py#L31-L44

The SatMAE README lists `pretrain-vit-large-e199.pth` under [Multi-Spectral SatMAE (fMoW-Sentinel)](https://github.com/sustainlab-group/SatMAE/blob/0b210aceb37a14bbbd897110db5b104b3271d818/README.md?plain=1#L193). The checkpoint's own `args.dataset_type` is `'sentinel'`. Its patch embedding is grouped (`patch_embed.0.proj.weight: (1024, 4, 8, 8)`), so `patch_embed.proj.*` is missing in the RGB ViT, and with `strict=False` it stays random.

| SatMAE | Clean | Corruption avg |
|---|---|---|
| `pretrain-vit-large-e199.pth` (as in the code) | 18.2 | 15.7 |
| `fmow_pretrain.pth` (RGB) | 90.7 ± 0.2 | 65.0 ± 0.0 |
| paper | 72.05 | 59.86 |

Neither gives 72.05. Could you check what L44 printed as "Missing keys after loading" in your run?

### 2. Scale-MAE: same checkpoint and preprocessing, still 91.4 vs 75.75

Here the checkpoint loads cleanly: the assert at [L52](https://github.com/lx709/REOBench/blob/0417584c68b2abad198f105502c5f90e7d247e46/Classification/scaleMAE_finetune.py#L52) passes. With `ToTensor()` only and batch size 16, I get 91.4 ± 0.3 (3 seeds). A 15-point gap seems too large for GPU or batch differences. The only input I can't reproduce is the data (`data_type/*.pth`, [L21-22](https://github.com/lx709/REOBench/blob/0417584c68b2abad198f105502c5f90e7d247e46/Classification/scaleMAE_finetune.py#L21-L22)). Could you share **any one** of these?
- `data_type/AID_train_data.pth` / `AID_test_data.pth`, or the script that created them
- the training log, or the saved head weights ([L102](https://github.com/lx709/REOBench/blob/0417584c68b2abad198f105502c5f90e7d247e46/Classification/scaleMAE_finetune.py#L102))

### 3. Normalization (minor)

As far as I can tell, both original repos normalize the input in pretraining and in evaluation:
- SatMAE `transforms.Normalize(mean, std)` with fMoW mean/std: [util/datasets.py L68, L83, L94-95](https://github.com/sustainlab-group/SatMAE/blob/0b210aceb37a14bbbd897110db5b104b3271d818/util/datasets.py#L50-L95)
- Scale-MAE `Normalize(config["data"]["mean"], ...)` with ImageNet mean/std: [main_pretrain.py L443](https://github.com/bair-climate-initiative/scale-mae/blob/89280d830037ff27c20459cdab03e01e633e29bb/mae/main_pretrain.py#L443-L453), [main_linprobe.py L345](https://github.com/bair-climate-initiative/scale-mae/blob/89280d830037ff27c20459cdab03e01e633e29bb/mae/main_linprobe.py#L345-L355)

This doesn't explain the clean gap, but it shifts the corruption averages by about 7 points.

My scripts, results and logs are here: https://github.com/SunwooKim11/REOBench/tree/repro_aid/repro_aid. `issue_check.py` re-runs your loading code for points 1–2 on CPU. Thanks again!
