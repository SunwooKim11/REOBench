Thank you for the reply! I re-checked everything against the code, and two things still don't add up. I've quoted the exact lines below so they're easy to check.

### 1. SatMAE: the checkpoint in the code is the multispectral (Sentinel) model

`SatMAE_finetune.py` loads `pretrain-vit-large-e199.pth`:
https://github.com/lx709/REOBench/blob/0417584c68b2abad198f105502c5f90e7d247e46/Classification/SatMAE_finetune.py#L31-L44

In the SatMAE README, this file is listed under **Multi-Spectral SatMAE (fMoW-Sentinel)**. The RGB model is `fmow_pretrain.pth`:

> [SatMAE README L193](https://github.com/sustainlab-group/SatMAE/blob/0b210aceb37a14bbbd897110db5b104b3271d818/README.md?plain=1#L193): `ViT-Large (200 epochs) | 63.84% | [download](https://zenodo.org/record/7338613/files/pretrain-vit-large-e199.pth)`

The checkpoint itself says so too. It has `args.dataset_type = 'sentinel'`, and its patch embedding is grouped (`patch_embed.0.proj.weight: (1024, 4, 8, 8)`, `patch_embed.1...`). Because of this, `patch_embed.proj.*` is missing when it is loaded into the RGB ViT, and with `strict=False` it stays randomly initialized. With this loading code (plus only the fixes listed in my first post), I get:

| SatMAE | Clean | Corruption avg |
|---|---|---|
| `pretrain-vit-large-e199.pth` (as in the code) | 18.2 | 15.7 |
| `fmow_pretrain.pth` (RGB) | 90.7 ± 0.2 | 65.0 ± 0.0 |
| paper | 72.05 | 59.86 |

Neither checkpoint gives 72.05. Could you check what L44 printed for "Missing keys after loading" in your run?

### 2. Scale-MAE: same checkpoint and preprocessing, still 91.4 vs 75.75

Here the checkpoint loads cleanly, since only the head and `pos_embed` are missing ([L33](https://github.com/lx709/REOBench/blob/0417584c68b2abad198f105502c5f90e7d247e46/Classification/scaleMAE_finetune.py#L33), [L52](https://github.com/lx709/REOBench/blob/0417584c68b2abad198f105502c5f90e7d247e46/Classification/scaleMAE_finetune.py#L52)). With `ToTensor()` only, batch size 16 and 3 seeds (std 0.3), I get 91.4 / 68.0 against the paper's 75.75 / 50.27. A 15-point gap seems too large to come from the GPU or the batch setting.

The one input I couldn't reproduce is the data itself:
https://github.com/lx709/REOBench/blob/0417584c68b2abad198f105502c5f90e7d247e46/Classification/scaleMAE_finetune.py#L21-L22

Could you share **any one** of these? Any of them would let me find where we differ.
- `data_type/AID_train_data.pth` / `AID_test_data.pth`, or the script that created them
- the training log, or the saved `{epoch}_AID_scalemae_model_weights.pth` ([L102](https://github.com/lx709/REOBench/blob/0417584c68b2abad198f105502c5f90e7d247e46/Classification/scaleMAE_finetune.py#L102))

### 3. Normalization (minor, separate from the gap above)

Both original repos normalize the input, in pretraining as well as in fine-tuning and linear probing:

- **SatMAE**: [`util/datasets.py`](https://github.com/sustainlab-group/SatMAE/blob/0b210aceb37a14bbbd897110db5b104b3271d818/util/datasets.py#L50-L90) applies `transforms.Normalize(mean, std)` in both the train (L68) and eval (L83) transforms, using fMoW mean/std for RGB (L94-95). Pretraining and fine-tuning both use it ([`main_pretrain.py` L136](https://github.com/sustainlab-group/SatMAE/blob/0b210aceb37a14bbbd897110db5b104b3271d818/main_pretrain.py#L136), [`main_finetune.py` L193-194](https://github.com/sustainlab-group/SatMAE/blob/0b210aceb37a14bbbd897110db5b104b3271d818/main_finetune.py#L193-L194)).
  ```python
  t.append(transforms.ToTensor())
  t.append(transforms.Normalize(mean, std))
  ```
- **Scale-MAE**: [`config/fmow.yaml`](https://github.com/bair-climate-initiative/scale-mae/blob/89280d830037ff27c20459cdab03e01e633e29bb/mae/config/fmow.yaml#L6-L7) sets ImageNet mean/std. That normalization is applied in pretraining ([`main_pretrain.py` L443-453](https://github.com/bair-climate-initiative/scale-mae/blob/89280d830037ff27c20459cdab03e01e633e29bb/mae/main_pretrain.py#L443-L453)) and in linear probing ([`main_linprobe.py` L345-355](https://github.com/bair-climate-initiative/scale-mae/blob/89280d830037ff27c20459cdab03e01e633e29bb/mae/main_linprobe.py#L345-L355)).
  ```python
  tv_transforms.Normalize(mean=config["data"]["mean"], std=config["data"]["std"])
  ```

This doesn't explain the clean gap, since I get ~91% even without normalization. But it changes the corruption numbers by about 7 points, so it might be worth a note in the paper.

I'm happy to share my scripts and logs, which follow the released code with only the fixes noted in my first post: https://github.com/SunwooKim11/REOBench/tree/repro_aid/repro_aid (`issue_check.py` re-runs the released loading code for points 1–2 on CPU). Thanks again for your time!
