# [Classification] SatMAE / Scale-MAE AID results not reproducible (≈91% clean vs 72–76% in Table 1)

Hi, thanks for releasing REOBench! I tried to reproduce the AID results for **SatMAE** and **Scale-MAE** (Table 1). Following the released scripts and Appendix F.1, I get **~91% clean accuracy** for both models. That is close to the paper's own **SatMAE++** row, not to the reported 72.05% / 75.75%.

**Setup:** released AID split and corrupted test set. Frozen ViT-L backbone with a linear head, Adam lr 1e-3, batch size 16, 100 epochs, `CosineAnnealingWarmRestarts(T_0=5)` stepped per epoch, `ToTensor()` only, 256×256 input. Checkpoints: SatMAE fMoW-RGB `fmow_pretrain.pth`, Scale-MAE `scalemae-vitlarge-800.pth`. Corruption accuracy is averaged over severities 1–5. 3 seeds.

| Model | Clean | Corruption avg |
|---|---|---|
| SatMAE (paper / mine) | 72.05 / **90.7 ± 0.2** | 59.86 / **65.0 ± 0.0** |
| Scale-MAE (paper / mine) | 75.75 / **91.4 ± 0.3** | 50.27 / **68.0 ± 0.1** |
| SatMAE++ (paper) | 91.35 | 67.50 |

**Things I noticed in the released code**
1. `SatMAE_finetune.py` runs `del model.head` and then calls `model(data)`, which raises `AttributeError` with timm 0.3.2. I used `forward_features()` instead.
2. `pretrain-vit-large-e199.pth` is the fMoW-**Sentinel** (multispectral) checkpoint. In the RGB ViT, `patch_embed` stays random (hidden by `strict=False`), and this gives 18% clean accuracy.
3. `SatMAE_eval.py` loads SatMAE++ weights (`100AID_satmaepp_model_weights.pth`).
4. `data_type/AID_train_data.pth` / `AID_test_data.pth` are not in the repo, so the preprocessing is unknown.
5. `scheduler.step()` is called every iteration, so the cosine cycle is 5 iterations instead of the 5 epochs in Appendix F.1. The effect is small (< 0.6 points).
6. MIM-based models use no input normalization, while CL-based models use ImageNet mean/std (plus rotation augmentation and an MLP head). Adding normalization alone raises the MIM corruption average by about 7 points.

**Questions**
1. Which checkpoints and preprocessing were used for the SatMAE / Scale-MAE rows?
2. Could you share the `data_type/*.pth` files or the script that generated them?
3. Was the protocol difference between MIM- and CL-based models (item 6) intended?

Happy to share my scripts and logs. Thanks!
