# AID classification reproduction for SatMAE / Scale-MAE (issue #4)

This folder backs the numbers in [issue #4](https://github.com/lx709/REOBench/issues/4). It follows the released
`Classification/SatMAE_finetune.py` and `scaleMAE_finetune.py`, and changes only what cannot run as released. Every
change is listed below.

## What is what

Both scripts run the **same protocol** as the released code. They differ only in how they are run:

| Folder | What it is | Used for |
|---|---|---|
| `A_paper/` | A direct port of the released scripts: one model per run, end to end, fp32 | the SatMAE run with the checkpoint named in the code (18.2) |
| `C_ablation/` | The same protocol with cached eval features, so that many seeds and settings are cheap to run | the 3-seed numbers in the issue |

The two runs in `C_ablation/` differ in **one setting only**, which is when the LR scheduler steps:

| Config name | `CosineAnnealingWarmRestarts(T_0=5)` steps… | Matches |
|---|---|---|
| `A0ep` (**main**) | once per **epoch**, so each cycle is 5 epochs | the paper, Appendix F.1 |
| `A0` | once per **iteration**, so each cycle is 5 iterations | the released code (`scheduler.step()` inside the batch loop) |

The two give nearly the same results (< 0.6 pt), so the scheduler does not explain the gap.

## Numbers in the issue → how to reproduce them

| Number in the issue | Script / config | Result file |
|---|---|---|
| SatMAE 90.7 ± 0.2 / 65.0 ± 0.0 | `C_ablation/run.py --model satmae --name A0ep --optim adam_wr_epoch --seed {0,1,2}` | `C_ablation/outputs/satmae/A0ep_s*.json` |
| Scale-MAE 91.4 ± 0.3 / 68.0 ± 0.1 | `C_ablation/run.py --model scalemae --name A0ep --optim adam_wr_epoch --bb_train --seed {0,1,2}` | `C_ablation/outputs/scalemae/A0ep_s*.json` |
| Scheduler stepped per iteration, as in the released code (< 0.6 pt difference) | same as above, but with `--name A0` and without `--optim` (default `adam_wr`) | `C_ablation/outputs/*/A0_s*.json` |
| SatMAE with the checkpoint named in the code (`pretrain-vit-large-e199.pth`): 18.2 clean | `A_paper/run.py --model satmae --ckpt satmae_sentinel --tag _sentinel` | `A_paper/outputs/satmae_sentinel_results.txt`, `.log` |
| Checkpoint-loading claims (Sentinel checkpoint, missing `patch_embed`, Scale-MAE loads cleanly) | `issue_check.py`, which runs the released loading code verbatim on CPU | printed |

Run `python C_ablation/summarize.py` to print the mean ± std table from the JSON files:

```
### satmae
config   n     clean       avg
paper    -     72.05     59.86
A0       3  90.5±0.2  65.0±0.2   <- scheduler per iteration (released code)
A0ep     3  90.7±0.2  65.0±0.0   <- scheduler per epoch (paper), main
### scalemae
paper    -     75.75     50.27
A0       3  91.1±0.2  67.4±0.2   <- scheduler per iteration (released code)
A0ep     3  91.4±0.3  68.0±0.1   <- scheduler per epoch (paper), main
```

## Setup

- Place this folder at the repo root, next to `Classification/`. It imports the released `models_vit.py`,
  `scale_models_vit.py` and `util/`.
- Data: the released AID split and corrupted test set. Set `REOBENCH_AID` to the folder that contains
  `AID_clean_renamed/{train,test}` and `corrupted_renamed/`.
- Weights: set `REOBENCH_WEIGHTS` to the folder that contains these files.
  - `satmae_fmow_non_temporal_rgb.pth`: SatMAE `fmow_pretrain.pth`, [zenodo 7369797](https://zenodo.org/record/7369797)
  - `scalemae-vitlarge-800.pth`: the Scale-MAE release
  - `pretrain-vit-large-e199.pth`: SatMAE fMoW-Sentinel, [zenodo 7338613](https://zenodo.org/record/7338613)
- Environment: torch 2.x and timm 1.0. The released code targets timm 0.3.2, and the only API difference used here
  is `timm.layers.trunc_normal_`.

## Protocol: kept from the released code

- `ToTensor()` only, with no normalization. Input 256×256, batch size 16, 100 epochs, Adam lr 1e-3,
  `CosineAnnealingWarmRestarts(T_0=5)`.
- The backbone is frozen and only a linear head is trained. The final-epoch weights are evaluated.
- SatMAE: the backbone runs in `eval()`, with a separate `nn.Linear(1024, 30)` head, as in `SatMAE_finetune.py`.
- Scale-MAE: the backbone runs in `train()`, so drop_path 0.1 is active during training. It uses `model.head` with
  `trunc_normal_(std=0.01)` and `input_res = 1`, as in `scaleMAE_finetune.py`.
- Checkpoint loading uses the same key-removal code as the released scripts.
- Corrupted test set: the accuracy for each corruption is averaged over severities 1–5, and the "corruption avg" is
  the mean over the 12 corruptions.

## Changes from the released code

1. `del model.head` followed by `model(data)` raises `AttributeError` in the SatMAE script. SatMAE features are
   taken with `forward_features()` instead, which gives the CLS token after the final norm.
2. SatMAE checkpoint: the RGB `fmow_pretrain.pth` is used. The released path (`pretrain-vit-large-e199.pth`) is the
   multispectral model, and its result is reported separately (18.2).
3. `data_type/AID_*_data.pth` is not released, so images are read from the released split with
   `ImageFolder` + `ToTensor()`.
4. `A0ep` steps the scheduler once per epoch, following Appendix F.1. `A0` keeps the released per-iteration step.
5. Speed only: in `C_ablation`, eval-mode features are cached and the frozen backbone runs under bf16 autocast.
   `A_paper` runs everything in fp32 (TF32), end to end, and gives the same picture (SatMAE 90.4, Scale-MAE 91.25,
   seed 0).
