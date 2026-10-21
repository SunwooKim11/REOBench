# Code review guide (what to read, in what order)

## 1. Folder map

See `repro/README.md` for the full layout. Only **two places hold our logic**:

| Path | Role | Review? |
|---|---|---|
| `src/reobench_tta/common.py` | AID data, backbones, feature extraction, evaluation | ★ base of every classification experiment |
| `src/reobench_tta/tta/` | **TTA method implementations** (Tent, SAR, DeYO, EATA, CoTTA, ViDA, LSCD, ISA) | ★★ main review target |
| `scripts/tta/cls_tta.py` | AID TTA protocol: per-(corruption, severity) reset, lr, method arguments | ★★ main review target |
| `scripts/seg/seg_tta.py` (now `H_seg_baseline/seg_tta.py`) | Potsdam TTA protocol and the segmentation versions of the methods | ★★ |
| `scripts/cls/`, `scripts/verify/`, `scripts/analysis/` | entry points: thin, mostly argument parsing and loops | as needed |
| `third_party/` | vendored code; only patched files tracked | patches only: `review/diffs/` |
| `experiments/` | results only | no |

## 2. How to check the code against the official code (no need to read whole files)

- `review/upstream/*__<repo>.py`: the original files, at the commits in `COMMITS.txt`.
- `review/diffs/<method>.diff`: what we changed. In VS Code, select two files and choose "Compare Selected".
- Every change in our files is marked `[REOBench]`: `grep -rn "\[REOBench\]" src/reobench_tta/tta`.

| Method | Our file | Original | Changed lines | What changed |
|---|---|---|---|---|
| EATA | `eata.py` | DeYO repo `methods/eata.py` | **0** | Same as the original |
| SAM (SAR optimizer) | `sam.py` | DeYO repo `methods/sam.py` | **0** | Same as the original |
| Tent / SAR / DeYO | `tent.py`, `sar.py`, `deyo.py` | DeYO repo | ~30 each | `collect_params` only. The hard-coded ViT-B rule (skip blocks 9-11) becomes "skip the top 25% of blocks" (`params.py`), so ViT-L skips blocks 18-23. The loss and the update are unchanged |
| CoTTA | `cotta.py`, `my_transforms.py` | CoTTA repo `imagenet/` | 16 / 0 | Image size made a variable (256 for AID), torchvision API for `RandomAffine`, augmentation threshold `ap` made an argument, a counter added |
| ViDA | `vida.py` | ViDA repo `imagenet/vida.py` + `inject_vida.py` | (two files merged) | The two files are merged into one; the high-rank scale bug is kept by default (`fix_hka`); checkpoint key mapping; memory frees. Details: `docs/verify/VIDA_SPEC.md` |
| LSCD | `lscd.py` | LSCD repo `models/cmp_losses.py` (loss only) | – | The loss is copied as is. The official loop has a SyntaxError and does not run, so the loop follows Tent |
| ISA | `isa.py` | – (our own baseline) | – | About 10 lines; read the whole file |

## 3. Checklist against the papers

The diffs show where we differ from the official code. These checks are about where the **official code and the
paper** differ from each other:

1. **Tent** (paper §3): entropy loss, normalization-layer affine parameters only, prediction taken before the update.
   - Compare `tent.py` `forward_and_adapt` with the paper.
2. **SAR** (paper Eq. 2-5): reliable-sample filter E0 = 0.4 ln C, SAM ρ = 0.05, model recovery (EMA < 0.2).
   - **Discrepancy found**: the official SAR (`mr-eggplant/SAR` `sar.py` L102) hard-codes the reset as `if ema < 0.2`.
     The DeYO repo copy we use resets at `ema < reset_constant_em` with a default of **0.005**, so model recovery
     almost never fires. Our ImageNet-C SAR matches DeYO Table 1, which was run with this same 0.005 value.
     The SAR paper's own value is 0.2.
3. **DeYO** (Eq. 8-11): τ_Ent = 0.5 ln C, PLPD > 0.3 (mild), weight 1/exp(Ent−Ent0) + 1/exp(−PLPD), 4×4 patch shuffle.
4. **EATA**: E0 = 0.4 ln C, ε = 0.05, Fisher β = 2000 computed on 2,000 source images. In `tta.py`, the Fisher block
   uses clean AID train images.
5. **CoTTA**: EMA 0.999, stochastic restore 0.001, 32 augmentations on low-confidence batches.
   - `tta.py` defaults p_th to the 5% quantile of source confidence, while the official ImageNet code uses 0.1 (see `docs/NOTES_tta.md` §3.3).
6. **ViDA**: the paper and the code differ in six places (`docs/verify/VIDA_SPEC.md`, table "Paper vs code").

## 4. Protocol (`scripts/tta/cls_tta.py`)

| Item | Location |
|---|---|
| Stream construction (2,000 images × (corruption, severity), shuffle, batch 64) | the "streams" block |
| Reset before every stream | `make_adapter()` |
| lr = 0.001/64 × bs; CoTTA 0.01 | `DEFAULT_LR` |
| Thresholds × ln C (C = 30) | inside `make_adapter()` |
| Online accuracy (prediction before the update) | main loop: `out = adapt(x, i)` |

## 5. How the implementations were checked (results only)

- **ImageNet-C** (`experiments/2026-10-14_verify_imagenetc/summary.txt`):
  - Tent, EATA, SAR and DeYO are within 0.35 of DeYO Table 1.
  - ViDA: 43.29 vs 43.4.
  - CoTTA matches the official code run in our environment (66.73 vs 66.56).
- **Segmentation** (`docs/NOTES_tta.md` §2.2, §5.2):
  - The source model matches CoTTA Table 5.
  - The PL "TENT" 10-round mean is 52.74 vs 52.3.
  - Entropy Tent with batch BN statistics (Δ −2.5) matches Yi et al. (Δ −2.5).

## 6. Segmentation (current state)

**Status: the RS segmentation TTA is not built yet.** All the segmentation code that exists today is the
Cityscapes→ACDC verification code in `third_party/cotta_acdc_svdp/`. It is CoTTA's ACDC code (SVDP copy) with mmseg 0.x and SegFormer-B5.

### 6.1 What to read in `third_party/cotta_acdc_svdp/` (verification code)

| What | Where | Note |
|---|---|---|
| Entry scripts | `tools/tent.py` (`--config/--checkpoint`), `tools/cotta.py` (positional), `tools/tent_entropy.py` | Rounds loop: `ROUNDS` env, default 10; 4 ACDC conditions in sequence, no reset |
| PL "TENT" (CoTTA code) | `mmseg/apis/test.py` `single_gpu_tent` (L298) | Prediction → CE on its own argmax; params with "norm"/"bn" in name; Adam 6e-5/8 |
| Original Tent (ours) | `mmseg/apis/test.py` `single_gpu_tent_entropy` (L611) | Pixel-entropy mean; same params/optimizer; `TENT_BN=batch` puts BN on test statistics (`tools/tent_entropy.py` reverts SyncBN→BN) |
| CoTTA (CoTTA code) | `mmseg/apis/test.py` `single_gpu_cotta` (L197); teacher/anchor creation `tools/cotta.py` L17, L226-228 | 14-view aug (`tools/cotta.py` L118-124, `encoder_decoder.aug_test` L355); anchor prob > 0.69 → single-scale label; EMA 0.999; restore 0.01 |
| Our patches vs SVDP copy | `review/diffs/seg__*.diff`, list in `third_party/cotta_acdc_svdp/README_REOBench.md` | 6 patches, all marked `[REOBench]` |

### 6.2 What still has to be built for RS segmentation

1. **Baseline first**: reproduce REOBench segmentation (Potsdam, UperNet, `~/REOBench/Segmentation/`) and check the two
   suspected issues (SatMAE normalization on a 0-255 scale, Sentinel checkpoint) — same as was done for AID.
2. **TTA on top**: REOBench segmentation is mmseg **0.21** (RVSA-based, mmcv-full 1.3.13–1.5.0), the same mmseg 0.x
   family as the verification code, so the `single_gpu_*` functions can be carried over almost as is.
   The differences are the ViT-L backbones and the UperNet + FCN head. The `segtta` env needs only the
   MMCV_MAX bump:
   - Tent: pixel entropy (validated in `single_gpu_tent_entropy`), decide BN handling of the UperNet decoder
   - CoTTA: multi-scale+flip teacher (validated logic in `single_gpu_cotta`)
   - ViDA: CoTTA segmentation loop + ViDA adapters (`src/reobench_tta/tta/vida.py` injection)
   - ISA: input transform, reuse `src/reobench_tta/tta/isa.py`
