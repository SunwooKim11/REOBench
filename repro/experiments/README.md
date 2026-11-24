# Experiments (results only)

Each folder holds the outputs of one experiment. The code that produced it is under `scripts/`, and the findings are
in `docs/NOTES_tta.md` (2026-10) and `docs/feedback/2026-11-16_direction.md` (2026-11 progress log).

| Folder | What | Script | Key result (details in NOTES_tta.md) |
|---|---|---|---|
| `2026-10-12_cls_released` | AID, REOBench released protocol, end to end (incl. the Sentinel-checkpoint run) | `scripts/cls/train_released.py` | SatMAE 90.4 / Scale-MAE 91.25 clean (paper 72.05 / 75.75); Sentinel ckpt 18.2 |
| `2026-10-12_cls_official` | AID, SatMAE / Scale-MAE official linear-probe protocol | `scripts/cls/train_official.py` | SatMAE 91.30 / Scale-MAE 93.85 clean (`*_results.txt`) |
| `2026-10-12_cls_ablation` | One-factor ablation A → B, 3 seeds; **P / PN heads used by all TTA runs** | `scripts/cls/ablation.py` | `summary.txt` (A0ep 90.7 / 91.4) |
| `2026-10-13_cls_tta` | AID TTA, mild protocol: 9 methods × {SatMAE, ScaleMAE} × {P, PN}, lr sweep, SAR reset 0.2, CoTTA diagnostics, PLPD / TRAP analysis | `scripts/tta/cls_tta.py`, `scripts/analysis/` | `summary_tta.txt`, `summary_lr.txt` |
| `2026-10-14_verify_imagenetc` | ImageNet-C check of the TTA implementations against the papers | `scripts/verify/imagenetc.py` | `summary.txt` (DeYO T1 within 0.35, ViDA 43.29 vs 43.4) |
| `2026-10-16_seg_baseline` | Potsdam, settings A (released) and B (fixed), 8 released corruptions | `scripts/seg/` | `summary.txt` (SatMAE B 61.90 / ScaleMAE B 64.97 clean) |
| `2026-10-20_seg_tta` | Potsdam TTA, mild protocol (none, bnadapt, isa, tent, sar, cotta) | `scripts/seg/seg_tta.py` | `outputs/` (best: Tent + test-batch BN +3.8 / +2.7 mIoU; ISA +2.6 / +3.2) |
| `2026-10-22_feature_shift` | SatMAE / ScaleMAE: how each corruption group moves the features (AID, severity 5) | `scripts/analysis/feature_shift.py` | `summary.txt` |
| `2026-10-28_entropy_bias` | Low-entropy samples and prediction bias of the source model (SatMAE PN) | `scripts/analysis/entropy_bias.py` | `summary.txt` |
| `2026-11-16_backbone_check` | Sanity check of the 17 backbones (kNN on AID, feature scale) | `scripts/tools/check_backbones.py` | `summary.txt` |
| `2026-11-16_backbone_feats` | Stage 1: saved features, linear heads, shift metrics and label-free signals on AID-C / RESISC45-C; seg drift, RRSHID | `scripts/analysis/` (`extract_feats.py`, `feat_metrics.py`, ...) | `RESULTS.md`, `ANALYSIS.md` |
| `2026-11-17_natural_shift` | Natural shift RESISC45 ↔ UC Merced (EarthShift classes) | `scripts/analysis/natural_shift.py` | `RESULTS.md` |
| `2026-11-17_shift_oracle` | Oracle removal of the shift direction (feature and pixel level) | `scripts/analysis/shift_direction_oracle.py` | `RESULTS.md` |
| `2026-11-17_flip_tta` | Feature-level TTA on saved features: alignment, flip filter, tent (bs 16 / 64) | `scripts/analysis/feat_tta.py` | `bs*/`, summary script `summarize_flip_tta.py` |
| `2026-11-17_fm_tta` | Full-backbone TTA on the 17 backbones (tent, sar, eata, flip filter, dsbr, ± alignment) | `scripts/tta/cls_tta_fm.py` | `RESULTS.md` |
| `2026-11-18_head_subspace` | Which part of the shift the linear head reads (head subspace energy) | `scripts/analysis/head_subspace.py` | per-model JSON |
| `2026-11-19_weighted_flips_repro` | Reproduction of Weighted Flips (Entropy Enigma, ImageNet-C, ResNet-50) | `scripts/verify/weighted_flips_repro.py` | `RESULTS.md` |
| `2026-11-19_dsbr` | DSBR: feature level, prediction bias, ImageNet-C check (full-backbone runs in `2026-11-17_fm_tta`) | `scripts/tta/cls_tta_fm.py`, `scripts/verify/dsbr_imagenet_c.py` | `RESULTS.md`, `pred_bias.md` |
| `2026-11-23_head_nostd` | Head-input alignment with a head trained on unstandardized features (head-recipe control) | `scripts/analysis/head_nostd.py` | `RESULTS.md` |
| `2026-11-23_seg_signals` | Potsdam segmentation: label-free signals (flip, confidence, prediction bias) and flip filter / DSBR | `scripts/seg/seg_signals.py`, `scripts/seg/seg_tta.py` | `RESULTS.md` |
| `2026-11-24_cluster_cot` | Entropy Enigma checks on the 17 backbones: COT accuracy estimate, Silhouette / Shift distance vs accuracy, k-means vs linear head, two phases of entropy minimisation (tent vs DSBR) | `scripts/analysis/cluster_cot.py`, `cluster_subspace.py`, `em_phases.py`, `plot_feature_dist.py` | `RESULTS.md`, `RESULTS_subspace.md`, `RESULTS_em_phases.md` |

The segmentation-TTA check on Cityscapes→ACDC and the official CoTTA run keep their logs inside `third_party/`.
Their numbers are recorded in NOTES_tta.md §2.
