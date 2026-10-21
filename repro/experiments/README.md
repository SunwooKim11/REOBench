# Experiments (results only)

Each folder holds the outputs of one experiment. The code that produced it is under `scripts/`, and the findings are
in `docs/NOTES_tta.md`.

| Folder | What | Script | Key result (details in NOTES_tta.md) |
|---|---|---|---|
| `2026-10-12_cls_released` | AID, REOBench released protocol, end to end (incl. the Sentinel-checkpoint run) | `scripts/cls/train_released.py` | SatMAE 90.4 / Scale-MAE 91.25 clean (paper 72.05 / 75.75); Sentinel ckpt 18.2 |
| `2026-10-12_cls_official` | AID, SatMAE / Scale-MAE official linear-probe protocol | `scripts/cls/train_official.py` | SatMAE 91.30 / Scale-MAE 93.85 clean (`*_results.txt`) |
| `2026-10-12_cls_ablation` | One-factor ablation A → B, 3 seeds; **P / PN heads used by all TTA runs** | `scripts/cls/ablation.py` | `summary.txt` (A0ep 90.7 / 91.4) |
| `2026-10-13_cls_tta` | AID TTA, mild protocol: 9 methods × {SatMAE, ScaleMAE} × {P, PN}, lr sweep, SAR reset 0.2, CoTTA diagnostics, PLPD / TRAP analysis | `scripts/tta/cls_tta.py`, `scripts/analysis/` | `summary_tta.txt`, `summary_lr.txt` |
| `2026-10-14_verify_imagenetc` | ImageNet-C check of the TTA implementations against the papers | `scripts/verify/imagenetc.py` | `summary.txt` (DeYO T1 within 0.35, ViDA 43.29 vs 43.4) |
| `2026-10-16_seg_baseline` | Potsdam, settings A (released) and B (fixed), 8 released corruptions | `scripts/seg/` | `summary.txt` (SatMAE B 61.90 / ScaleMAE B 64.97 clean) |
| `2026-10-20_seg_tta` | Potsdam TTA, mild protocol (none, bnadapt, isa, tent, sar, cotta) | `scripts/seg/seg_tta.py` | `outputs/` (best: Tent + test-batch BN +3.8 / +2.7 mIoU; ISA +2.6 / +3.2) |

The segmentation-TTA check on Cityscapes→ACDC and the official CoTTA run keep their logs inside `third_party/`.
Their numbers are recorded in NOTES_tta.md §2.
