# Archived launchers

These shell scripts launched experiments that have already finished. They are kept as a record of the exact commands
and orderings, but they predate the `repro/` restructuring (2026-10-21), so their relative paths (`D_tta/`,
`C_ablation/`, `E_verify/`, `../C_ablation/outputs`) no longer exist. To rerun something, use the current entry points
in `scripts/` with the same arguments. The argument lists are in these files and in `scripts/cls/ablation_jobs*.sh`.

- `cls/`: classification ablation launchers (`one.sh`, `run_all.sh`, `run_sched.sh`)
- `tta/`: AID TTA launchers (head training, main runs, lr sweep, new baselines, CoTTA reruns and diagnostics, SAR reset 0.2)
- `verify/`: ImageNet-C verification launcher
- `run_queue*.sh`: GPU queues used after the CoTTA OOM

## Launchers of the 2026-10-22 ~ 2026-11-20 runs (moved here on 2026-11-20)

These still work: each one does `cd "$(dirname "$0")/../../.."`, i.e. to `repro/`, before anything else (or
`cd ~/REOBench/repro`). Run them from `repro/` as `bash scripts/archive/<topic>/<name>.sh`.

| Launcher | Results |
|---|---|
| `seg/run_depth_sweep.sh`, `seg/run_combo.sh` | `experiments/2026-10-20_seg_tta/` (alignment position sweep, input + feature alignment) |
| `tta/run_src_ablation.sh` | `experiments/2026-10-13_cls_tta/` (how much source information ISA / alignment need) |
| `analysis/run_extract_feats.sh`, `analysis/run_feat_metrics.sh`, `analysis/run_stage1_extra.sh`, `analysis/run_seg_drift.sh` | `experiments/2026-11-16_backbone_feats/` |
| `analysis/run_natural_shift.sh` | `experiments/2026-11-17_natural_shift/` |
| `analysis/run_oracle_pixel.sh` | `experiments/2026-11-17_shift_oracle/` |
| `tta/run_flip_tta.sh` | `experiments/2026-11-17_flip_tta/` |
| `tta/run_fm_tta.sh`, `tta/run_fm_tta_eata.sh` | `experiments/2026-11-17_fm_tta/` |
| `verify/run_wf_repro.sh` | `experiments/2026-11-19_weighted_flips_repro/` |
| `tta/run_dsbr.sh`, `verify/run_dsbr_norm.sh`, `verify/run_dsbr_match.sh` | `experiments/2026-11-19_dsbr/` (+ `2026-11-17_fm_tta/` for the full-backbone DSBR runs) |
| `seg/run_seg_nov.sh` | `experiments/2026-11-23_seg_signals/` |
| `analysis/run_cluster_cot.sh`, `analysis/run_em_phases.sh` | `experiments/2026-11-24_cluster_cot/` |
