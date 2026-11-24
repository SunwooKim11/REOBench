# REOBench reproduction and test-time adaptation (TTA)

This folder reproduces the REOBench baselines (AID classification, Potsdam segmentation), runs TTA experiments on top of
them, and since 2026-11 studies label-free shift signals on 17 frozen backbones (AID-C, RESISC45-C). The layout follows
the usual research-repo split: logic in one importable package, thin entry scripts, results kept apart from code, and
notes kept apart from results.

```
repro/
├── src/reobench_tta/        library code: the only place for shared logic
│   ├── common.py            datasets (AID, corruptions), REOBench backbones, feature extraction, evaluation helpers
│   ├── backbones.py         the 17 frozen backbones of the 2026-11 study (ImageNet / fMoW / satellite pretraining pairs
│   │                        and supervised ImageNet anchors); weights via scripts/tools/download_backbones.sh
│   ├── paths.py             repository / data / experiment locations
│   └── tta/                 TTA methods: tent, sar (+sam), deyo, eata, cotta, lscd, isa, stat_align, params
├── scripts/                 entry points, run from repro/
│   ├── cls/                 classification heads: train_released (REOBench protocol), train_official, ablation
│   ├── tta/                 cls_tta.py (AID, SatMAE/ScaleMAE), cls_tta_fm.py (17 backbones, full backbone), summaries
│   ├── seg/                 Potsdam: train / eval / seg_tta / summaries
│   ├── analysis/            feature extraction and analyses on saved features (stage-1 metrics, label-free signals,
│   │                        flip filter / feature-level TTA, shift oracle, head subspace, prediction bias, ...)
│   ├── verify/              checks against published numbers: imagenetc.py (our TTA methods), weighted_flips_repro.py
│   │                        (Entropy Enigma), dsbr_imagenet_c.py (DSBR); assets/
│   ├── tools/               data and backbone preparation (corrupted sets, splits, downloads, checks)
│   └── archive/             launchers (run_*.sh) of runs that have finished, by topic; see archive/README.md
├── third_party/             vendored code; only the files we patched are tracked (see third_party/README.md)
├── experiments/             results only, one folder per experiment: <date>_<name>/ (index: experiments/README.md)
├── docs/                    notes (index below)
└── review/                  code-review guide, pinned upstream sources, diffs of our changes
```

Outside this folder: data under `/data/rs` (saved features `/data/rs/feats/<model>/`), ImageNet-C under `/data/tta/`,
paper PDFs under `~/REOBench/papers/`.

## docs/

| Path | Content |
|---|---|
| `NOTES_tta.md` | all results of the TTA phase (2026-10) by topic; start here for that phase |
| `NOTES_concepts.md` | concepts explained along the way |
| `SETTINGS.md` | the single file for settings: backbones (pretraining data, augmentation), datasets, corruptions and how they were generated, scope of each experiment |
| `STUDY_*.md` | study notes (alignment, stage-1 metrics) |
| `PRIOR_ART_*.md` | prior-art surveys (statistics alignment, feature signals, CLIP segmentation TTA, SAE pruning) |
| `PAPERS_tta_2026-10.md`, `PPT_prompt_2026-10.md` | paper list and slide prompt of the TTA phase |
| `reading/` | one note per paper (index: `reading/README.md`) |
| `feedback/` | advisor feedback, one file per meeting, plus the running progress log `2026-11-16_direction.md` |
| `verify/`, `seg/`, `issue4/` | verification references, segmentation study notes, draft of the REOBench GitHub issue |
| `figs/`, `archive/` | figures for meetings, superseded notes |

## Quick start (from `repro/`)

```bash
PY=~/REOBench/.venv/bin/python
$PY scripts/cls/ablation.py --prep --model satmae --norm                  # feature cache
$PY scripts/cls/ablation.py --model satmae --name PN --norm --optim adam_wr_epoch --save_head
$PY scripts/tta/cls_tta.py --model satmae --head_cfg PN --method tent      # -> experiments/2026-10-13_cls_tta/
$PY scripts/tta/cls_tta_fm.py --datasets aid --methods tent dsbr          # 17 backbones -> experiments/2026-11-17_fm_tta/
$PY scripts/verify/imagenetc.py --setting deyo_mild_r50 --arch r50 --norm imagenet --method tent
```

## Conventions
- Logic lives in `src/` or in the entry script that uses it. The feature helpers in `scripts/analysis/`
  (`extract_feats.py`, `feat_metrics.py`) are also imported by `scripts/tta/` and are the one exception.
- Results go to `experiments/<date>_<name>/` (JSON and `RESULTS.md` tracked; logs, checkpoints and caches git-ignored).
  `RESULTS.md` holds numbers only; interpretation goes to `docs/`.
- Long runs are started from a `scripts/run_*.sh` launcher in tmux; once the run has finished the launcher moves to
  `scripts/archive/<topic>/`. Launchers `cd` to `repro/` themselves, so they keep working from the archive.
- Every change to third-party code is marked `[REOBench]` or `[REOBench repro]`.
- Environments: `~/REOBench/.venv` (classification, features, ImageNet-C) and conda `segtta` (segmentation; torch 2.0.1 +
  mmcv-full 1.7.2, needed for the H200).
