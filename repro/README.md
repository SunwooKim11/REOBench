# REOBench reproduction and test-time adaptation (TTA)

This folder reproduces the REOBench baselines (AID classification, Potsdam segmentation) and runs TTA experiments on
top of them. The layout follows the usual research-repo split: logic in one importable package, thin entry scripts,
and results kept apart from code.

```
repro/
├── src/reobench_tta/      library code: the only place for logic
│   ├── common.py          AID data, REOBench backbones, feature extraction, evaluation helpers
│   ├── paths.py           repository / experiment locations
│   └── tta/               TTA methods: tent, sar (+sam), deyo, eata, cotta, vida, lscd, isa, params
├── scripts/               entry points (run from repro/)
│   ├── cls/               classification heads: train_released (REOBench protocol), train_official, ablation
│   ├── tta/               cls_tta.py (AID TTA, mild protocol), summaries
│   ├── seg/               Potsdam: train / eval / seg_tta / summaries            (moving from H_seg_baseline/)
│   ├── verify/            imagenetc.py (+ assets/): ImageNet-C check of src/reobench_tta/tta
│   ├── analysis/          diag, plpd_analysis, trap_analysis
│   ├── tools/             recover_from_log
│   └── archive/           launchers that have already been run (kept for the record; their paths are outdated)
├── third_party/           vendored code; only the files we patched are tracked (see third_party/README.md)
│   ├── cotta_official/    qinenergy/cotta (ImageNet-C), run in our environment
│   ├── cotta_acdc_svdp/   CoTTA's Cityscapes->ACDC code (SVDP copy), segmentation TTA check
│   └── reobench_seg/      REOBench Segmentation/ (mmseg 0.21) with our fixes   (moving from H_seg_baseline/seg)
├── experiments/           results only, one folder per experiment (see experiments/README.md)
├── docs/                  NOTES_tta.md (all results by topic; start here), NOTES_concepts.md, verify/, issue4/
└── review/                code-review guide, pinned upstream sources, diffs of our changes
```

## Quick start (from `repro/`)

```bash
PY=~/REOBench/.venv/bin/python
$PY scripts/cls/ablation.py --prep --model satmae --norm                  # feature cache
$PY scripts/cls/ablation.py --model satmae --name PN --norm --optim adam_wr_epoch --save_head
$PY scripts/tta/cls_tta.py --model satmae --head_cfg PN --method tent      # -> experiments/2026-10-13_cls_tta/
$PY scripts/tta/summarize_cls_tta.py
$PY scripts/verify/imagenetc.py --setting deyo_mild_r50 --arch r50 --norm imagenet --method tent
```

## Conventions
- Every change to third-party code is marked `[REOBench]` or `[REOBench repro]`.
- Results go to `experiments/<date>_<name>/` (JSON tracked; logs, checkpoints and caches git-ignored).
- Environments: `~/REOBench/.venv` (classification, ImageNet-C) and conda `segtta` (segmentation; torch 2.0.1 + mmcv-full 1.7.2,
  needed for the H200).
