# REOBench reproduction and test-time adaptation (TTA)

This folder holds the reproduction of the REOBench baselines (AID classification, Potsdam segmentation) and the TTA
experiments built on top of them. The folders are lettered in the order the work was done, and each one is a
self-contained stage with its own run scripts and results.

| Stage | Folder | What it is | Main entry | Results |
|---|---|---|---|---|
| A | `A_paper/` | AID classification, released REOBench protocol, end to end | `run.py` | `outputs/*_results.txt` |
| B | `B_official/` | AID classification, SatMAE / Scale-MAE official linear-probe protocol | `run.py` | `outputs/` |
| C | `C_ablation/` | One-factor ablation A → B (3 seeds); also trains the heads that D uses (`P`, `PN`) | `run.py`, `jobs*.sh` | `summary.txt` |
| D | `D_tta/` | TTA on AID: `methods/` (Tent, SAR, DeYO, EATA, CoTTA, ViDA, LSCD, ISA) + protocol `tta.py` | `tta.py`, `run_*.sh` | `summary_tta.txt`, `summary_lr.txt` |
| E | `E_verify/` | ImageNet-C check of `D_tta/methods` against the papers | `imagenetc.py`, `run_verify.sh` | `summary.txt`, `REFERENCES.md` |
| F | `F_cotta_official/` | Official CoTTA repo run in our environment (only our scripts and patch are tracked) | `imagenet/run_official.sh` | |
| G | `G_seg_verify/` | Segmentation TTA check on Cityscapes→ACDC (SVDP copy of the CoTTA code; patched files only) | `run_*.sh` | `README_REOBench.md` |
| H | `H_seg_baseline/` | Potsdam segmentation baseline, A (released config, paper schedule) and B (fixed) | `run_train.sh`, `eval_all.sh` | `summary.txt` |

Other folders:
- `common.py`: data, backbone and feature code shared by A to E.
- `docs/`: `NOTES_tta.md` (all results by topic, start here), `NOTES_concepts.md`, `issue4/` (REOBench issue #4
  drafts and `issue_check.py`), `archive/`.
- `review/`: code-review guide, pinned upstream sources and diffs of our changes (`REVIEW_GUIDE.md`).
- `scripts/archive/`: one-off GPU queue scripts that have already been run. They are kept for the record, and their paths are no longer valid.

Conventions:
- Every change to third-party code is marked `[REOBench]` or `[REOBench repro]`.
- Generated files (logs, checkpoints, feature caches) are git-ignored. Result JSONs are tracked.
- Python: `~/REOBench/.venv` (A to F); conda env `segtta` (G, H). See `docs/NOTES_tta.md` §2.2.
