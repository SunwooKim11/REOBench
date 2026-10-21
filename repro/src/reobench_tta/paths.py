"""Repository locations, resolved from this file so they survive folder renames of the parent directories."""
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]          # repro/src
REPO = SRC.parent                                   # repro/
REOBENCH = REPO.parent                              # REOBench checkout (released Classification/, Segmentation/)
EXP = REPO / "experiments"
THIRD = REPO / "third_party"

# experiment folders (results only; see experiments/README.md)
CLS_RELEASED = EXP / "2026-10-12_cls_released"
CLS_OFFICIAL = EXP / "2026-10-12_cls_official"
CLS_ABLATION = EXP / "2026-10-12_cls_ablation"      # also holds the P / PN heads used by TTA
CLS_TTA = EXP / "2026-10-13_cls_tta"
VERIFY_IMAGENETC = EXP / "2026-10-14_verify_imagenetc"
