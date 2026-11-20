"""How much source information do ISA and feature alignment need? (AID, PN heads, scripts/archive/tta/run_src_ablation.sh)

Rows per model: no adaptation, then each method with source statistics from 2,000 clean train images (default), from 32
images (_n32), and source-free ISA whose target is the pretraining Normalize constants shipped with the model (_isapre).
Groups as in summarize_depth_sweep.py. Other rows = change vs none.

  ~/REOBench/.venv/bin/python scripts/analysis/summarize_src_ablation.py   # -> experiments/2026-10-13_cls_tta/summary_src_ablation.txt
"""
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):  # reuse the group definitions; silence its own tables
    from summarize_depth_sweep import cls_scores, table
from reobench_tta import paths  # noqa: E402

ROWS = [("none", "none_s0"),
        ("ISA (2000)", "isa_batch_s0"), ("ISA (32)", "isa_batch_s0_n32"), ("ISA (pretrain)", "isa_batch_s0_isapre"),
        ("blk7 (2000)", "fsa_s0_blocks.7"), ("blk7 (32)", "fsa_s0_blocks.7_n32"),
        ("headin (2000)", "fsa_s0_headin"), ("headin (32)", "fsa_s0_headin_n32"),
        ("ISA+blk7 (2000)", "fsa_s0_blocks.7_isa"), ("ISA+blk7 (32)", "fsa_s0_blocks.7_isa_n32"),
        ("ISA+blk7 (pre)", "fsa_s0_blocks.7_isa_isapre")]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    for m in ["satmae", "scalemae"]:
        d = paths.CLS_TTA / "outputs" / m / "PN"
        rs = {n: json.load(open(d / f"{f}.json")) for n, f in ROWS if (d / f"{f}.json").exists()}
        rows = {n: cls_scores(r) for n, r in rs.items() if "avg" in r}  # skip runs still in progress
        table(f"{m} PN, AID, source statistics from N clean train images", rows)
print(buf.getvalue())
open(paths.CLS_TTA / "summary_src_ablation.txt", "w").write(buf.getvalue())
