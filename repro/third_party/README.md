# Vendored third-party code

These are copies of other repositories that must run from their own folder layout (mmseg / robustbench style).
Only our scripts and the files we patched are tracked in git (see `.gitignore`). To rebuild a folder, clone the
source at the listed commit and copy the tracked files over it. Every change is marked `[REOBench]` or
`[REOBench repro]`; diffs are in `review/diffs/`.

| Folder | Source | What we changed | Used for |
|---|---|---|---|
| `cotta_official/` | qinenergy/cotta @ c212a20 (`imagenet/`) | `cotta.py`: torchvision `RandomAffine` API (one line); `cfgs/std_*.yaml`: standard corruption order; `run_official.sh` | ImageNet-C: our CoTTA port vs the official code in the same environment (66.73 vs 66.56) |
| `cotta_acdc_svdp/` | Anonymous-012/SVDP (copy of CoTTA's Cityscapes→ACDC code; the original is OneDrive-only) | 6 patches listed in `README_REOBench.md` + entropy Tent (`single_gpu_tent_entropy`, `tools/tent_entropy.py`) | Segmentation TTA check: source, PL-TENT, entropy Tent, CoTTA vs CoTTA Table 5 |
| `reobench_seg/` | this repo's `Segmentation/` (mmseg 0.21, RVSA based) | build_backbone `.model`, configurable checkpoint / feature blocks, pos-embed interpolation, label 255 → ignore, mmcv max version; configs in `configs/repro/` | Potsdam baselines A / B and segmentation TTA |
| `satdino/` | strakaj/SatDINO @ b033d3f (`satdino/__init__.py`, `utils.py`, `vision_transformer_satdino.py`, `LICENSE`, Apache-2.0) | nothing (byte-identical) | SatDINO backbones in `src/reobench_tta/backbones.py` |

Environment for the two segmentation copies: conda `segtta` (python 3.10, torch 2.0.1+cu118, mmcv-full 1.7.2).
Set `TIMM_FUSED_ATTN=0` for `reobench_seg` (torch 2.0.1 fused attention backward crashes on the H200).
