# Prior art: TTA for CLIP-based segmentation (survey by agent, 2026-10-22)

## Conclusion
- **Every CLIP segmentation TTA found is zero-shot / open-vocabulary (8 methods).** All of them assume a classifier made of text embeddings.
- **No dedicated method exists for the closed-set "CLIP image encoder + trained decoder (UperNet)" setting that REOBench uses.**
  Once a decoder replaces the text tower, the model is an ordinary ViT segmenter, so the standard segmentation TTA methods apply.

## SegEarth-OV (CVPR 2025), checked against the paper and code

The user's description is correct. These details were missing:
- **Backbone:** the main results use OpenAI CLIP ViT-B/16. RemoteCLIP, GeoRSCLIP and SkyCLIP appear only in a side table, at B/32, each with its own SimFeatUp.
- **SimFeatUp is trained once, without labels.**
  - Data: 16k random Million-AID images, 1 epoch, 224 crops, CLIP frozen.
  - Loss: FeatUp-style multi-view reconstruction plus image reconstruction (γ = 0.1) through a content-retention network.
  - Design: one parameterized JBU applied repeatedly, 11×11 kernel.
  - The same weights are used for every dataset. "Training-free" means no segmentation labels and no per-dataset training.
- **Last-block surgery (ClearCLIP/SCLIP style):** the FFN and residual are removed, and attention becomes q-q + k-k + v-v.
- **Global Bias Alleviation:** Ô = O_patch − 0.3·O_CLS (subtracts the CLS token).
- **Inference:**
  - Features: taken at the input of the last block, after projection.
  - Resizing: long side 448; sliding window 224 with stride 112.
  - Prompt: "a photo of a {}".
  - Decision: cosine similarity with the text embeddings → argmax.
  - Class names: renamed, with synonym sets (e.g. clutter→background), taking the max over synonyms.
- **No TTA and no corruption experiments.**

## 1. TTA / test-time enhancement for open-vocabulary CLIP segmentation

| Method | Venue | What is adapted | Benchmark, RS / corruptions |
|---|---|---|---|
| MLMP | NeurIPS 2025 | Vision-encoder LN; entropy fused over intermediate layers and prompts | 87 scenarios (15 ImageNet-C corruptions + ACDC, GTA-V), host NACLIP L/14; no RS |
| Seg-TTO | arXiv 2025 | Text prompts plus visual feature aggregation | MESS 22 datasets (includes Potsdam, iSAID); no corruptions |
| DAF | arXiv 2026 | CLIP LN, continual | **LoveDA-C** (source 9.2, TENT 1.2, SAR 7.6, MLMP 5.2, DAF 14.9) |
| **TMPA** | **CVPR 2026** | **Text only** (LLM descriptions + intensity α), 3 steps per image | 17 RS datasets (the SegEarth-OV suite); **no corruptions**. **The only RS TTA found** |
| TPA | arXiv 2026 | No gradients; DINO prototypes | no corruptions |
| Balanced Prompt Adaptation | arXiv 2026 | Text prompt residuals | binary segmentation [uncertain CLIP] |
| FLOSS | ICCV 2025 | Template selection by entropy | |
| SeeCo | arXiv 2026 | On-the-fly recalibration | 8 RS OVSS sets; no corruptions |

**Training-free architecture tweaks (not per-sample adaptation):** MaskCLIP, SCLIP, ClearCLIP, ProxyCLIP, NACLIP, CLIP-DIY, CASS, SegEarth-OV. These act as the host models that TTA methods are added to.

**VLM segmentation corruption benchmarks:** MLMP's suite (natural images) and DAF's **LoveDA-C**, the only RS one found.

## 2. RS open-vocabulary segmentation family

SegEarth-OV and its extensions, SegEarth-OV3 (SAM 3), GSNet (AAAI 2025), RSKT-Seg (AAAI 2026), SkySense-O (CVPR 2025), SCORE (ICCV 2025), AerOSeg.
None of them does TTA or tests corruption robustness, except TMPA and SeeCo.

## 3. Closed-set "CLIP encoder + trained decoder"

**No dedicated TTA method found**; CLIP backbones are treated as ordinary ViTs.

Robustness evidence (domain generalization, not TTA):
- VLTSeg (ACCV 2024): fine-tuned CLIP + Mask2Former generalizes better than supervised or self-supervised pretraining.
- Rein (CVPR 2024), frozen-backbone DG on Cityscapes: CLIP-L 53.7 vs **MAE-L 43.3**, DINOv2 63.3. These numbers come from a search snippet; check the paper's table before citing.
- REOBench: GeoRSCLIP is the most robust segmenter.

TTA evidence with CLIP (classification only):
- TENT gives small gains on zero-shot CLIP, or **hurts** it under corruptions (CLIPTTA: 40.3 → 35.1).
- CLIP is vulnerable to the loss of embedding uniformity (UnInfo, TMLR).

## 4. CLIP classification TTA → dense prediction

TPT, TDA, DiffTPT, C-TPT, WATT, CLIPArTT, ZERO, BoostAdapter, DPE and RLCF have **no segmentation extension from their authors**.
In MLMP and DAF, TPT, WATT and CLIPArTT were reimplemented as baselines: they stay flat or degrade, and some run out of memory.

## Implications for our study

- **Main track (closed set):** compare SatMAE / Scale-MAE vs RemoteCLIP / GeoRSCLIP + UperNet under the same TTA set:
  - none, bnadapt, Tent, SAR/EATA, CoTTA, ViDA, ISA
  - No prior work makes this comparison, so it is publishable as is.
  - Expect CLIP to start more robust, which leaves less TTA headroom. Entropy TTA may collapse where source accuracy is low.
  - Report the collapse rate per backbone and ECE / entropy-error correlation.
- **Fairness:** MIM-specific TTA (TTT-MAE, Continual-MAE) gives MIM an extra objective. Either leave it out or report it as a separate row.
- **Optional CLIP-aware variant (novel):** an anchor-consistency loss to the frozen CLIP features (as in DAF), or an auxiliary head built from frozen text embeddings, which would make MLMP-style multi-prompt entropy usable.
- **Secondary zero-shot track (optional, small):**
  - Setup: SegEarth-OV + {OpenAI CLIP, RemoteCLIP, GeoRSCLIP} (B/32 SimFeatUp weights exist) on LoveDA / Potsdam with REOBench corruptions.
  - TTA arms: TENT-LN, MLMP, TMPA.
  - MIM models cannot join this track, so keep it as an appendix or a secondary axis.
- **Classification:** CLIP-specific TTA (TPT, TDA, …) belongs only in the zero-shot track. The closed-set track uses the same TTA set for both backbone families.
