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

## Detail (checked against the arXiv pages / HTML, 2026-10-22)

| Short name | Full title | Authors / venue |
|---|---|---|
| MLMP | Test-Time Adaptation of Vision-Language Models for Open-Vocabulary Semantic Segmentation | Noori, Osowiechi, …, Ben Ayed, Desrosiers; arXiv 2505.21844 (venue not on the page; listed as NeurIPS 2025 by the GitHub page) |
| DAF | Towards Continual Test-Time Adaptation of Vision-Language Models in Open-Vocabulary Semantic Segmentation | Doloriel et al.; arXiv 2608.29923, under review |
| Seg-TTO | Test-Time Optimization for Domain Adaptive Open Vocabulary Segmentation | De Silva, …, Naseer, Rodrigo; arXiv 2501.04696 |
| TPA | Test-Time Prototype Adaptation for Open-Vocabulary Semantic Segmentation | Wang, Cheng, Li, Tang; arXiv 2608.08290 |
| FLOSS | FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation | Benigmim, Fahes, Vu, Bursuc, de Charette; ICCV 2025 |
| SeeCo | Seeking Consensus: Geometric-Semantic On-the-Fly Recalibration for Open-Vocabulary Remote Sensing Semantic Segmentation | Wang et al.; arXiv 2604.26221 |
| BAPA | Balanced Prompt Adaptation against Entropy-Induced Collapse for Test-Time Binary Segmentation | Wang et al.; arXiv 2609.21743 |

### MLMP
- **Updated parameters:** only the vision-encoder LayerNorm (~0.02%); the text encoder is frozen.
- **Host:** NACLIP ViT-L/14.
- **Loss:**
  - Image-level: CLS entropy (ILE).
  - Pixel-level: uncertainty-weighted entropy of fused intermediate-layer features (UAML). Layers L7–L24 work best; layer weights are softmax(−β·entropy) with β = 0 during adaptation and β = 1 at inference.
  - Averaged over 7 templates.
- **Optimization:** Adam, lr 1e-3, 10 steps, batches of 2 images, reset after every batch (episodic).
- **Results:** V20 75.91 → 83.76; V20-C 69.01 → 77.58. The benchmark has 87 scenarios (15 corruptions at severity 5, plus ACDC and GTA-V).

### DAF (Diversify, Anchor, Filter)
- **Setting:** continual, no reset. CLIP image-encoder LN only. Host NACLIP ViT-B/32. Batch 8, SGD lr 1e-3, 1 step.
- **Loss:** L = entropy (TENT/MLMP) + 2.0·marginal diversity + 0.5·cross-modal anchor consistency.
  - Marginal diversity: entropy of the batch-level class marginal; resists patch-level class collapse.
  - Cross-modal anchor consistency: assigns each patch to its nearest text prototype under the frozen source model, then applies hinge penalties for drifting away from the assigned prototype or toward the others.
- **Filter (SAFS):** samples whose adapted-vs-source token similarity barely changes (shift score Δ ≤ μ − 0.5σ) skip the backward pass.
- **LoveDA results (mIoU):**

  | | Source | TENT | SAR | MLMP | DAF-M |
  |---|---|---|---|---|---|
  | LoveDA mean | 22.7 | 20.3 | 22.1 | 16.7 | 31.8 |
  | LoveDA-C | 9.2 | 1.2 | 7.6 | 5.2 | 14.9 |

  (LoveDA-C: 15 corruptions, ImageNet-C protocol, severity 5.)

### Seg-TTO
- **Text side (gradients, per image):** 5 shared "general" embeddings + 1 per-category embedding; per-category visual attributes from an LLM (Llama 3 70B), weighted by similarity.
- **Visual side (no gradients):** 64 augmented views; the 20% with the lowest entropy are added back into the original feature map at their crop locations and normalized.
- **Loss and optimization:** pixel entropy + pseudo-label CE merged with PCGrad; AdamW lr 5e-3, 2 + 3 steps.
- **Hosts:** CAT-Seg-B/L and CLIP-DINOiser.
- **Benchmark:** MESS, 22 datasets; the Earth Monitoring group is iSAID, **Potsdam**, WorldFloods, FloodNet and UAVid.
- **Results:** +1.6 to +2.6 mIoU on average. Inference takes 1.5 s per image.

### TPA
- **Mechanism:** training-free, output level, no gradients.
  1. From a small pool of unlabeled deployment-domain images (~10%), pick confident anchor patches using the host's own predictions.
  2. Average their frozen DINO features into per-class prototypes.
  3. At inference, the cosine similarity to the prototypes is fused linearly with the host logits.
- **Evaluation:** 5 hosts, 3 CLIP backbones, 8 benchmarks.

### FLOSS
- **Mechanism:** instead of averaging prompt templates, pick for each class the single template whose predictions have the lowest entropy ("class expert") and fuse the experts.
- **Cost:** no labels, no training. Plug-and-play. Code: github.com/yasserben/FLOSS.

### SeeCo
- **Mechanism:** for an RS training-free OVSS host, recalibrates each scene during inference:
  - visual side: geometric consensus across views (GCL)
  - text side: description-based semantic consensus (SCL)
  - both are injected through an online consensus injector (OCI)
- **Cost:** no training.
- **Evaluation:** 8 RS benchmarks. Details are not in the abstract.

### BAPA
- **Problem:** in imbalanced binary segmentation, pixel entropy lets the larger class dominate and collapses the mask (shown in theory).
- **Method:**
  - Class-Balanced Anchors: high-confidence anchors per class with equal total loss weight.
  - Dynamic Prompt Adaptation: only the text-side prompt residuals are updated, with anchors refreshed after each step.
- **Unclear:** the VLM is not named.
