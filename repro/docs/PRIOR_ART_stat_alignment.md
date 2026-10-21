# Prior art: statistics-alignment TTA (survey by agent, 2026-10-22)

About 40 web searches; abstracts and pages read for the key items. Items marked [uncertain] were not fully verified.
This survey is not exhaustive.

## A. Input-level / pixel-level alignment

| Paper | Venue | How close it is to ISA |
|---|---|---|
| Karani et al., test-time adaptable networks (normalization CNN in front of a frozen segmenter) | MedIA 2021 | Learned, nonlinear input normalization; ISA is its affine, learning-free special case |
| Mummadi et al., confidence maximization + input transformation (channel-wise affine input) | arXiv 2021 | ISA with learned parameters, trained with an entropy-type loss |
| Thopalli et al., CATTAn | ACML 2022 | Remarks informally that input transforms suit pixel-level corruptions |
| Chen et al., VPTTA (low-frequency input prompt trained to match BN statistics) | CVPR 2024 | Already couples the input level and feature statistics; no analysis by corruption type |
| Niu et al., FOA (input prompt + back-to-source activation shifting, ViT, no backprop) | ICML 2024 | Aligns at two depths for ViTs; no analysis of which mechanism needs which depth |
| Gao et al., DDA (diffusion input adaptation) | CVPR 2023 | The heavy counterpart; main "input vs model adaptation" reference |
| Park et al., test-time style shifting (AdaIN on early-layer statistics) | ICML 2023 | ISA moved one layer deeper |
| Test-time Fourier style calibration; FDA (Yang & Soatto) | arXiv 2022 / CVPR 2020 | Per-channel mean/std matching = the DC term plus a global amplitude scale of this family |
| StandardGAN (CVPRW 2020, EarthVision); DAugNet (arXiv 2020) | | RS input standardization (histogram matching, z-score) against cross-city shift is established practice |
| Relative radiometric normalization (e.g. LIRRN, Sensors 2024) | | Pixel moment / histogram matching is textbook RS preprocessing |

## B. Feature / normalization statistics, and which layer to align

- **BN adaptation** (= our bnadapt):
  - Schneider et al. NeurIPS 2020
  - Nado et al. arXiv 2020
  - Benz et al. WACV 2021
  - DUA CVPR 2022
- **TTN** (ICLR 2023): mixes test and source statistics per layer, weighted by shift sensitivity. The weights are static, and the authors list shift-dependent mixing as future work.
- **Segmentation baselines:** DIGA (CVPR 2023, backward-free); Unmix-TNS (ICLR 2024, robust to non-i.i.d. batches).
- **Surgical fine-tuning** (Lee et al., ICLR 2023): input-level corruptions favour adapting the early layers. This is the closest published form of "align at the layer matching the mechanism", but it is supervised fine-tuning.
- **Automatic layer selection:**
  - LAW (WACV 2024, Fisher layer weighting)
  - Domain-specific block selection (CVPR 2024)
- **ViT statistics recalibration:** AQR (arXiv 2025, recalibrating the top half was best); QuAR (arXiv 2026).
- **Feature-distribution alignment:** CAFe, CAFA (ICCV 2023), TTAC (NeurIPS 2022), FAU. CAFA's only failure on TinyImageNet-C is contrast, an anecdote that a radiometric shift needs input-level alignment.

## C. Analyses: entropy TTA fails, or the gain comes from statistics

- **TTAB** (ICML 2023): hyperparameter selection is hard, gains depend on the base model, and no method handles all shift classes.
- **Yu et al.** (arXiv 2023): benchmark of 13 methods.
- **Yi et al.** (ACM MM 2024): in segmentation, BN-statistics updates help only slightly or hurt, and teacher-student methods do not beat the source.
  - This **conflicts with our bnadapt +8 to +10 on noise/JPEG**, so the difference must be explained.
  - Likely reasons: synthetic sensor degradation vs natural shift, and a decoder whose only BN sits on a frozen LN backbone.
- **LAME** (CVPR 2022), **RDumb** (NeurIPS 2023): parameter updates are brittle; periodic reset matches the state of the art. Reviewers will ask for a continual run.
- **SAR, DeYO:** already argue that entropy is unreliable.
- arXiv 2608.22233 [uncertain]: on CIFAR-10-C, Tent's gradient step on top of BN re-estimation sometimes hurts.
- No controlled study shows, for RS or ViT foundation models, that "Tent's gain is almost all from BN statistics".

## D. Corruption-type-aware / shift diagnosis

- **Diagnosis that only selects BN statistics:**
  - PAN (arXiv 2024): corruption ID → per-corruption BN statistics
  - Rezaei & Norouzzadeh (arXiv 2023): Fourier corruption-type detection → BN statistics lookup
  - FROST (arXiv 2024): FFT prototypes
  - None of them chooses between input-level and feature-level alignment.
- **Routing by something else:**
  - Hybrid-TTA (ICCV 2025): routes by shift magnitude, not type or layer
  - DPCore (ICML 2025), MoASE, frequency-aware experts (arXiv 2507.00502), TiME (ICML 2026): expert routing
- **TTT-MAE** (NeurIPS 2022): frozen MAE ViT-L plus reconstruction TTA. Same setup as ours, so it is an important baseline.

## E. Remote sensing

- **REOBench** (NeurIPS 2025 D&B): robustness benchmark only; no TTA evaluation seen [moderately confident].
- **Entropy, prompt or teacher based TTA:**
  - LSCD-TTA (entropy family)
  - USP (Remote Sensing 2024, self-prompt segmentation TTA)
  - HyperTTA (arXiv 2025, hyperspectral)
  - Open-vocabulary RS segmentation TTA (CVPR 2026)
- **CTAOD-RS-like TCSVT 2026** [uncertain, full text not seen]: continual detection TTA with frequency/spatial "spectral routing" and reconstruction. The closest RS analogue of idea (ii); **read it first**.
- **Other:** AID-NN (natural noise robustness, no TTA); EarthShift (arXiv 2026, real EO shifts; testbed candidate); LiDAR point-cloud TTA.

## Verdict

1. **ISA on its own is not novel.** It equals first-two-moment histogram matching, Reinhard colour transfer, z-score standardization, AdaIN/AdaBN at layer 0, and the DC term of FDA. Present it as a strong, under-reported baseline.
2. **"Align at the layer matching the mechanism" is partly published** (surgical fine-tuning, TTN, style shifting, CATTAn, VPTTA/FOA; PAN-style diagnosis selects only BN statistics). **Not found anywhere:** the complementarity "input moment alignment fixes affine radiometric corruptions (brightness, contrast, haze), while BN/feature alignment fixes non-affine, high-frequency degradations (noise, S&P, JPEG)", and "learning-free alignment beats entropy TTA on RS foundation models".
   - Sharper claim: "channel-affine corruptions are invertible in pixel space; non-affine, local degradations are not and need renormalization deeper in the network".
   - This is a refinement of Lee et al. (by their logic noise and JPEG are also input-level, yet feature-level alignment is what fixes them here), and the tension is a selling point.
3. **Venue:**
   - **RS venue (TGRS, ISPRS, EarthVision):** the clearest gap is the analysis plus a lightweight corruption-family router. Every RS TTA paper found is entropy, prompt or teacher based.
   - **General ML venue:** weak as it stands. It would need ImageNet-C or non-RS evidence, several architectures, and theory (affine invertibility) plus a depth sweep.

## Experiments reviewers will demand

- **Baselines:**
  - Input side: histogram matching, Reinhard/AdaIN, FDA
  - Statistics side: TTN, DIGA, Unmix-TNS, PAN-style per-corruption statistics, ViT activation/LN recalibration (FOA, QuAR/AQR)
  - Others: VPTTA, TTT-MAE, DDA (if compute allows), an oracle per-corruption best-method upper bound
- **Depth sweep:** align at the input, at each backbone block (LN statistics) and at the decoder BN, per corruption. Back it with the per-channel affine fit R² of radiometric corruptions and the spectral or higher-order change of noise/JPEG.
- **Batch regime:** batch size down to 1, class-imbalanced or correlated batches, mixed corruptions within a batch, a continual stream without reset.
- **Real shifts:** Potsdam→Vaihingen, LoveDA urban↔rural, cross-sensor or seasonal (e.g. EarthShift).
- **Breadth:** more RS FMs (including CNN/BN backbones and non-MAE models), more datasets (NWPU, UCM, iSAID/DIOR), multispectral inputs.
- **Classification arm:** a frozen ViT with a linear head has no BN, so define clearly what "bnadapt" means for AID.
- **Why entropy fails:** calibration of the linear probe, and the correlation between entropy and accuracy per corruption.
- **Router:** detector accuracy, unseen corruption types, failure cases, overhead, seeds/variance.
- **The Yi et al. contradiction** (their BN gains are near zero in segmentation) must be explained.
