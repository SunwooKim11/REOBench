# ImageNet-C reference numbers (severity 5) used by `summarize.py`

All per-corruption numbers were read from 300-dpi page renders of the PDFs in `papers/TTA/`.
Order: gauss, shot, impulse, defocus, glass, motion, zoom, snow, frost, fog, bright, contrast, elastic, pixel, jpeg | avg.

## B. DeYO Table 1 — mild, ResNet-50-BN (torchvision v1), reset per corruption, bs 64, accuracy
Setting `deyo_mild_r50`: `--arch r50 --norm imagenet --prep crop224 --data full --stream reset`
| method | numbers | avg |
|---|---|---|
| none | 2.2 2.9 1.8 17.9 9.8 14.8 22.5 16.9 23.3 24.4 58.9 5.4 16.9 20.7 31.7 | 18.0 |
| tent | 29.2 31.2 30.1 28.1 27.7 41.4 49.4 47.2 41.5 57.7 67.4 29.2 54.8 58.5 52.4 | 43.1 |
| eata | 34.9 37.1 35.8 33.4 33.0 47.1 52.7 51.6 45.7 60.0 68.1 44.4 57.9 60.6 55.1 | 47.8 |
| sar | 30.6 30.6 31.3 28.5 28.5 41.9 49.4 47.1 42.2 57.5 67.3 37.8 54.6 58.4 52.1 | 43.9 |
| deyo | 35.6 37.9 37.1 33.8 34.1 48.5 52.8 52.7 46.4 60.6 68.0 46.1 58.4 61.5 55.7 | 48.6 |

## C. ViT-B LN (timm augreg_in21k_ft_in1k), online label shift ir=inf, reset per corruption, bs 64, accuracy
Setting `ls_vitb`: `--arch vitb_augreg --norm imagenet --prep crop224 --data full --stream label_shift` (seed 2024 order)
DeYO Table 5 (DeYO's reruns) / SAR Table 2 (SAR's runs, seed 2021 order):
| method | DeYO T5 | SAR T2 |
|---|---|---|
| none | 9.4 6.7 8.3 29.1 23.4 34.0 27.1 15.8 26.4 47.4 54.7 44.0 30.5 44.5 47.6 → 29.9 | 29.9 |
| tent | 33.9 1.8 27.2 54.8 52.9 58.6 54.3 12.4 11.7 69.7 76.3 66.3 59.6 69.7 66.6 → 47.7 | 47.3 |
| eata | 36.2 34.7 35.5 43.4 44.3 49.3 48.5 53.2 53.5 62.3 72.7 18.8 58.0 64.7 62.8 → 49.2 | 49.9 |
| sar | 42.3 34.9 44.1 50.0 50.5 55.6 53.1 59.7 47.2 66.2 75.2 50.3 60.1 67.3 65.0 → 54.8 | 58.0 |
| deyo | 53.5 36.0 54.6 57.6 58.7 63.7 46.2 67.6 66.0 73.2 77.9 66.7 69.0 73.5 70.3 → 62.3 | – |
Collapse-prone columns (shot, snow, frost, contrast) have large stds (DeYO shot ±25); compare the average.

## D. Continual (CoTTA / ViDA protocol): robustbench 5,000 ids, Res256Crop224, no reset, standard order, error
ViDA Table 1 (R50 rows = CoTTA repo default order):
| setting / method | numbers (error) | mean |
|---|---|---|
| `cont_r50` none | 97.8 97.1 98.2 81.7 89.8 85.2 78.0 83.5 77.1 75.9 41.3 94.5 82.5 79.3 68.6 | 82.0 |
| `cont_r50` cotta | 84.7 82.1 80.6 81.3 79.0 68.6 57.5 60.3 60.5 48.3 36.6 66.1 47.2 41.2 46.0 | 62.7 |
| `cont_vitb` none | 53.0 51.8 52.1 68.5 78.8 58.5 63.3 49.9 54.2 57.7 26.4 91.4 57.5 38.0 36.2 | 55.8 |
| `cont_vitb` vida | 47.7 42.5 42.9 52.2 56.9 45.5 48.9 38.9 42.7 40.7 24.3 52.8 49.1 33.5 33.1 | 43.4 |
`cont_r50`: `--arch r50 --norm imagenet --prep res256crop224 --data rb5000 --stream continual` (CoTTA SGD 0.01, ap 0.1)
`cont_vitb`: `--arch vitb_augreg --norm none --prep res256crop224 --data rb5000 --stream continual --bs 50`
(ViDA repo feeds raw [0,1] to the augreg ViT; ViDA repo log: 43.37). CoTTA paper itself: 63.0 ± 1.8 over 10 orders.

## E. EATA-C Table A — single-domain (reset per corruption), ViT-B "from timm" (source 39.84), bs 64, accuracy
Checkpoint/normalization not stated; `src_*` runs in setting `eatac_vitb` identify it (target none avg 39.84,
gauss 12.9). Then:
| method | numbers | avg |
|---|---|---|
| none | 12.9 17.6 11.7 34.4 27.7 43.7 36.2 43.4 45.4 52.8 73.3 45.5 37.9 54.7 60.2 | 39.84 |
| tent | 33.4 42.1 41.4 48.8 45.2 54.9 48.2 55.6 55.1 64.4 75.2 62.5 51.6 65.5 65.0 | 53.92 |
| cotta | 45.6 46.8 28.1 45.3 42.0 57.6 49.8 59.5 57.6 65.2 74.9 58.2 55.5 66.3 65.7 | 54.54 |
| sar | 43.1 44.0 44.3 50.4 48.2 55.1 49.0 56.9 56.0 64.7 75.3 62.4 53.5 65.3 64.9 | 55.53 |
| eata | 50.5 51.3 51.3 55.9 56.0 60.7 58.0 64.2 62.2 70.1 77.4 66.7 63.9 70.5 68.9 | 61.84 |
EATA-C settings: Tent/EATA SGD lr 0.001 bs 64; CoTTA ViT lr 0.005, p_th 0.1. This is the setting closest to our
REOBench mild protocol (reset per stream, ViT-LN, uniform labels).

## Not verifiable on ImageNet-C
- Tent paper: only a bar chart (all severities, R50 pycls) — Tent is checked through DeYO/SAR/EATA-C/ViDA tables.
- LSCD-TTA: RS datasets only. ISA: our own baseline.
