#!/bin/bash
# Datasets for the cross-dataset / natural-shift plan that are not yet under /data/rs.
# Already present: AID-C, Potsdam-C (reobench/), Potsdam + Vaihingen (seg/isprs/), LoveDA (loveda_uda/, seg/loveda/),
#                  m-EuroSAT (geobench/).
#   resisc45  : torchgeo HF mirror of NWPU-RESISC45 (31,500 images, 45 classes, 256px) + torchgeo train/val/test lists
#   ucmerced  : torchgeo HF mirror of UC Merced Land Use (2,100 images, 21 classes, 256px) + split lists
#   deepglobe : DeepGlobe Land Cover, the 803 labelled train tiles (2448px) as parquet (unofficial HF upload,
#               ratnaonline1; the official labelled set is also 803 tiles). Optional: only for DeepGlobe -> DFC2022.
# DFC2022 is not included: the only HF copy (snchen1230/DFC2022, 60 GB) has no README and an unknown chunk format;
# use the official IEEE DataPort release (login required) if needed.
#   ~/.local/tmux-env/bin/tmux new -s dl 'bash ~/REOBench/repro/scripts/tools/download_extra_rs.sh 2>&1 | tee /data/rs/_download_extra.log'
set -e
PY="$HOME/REOBench/.venv/bin/python -I"
get() {  # repo, target dir
  echo "[$(date '+%F %T')] $1 -> $2"
  $PY -c "from huggingface_hub import snapshot_download as s; s('$1', repo_type='dataset', local_dir='$2')"
}

get torchgeo/resisc45 /data/rs/resisc45
(cd /data/rs/resisc45 && unzip -q -n NWPU-RESISC45.zip)

get torchgeo/ucmerced /data/rs/ucmerced
(cd /data/rs/ucmerced && unzip -q -n UCMerced_LandUse.zip)

get ratnaonline1/deepglobe-land-cover-classification-dataset /data/rs/seg/deepglobe_landcover

echo "[$(date '+%F %T')] DONE"
du -sh /data/rs/resisc45 /data/rs/ucmerced /data/rs/seg/deepglobe_landcover
