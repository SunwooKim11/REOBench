#!/bin/bash
# Unpack the released REOBench Potsdam zips (zip-of-zips) into plain folders mmseg can read.
#   /data/rs/reobench/Potsdam_extracted/
#     images/train, images/<cond>/...   (cond = clear_test, haze, ...; inner layout kept as released)
#     anns/train,   anns/<cond>/...     (clear_test, create_data_gaps, rotate, scale_image, translate_image)
set -e
SRC=/data/rs/reobench/Potsdam
DST=/data/rs/reobench/Potsdam_extracted
mkdir -p $DST/images $DST/anns $DST/_stage
echo "[$(date '+%F %T')] train images/anns"
unzip -q -o $SRC/Potsdam_Images_train.zip -d $DST/images
unzip -q -o $SRC/Potsdam_Anns_train.zip -d $DST/anns
for kind in Anns Images; do
  echo "[$(date '+%F %T')] test $kind (outer)"
  unzip -q -o $SRC/Potsdam_${kind}_test.zip -d $DST/_stage
  for z in $(find $DST/_stage -name '*.zip'); do
    name=$(basename $z .zip); name=${name%_img}; name=${name%_ann}
    out=$DST/$( [ $kind = Anns ] && echo anns || echo images )/$name
    echo "[$(date '+%F %T')]   $name -> $out"
    mkdir -p $out && unzip -q -o $z -d $out && rm $z
  done
done
rmdir $DST/_stage/* 2>/dev/null || true; rmdir $DST/_stage 2>/dev/null || true
echo "[$(date '+%F %T')] DONE"; du -sh $DST/images/* $DST/anns/*
