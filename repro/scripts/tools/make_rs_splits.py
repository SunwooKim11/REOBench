"""RESISC45 <-> UC Merced scale-shift splits with the 19 shared classes of EarthShift (Doerksen & Kerner 2026).

Class pairing = github.com/kerner-lab/earthshift configs/scale-shift-exp.json (`finetune_classes` / `test_classes`,
index-aligned, no merged classes; the paper says "highway", the config uses freeway on both sides). EarthShift loads the
torchgeo splits and keeps the 19 classes; labels = position in the alphabetically sorted list (same order here).
Check: RESISC45 filtered counts reproduce the paper's Table 1 exactly (train 7,915 / val 2,726 / ID test 2,659).
UC Merced filtered torchgeo splits are 1,142 / 379 / 379 (1,900 images in all); the paper's UCM numbers (420 / 420 /
1,260) do not match any filtered split, so the OOD test set is written both as the torchgeo test split and as all images.

  python3 scripts/tools/make_rs_splits.py
Output: /data/rs/resisc45/earthshift19/{train,val,test}.txt and /data/rs/ucmerced/earthshift19/{train,val,test,all}.txt,
lines "<class folder>/<file> <label>", plus classes.tsv (label, RESISC45 name, UC Merced name).
"""
import os
import re

R45 = ["airplane", "baseball_diamond", "beach", "chaparral", "dense_residential", "forest", "freeway", "golf_course",
       "harbor", "intersection", "medium_residential", "mobile_home_park", "overpass", "parking_lot", "river", "runway",
       "sparse_residential", "storage_tank", "tennis_court"]
UCM = ["airplane", "baseballdiamond", "beach", "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
       "harbor", "intersection", "mediumresidential", "mobilehomepark", "overpass", "parkinglot", "river", "runway",
       "sparseresidential", "storagetanks", "tenniscourt"]
assert R45 == sorted(R45) and UCM == sorted(UCM)  # alphabetical label order is the same on both sides

SETS = [("/data/rs/resisc45", "NWPU-RESISC45", "resisc45-{}.txt", R45),
        ("/data/rs/ucmerced", "UCMerced_LandUse/Images", "uc_merced-{}.txt", UCM)]
for root, img_dir, split_fmt, classes in SETS:
    out = f"{root}/earthshift19"
    os.makedirs(out, exist_ok=True)
    every = []
    for split in ["train", "val", "test"]:
        lines = []
        for f in (l.strip() for l in open(f"{root}/{split_fmt.format(split)}")):
            c = re.sub(r"_?\d+\.\w+$", "", f)
            if c in classes:
                assert os.path.exists(f"{root}/{img_dir}/{c}/{f}"), f
                lines.append(f"{c}/{f} {classes.index(c)}")
        open(f"{out}/{split}.txt", "w").write("\n".join(lines) + "\n")
        every += lines
        print(f"{out}/{split}.txt: {len(lines)}")
    if "ucmerced" in root:
        open(f"{out}/all.txt", "w").write("\n".join(sorted(every)) + "\n")
        print(f"{out}/all.txt: {len(every)}")
    with open(f"{out}/classes.tsv", "w") as fh:
        fh.write("label\tresisc45\tucmerced\n")
        for i, (a, b) in enumerate(zip(R45, UCM)):
            fh.write(f"{i}\t{a}\t{b}\n")
