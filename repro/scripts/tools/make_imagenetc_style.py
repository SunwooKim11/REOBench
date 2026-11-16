"""ImageNet-C corruptions with ImageNet-C's own parameters, applied to RS test sets (question (b) in
docs/reading/imagenet_c.md §6: does the *same* corruption act differently on natural and RS images?).

Functions: `imagecorruptions` 1.1.2 (bethgelab), the same severity tables as hendrycks/robustness make_imagenet_c.py;
motion blur is its numpy port of the Wand blur (same radius / sigma / random angle in [-45, 45]).
Corruptions = the ImageNet-C types that have a REOBench counterpart:
  gaussian_noise, impulse_noise (= salt & pepper), gaussian_blur (ImageNet-C extra set), motion_blur, jpeg_compression,
  brightness, contrast
Inputs follow ImageNet-C's geometry where possible:
  aid, r45  : test split resized (whole image) to 224 x 224 first, as ImageNet-C corrupts 224 crops; saved as JPEG q85
              (ImageNet-C's save setting); the clean reference is the same 224 image saved the same way
  potsdam   : the 2,016 REOBench test patches at their native 512 x 512 (the segmentation models' input); PNG, labels
              unchanged (all seven corruptions are photometric or blur)
Seeded per (image, corruption, severity). ImageNet side: the stored ImageNet-C (/data/tta/ImageNet_Family/ImageNet_C).

  ~/REOBench/.venv-corrupt/bin/python scripts/tools/make_imagenetc_style.py --sets aid r45 potsdam --workers 64
Output: /data/rs/imagenetc_style/<set>/{clean,<corruption>/<severity>}/<relative path>
"""
import argparse
import os
import random
import zlib
from multiprocessing import Pool

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402
import imagecorruptions.corruptions as _ic  # noqa: E402
from imagecorruptions import corrupt  # noqa: E402
from PIL import Image  # noqa: E402

_gaussian = _ic.gaussian  # skimage >= 0.19 renamed multichannel=True to channel_axis=-1 (imagecorruptions 1.1.2 predates it)
_ic.gaussian = lambda *a, multichannel=False, **k: _gaussian(*a, channel_axis=-1 if multichannel else None, **k)

CORRS = ["gaussian_noise", "impulse_noise", "gaussian_blur", "motion_blur", "jpeg_compression", "brightness", "contrast"]
OUT = "/data/rs/imagenetc_style"


def items(name):
    if name == "aid":
        root = "/data/rs/reobench/AID/AID_clean_renamed/test"
        return root, [f"{c}/{f}" for c in sorted(os.listdir(root)) for f in sorted(os.listdir(f"{root}/{c}"))]
    if name == "r45":
        root = "/data/rs/resisc45/NWPU-RESISC45"
        names = [l.strip() for l in open("/data/rs/resisc45/resisc45-test.txt") if l.strip()]
        return root, [f"{f.rsplit('_', 1)[0]}/{f}" for f in names]
    root = "/data/rs/reobench/Potsdam_extracted/test/clean"
    return root, sorted(os.listdir(root))


def work(job):
    name, root, rel = job
    img = Image.open(f"{root}/{rel}").convert("RGB")
    if name != "potsdam":
        img = img.resize((224, 224), Image.BILINEAR)
        rel = os.path.splitext(rel)[0] + ".jpg"
    save = (lambda im, p: im.save(p, quality=85, optimize=True)) if name != "potsdam" else (lambda im, p: im.save(p))
    n = 0
    dst = f"{OUT}/{name}/clean/{rel}"
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        save(img, dst); n += 1
    x = np.asarray(img)
    for c in CORRS:
        for s in range(1, 6):
            dst = f"{OUT}/{name}/{c}/{s}/{rel}"
            if os.path.exists(dst):
                continue
            seed = zlib.crc32(f"{rel}|{c}|{s}".encode())
            np.random.seed(seed); random.seed(seed)
            y = corrupt(x, corruption_name=c, severity=s)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            save(Image.fromarray(np.uint8(np.clip(y, 0, 255))), dst); n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["aid", "r45", "potsdam"])
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    for name in args.sets:
        root, rels = items(name)
        jobs = [(name, root, r) for r in rels]
        print(f"{name}: {len(jobs)} images x {len(CORRS)} corruptions x 5 + clean", flush=True)
        done = 0
        with Pool(args.workers) as pool:
            for i, n in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
                done += n
                if i % 1000 == 0 or i == len(jobs):
                    print(f"  [{i}/{len(jobs)}] {done} files", flush=True)
