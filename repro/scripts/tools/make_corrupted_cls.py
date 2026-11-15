"""Corrupted copy of a scene-classification test set with the REOBench corruptions (e.g. RESISC45-C), AID-C layout.

Corruption functions are executed verbatim from ~/REOBench/Corrupt Image Generation/ (photometric: the "without
annotations" script; rotate / translate: the "with annotations" script, called with no boxes). Differences from the
released code, each fitted against AID-C (2026-11-16, 8-10 AID test images per severity, after re-saving as JPEG):
  - motion_blur kernel length 10/15/20/25/30 (released 2..10; MAE 1.5-2.0 vs 7.4-8.4 with the released values)
  - compression_artifacts JPEG quality 50/40/30/20/10 (released 30..10; MAE 0.7-1.8)
  - data_gaps stripe width 5..9 (released 3..7; stripe count 2..6 matches; best-IoU fit of the stripe mask)
  - scale_image: resize by 0.9..0.5 then reflect-pad to the original size (BORDER_REFLECT, MAE 4.2-6.4), not the
    released black canvas (AID-C has no black pixels)
  - matching the released values (MAE 1-3): gaussian_blur, haze, brightness_contrast, rotate (reflect), translate
    (reflect, best of the 4 random directions), gaussian_noise sigma (std after JPEG).
  - clouds NOT matched: AID-C cloud coverage is lower than the released densities give (estimated 0.013/0.026/0.048/
    0.082/0.129 vs about 0.04-0.22); the cause was not found, so the released values are kept.
  - every output is saved as .jpg with PIL defaults (quality 75), as AID-C is. This is why AID-C gaussian noise std is
    0.028 / 0.056 for the released sigma 0.04 / 0.08 (reproduced: 0.029 / 0.059).
  - seeded: every (image, corruption, severity) gets its own seed (crc32), so the set is reproducible. The released code
    is unseeded, so random corruptions (noise, clouds, gaps, translate direction) differ pixel-wise from AID-C.
  - runs on CPU (clouds gets device=cpu); the released fused-noise branch is not used.
Names follow AID-C: gaussian_noise_new, haze_update_new, rotate_new, translate_image_new, data_gaps, ...

  ~/REOBench/.venv-corrupt/bin/python scripts/tools/make_corrupted_cls.py \
      --root /data/rs/resisc45/NWPU-RESISC45 --list /data/rs/resisc45/resisc45-test.txt \
      --out /data/rs/resisc45/RESISC45-C --workers 64
Output: <out>/<Class>/<corruption>/severity_<s>/<class index>/<file stem>.jpg, class index = sorted class folders.
"""
import argparse
import os
import random
import types
import zlib
from multiprocessing import Pool

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

GEN = os.path.expanduser("~/REOBench/Corrupt Image Generation")


def load(fn, stop):
    """Execute a REOBench script up to (not including) the line starting with `stop`; return its namespace."""
    src = open(os.path.join(GEN, fn)).read()
    ns = {"__name__": "reobench_corrupt"}
    exec(compile(src[:src.index(stop)], fn, "exec"), ns)
    return ns


P = load("image_generation_without_annotations.py", "parser = argparse")
G = load("image_feneration_with_annotations.py", "# Process image and annotations")
_random_noise = P["random_noise"]


def with_levels(fn, old, new):
    """Copy of a released function with one severity table replaced (lists of constants are tuple constants)."""
    co = fn.__code__
    assert old in co.co_consts, (fn.__name__, old)
    return types.FunctionType(co.replace(co_consts=tuple(new if c == old else c for c in co.co_consts)),
                              fn.__globals__, fn.__name__, fn.__defaults__)


motion_blur = with_levels(P["apply_motion_blur_batch"], (2, 4, 6, 8, 10), (10, 15, 20, 25, 30))
compression = with_levels(P["add_compression_artifacts_batch"], (30, 25, 20, 15, 10), (50, 40, 30, 20, 10))
data_gaps = with_levels(P["create_data_gaps_batch"], (3, 4, 5, 6, 7), (5, 6, 7, 8, 9))


def scale_reflect(img, severity=1):  # = AID-C scale_image (see docstring)
    s = [0.9, 0.8, 0.7, 0.6, 0.5][severity - 1]
    x = np.array(img)
    h, w = x.shape[:2]
    nh, nw = round(h * s), round(w * s)
    y = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_LINEAR)
    t, l = (h - nh) // 2, (w - nw) // 2
    return Image.fromarray(cv2.copyMakeBorder(y, t, h - nh - t, l, w - nw - l, cv2.BORDER_REFLECT))


CORRS = {  # AID-C name -> f(PIL image, severity) -> PIL image
    "gaussian_noise_new": lambda im, s: P["add_gaussian_noise_batch"]([im], s)[0],
    "salt_and_pepper_noise": lambda im, s: P["add_salt_and_pepper_noise_batch"]([im], s)[0],
    "gaussian_blur": lambda im, s: P["apply_gaussian_blur_batch"]([im], s)[0],
    "motion_blur": lambda im, s: motion_blur([im], s)[0],
    "brightness_contrast": lambda im, s: P["adjust_brightness_contrast_batch"]([im], s)[0],
    "clouds": lambda im, s: P["add_clouds_batch"]([im], s, device="cpu")[0],
    "haze_update_new": lambda im, s: P["add_haze_batch"]([im], s)[0],
    "data_gaps": lambda im, s: data_gaps([im], s)[0],
    "compression_artifacts": lambda im, s: compression([im], s)[0],
    "rotate_new": lambda im, s: G["rotate_image_and_annotations"](im, [], s)[0],
    "scale_image": scale_reflect,
    "translate_image_new": lambda im, s: G["translate_image_and_annotations"](im, [], s)[0],
}


def work(job):
    path, cls_name, cls_idx, out = job
    stem = os.path.splitext(os.path.basename(path))[0]
    img = Image.open(path).convert("RGB")
    n = 0
    for c, f in CORRS.items():
        for s in range(1, 6):
            dst = f"{out}/{cls_name}/{c}/severity_{s}/{cls_idx}/{stem}.jpg"
            if os.path.exists(dst):
                continue
            seed = zlib.crc32(f"{stem}|{c}|{s}".encode())
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            P["random_noise"] = lambda *a, **k: _random_noise(*a, rng=seed, **k)  # salt & pepper
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            f(img.copy(), s).convert("RGB").save(dst)
            n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with one sub-folder per class")
    ap.add_argument("--list", default=None, help="file names to corrupt (torchgeo split file); default = all images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(1)
    classes = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    files = {f: c for c in classes for f in os.listdir(os.path.join(args.root, c))}
    names = [l.strip() for l in open(args.list) if l.strip()] if args.list else sorted(files)
    missing = [f for f in names if f not in files]
    assert not missing, f"{len(missing)} listed files not found, e.g. {missing[:3]}"
    jobs = [(os.path.join(args.root, files[f], f), files[f], classes.index(files[f]), args.out) for f in names]
    print(f"{len(jobs)} images x {len(CORRS)} corruptions x 5 severities -> {args.out}", flush=True)
    done = 0
    with Pool(args.workers) as pool:
        for i, n in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            done += n
            if i % 500 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] images, {done} files written", flush=True)
