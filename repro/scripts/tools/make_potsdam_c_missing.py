"""The four Potsdam-C corruptions whose images are missing from the REOBench release: rotate, scale_image, translate_image,
data_gaps (the release has the labels of all four but only 9 image sets; see docs/SETTINGS.md §2.5).

What the release tells us (checked 2026-11-24):
  - the released Potsdam-C images of motion blur / JPEG / gaussian blur equal the output of the released generation code
    exactly (mean absolute error 0.00), so the released code and its severity tables are used here unchanged
    (data_gaps: 2..6 stripes of width 3..7; not the AID-C-fitted values of make_corrupted_cls.py)
  - released labels: rotate = 30/45/60/75/90 degrees, reflect border; scale = 0.9..0.5 with reflect padding (not the black
    canvas of the released function); translate = 15..35 px, reflect border, direction random per image
  - the released rotate / scale labels were interpolated linearly (values outside the class ids), so they cannot be used
Images: released functions (bilinear), scale with reflect padding as in the released labels.
Labels (rotate, scale_image, translate_image): the same geometry on the clean label with nearest-neighbour interpolation
  and reflect border. translate uses the direction of the released label of the same image (found by matching the four
  candidates; must match exactly, otherwise the script stops). data_gaps keeps the clean label (occlusion does not move it).
Seeded per (image, corruption, severity) with crc32, as make_corrupted_cls.py.

  ~/REOBench/.venv-corrupt/bin/python scripts/tools/make_potsdam_c_missing.py --workers 3      # from repro/, CPU
Output: /data/rs/reobench/Potsdam_extracted/test/<corruption>/severity_<s>/<name>.png  (images)
        /data/rs/reobench/Potsdam_extracted/test/<corruption>_ann/severity_<s>/<name>.png  (labels, geometric ones only)
"""
import argparse
import os
import random
import sys
import zlib
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_corrupted_cls as M  # noqa: E402  (executes the released generation scripts, CPU only)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

T = "/data/rs/reobench/Potsdam_extracted/test"
A = "/data/rs/reobench/Potsdam_extracted/anns"
ANGLE, SCALE, SHIFT = [30, 45, 60, 75, 90], [0.9, 0.8, 0.7, 0.6, 0.5], [15, 20, 25, 30, 35]


def rotate(x, s, interp):
    h, w = x.shape[:2]
    return cv2.warpAffine(x, cv2.getRotationMatrix2D((w / 2, h / 2), ANGLE[s - 1], 1.0), (w, h), flags=interp, borderMode=cv2.BORDER_REFLECT)


def scale(x, s, interp):
    h, w = x.shape[:2]
    y = cv2.resize(x, (int(w * SCALE[s - 1]), int(h * SCALE[s - 1])), interpolation=interp)
    nh, nw = y.shape[:2]
    t, l = (h - nh) // 2, (w - nw) // 2
    return cv2.copyMakeBorder(y, t, h - nh - t, l, w - nw - l, cv2.BORDER_REFLECT)


def translate(x, dx, dy, interp):
    h, w = x.shape[:2]
    return cv2.warpAffine(x, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=interp, borderMode=cv2.BORDER_REFLECT)


def work(name):
    img = np.array(Image.open(f"{T}/clean/{name}").convert("RGB"))
    lab = np.array(Image.open(f"{T}/clean_ann/{name}").convert("L"))
    n = 0
    for s in range(1, 6):
        out = {}
        out["rotate"] = (rotate(img, s, cv2.INTER_LINEAR), rotate(lab, s, cv2.INTER_NEAREST))
        out["scale_image"] = (scale(img, s, cv2.INTER_LINEAR), scale(lab, s, cv2.INTER_NEAREST))
        rel = np.array(Image.open(f"{A}/translate_image/translate_image/severity_{s}/ann/{name}").convert("L"))
        sh = SHIFT[s - 1]
        cand = [(dx, dy) for dx in (-sh, sh) for dy in (-sh, sh) if np.array_equal(translate(lab, dx, dy, cv2.INTER_NEAREST), rel)]
        assert cand, f"no translate direction reproduces the released label: {name} severity {s}"
        dx, dy = cand[0]
        out["translate_image"] = (translate(img, dx, dy, cv2.INTER_LINEAR), rel)
        seed = zlib.crc32(f"{name}|data_gaps|{s}".encode())
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        out["data_gaps"] = (np.array(M.P["create_data_gaps_batch"]([Image.fromarray(img)], s)[0].convert("RGB")), None)
        for c, (x, y) in out.items():
            for arr, d in ((x, f"{T}/{c}/severity_{s}"), (y, f"{T}/{c}_ann/severity_{s}")):
                if arr is None:
                    continue
                os.makedirs(d, exist_ok=True)
                Image.fromarray(arr).save(f"{d}/{name}")
                n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(1)
    names = sorted(os.listdir(f"{T}/clean"))[: args.limit]
    done = 0
    with Pool(args.workers) as pool:
        for i, n in enumerate(pool.imap_unordered(work, names, chunksize=8), 1):
            done += n
            if i % 200 == 0 or i == len(names):
                print(f"[{i}/{len(names)}] images, {done} files written", flush=True)
