"""
Phase 7c, step 1: duplicate-frame check between training.zip and our val / test scenes.

Every frame of training.zip and every 10th frame of the val / test scenes of val.zip is reduced to a 64x36 grey thumbnail
(and a 16x16 difference hash). For every sampled frame the nearest training frame is found: mean absolute grey difference
(0-255) and Hamming distance of the 256-bit hash. Context distances: adjacent frames inside one scene (a near-duplicate
looks like this) and the nearest frame of a different scene. Writes results/phase7c/duplicate_check.json.
"""
import json
import os
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

import fw_uav_io as io

TRAIN_ZIP = os.path.join(io.DATA_DIR, "training.zip")
CACHE = os.path.join(io.DATA_DIR, "thumbs_7c.npz")
OUT = os.path.join("results", "phase7c")


def thumb(raw):
    g = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    t = cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    h = cv2.resize(g, (17, 16), interpolation=cv2.INTER_AREA)
    return t, (h[:, 1:] > h[:, :-1]).ravel()


def job(args):
    path, prefix, scene, step = args
    out = []
    with zipfile.ZipFile(path) as zf:
        names = sorted(n for n in zf.namelist() if n.startswith(f"{prefix}/{scene}/rgb/") and n.endswith(".png"))
        for n in names[::step]:
            t, h = thumb(zf.read(n))
            out.append((scene, int(os.path.basename(n)[:-4]), t, h))
    return out


def collect(path, prefix, scenes, step):
    with Pool(6) as p:
        res = p.map(job, [(path, prefix, s, step) for s in scenes])
    flat = [r for rs in res for r in rs]
    return (np.array([r[0] for r in flat]), np.array([r[1] for r in flat]), np.stack([r[2] for r in flat]),
            np.stack([r[3] for r in flat]))


def main():
    split = json.load(open("fw_uav_split.json"))
    with zipfile.ZipFile(TRAIN_ZIP) as zf:
        train_scenes = sorted({n.split("/")[1] for n in zf.namelist() if n.count("/") >= 2})
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        tr = (z["sc"], z["fr"], z["t"], z["h"])
    else:
        tr = collect(TRAIN_ZIP, "training", train_scenes, 1)
        np.savez_compressed(CACHE, sc=tr[0], fr=tr[1], t=tr[2], h=tr[3])
    print("training frames hashed:", len(tr[0]))
    res = {}
    for part in ("test", "val"):
        sc, fr, t, h = collect(io.ZIP_PATH, "val", split[part], 10)
        mad = np.abs(t[:, None] - tr[2][None]).mean(axis=(2, 3))             # (n, n_train)
        ham = (h[:, None] != tr[3][None]).sum(axis=2)
        i, j = mad.argmin(1), ham.argmin(1)
        rows = [{"scene": s, "frame": int(f), "min_mad": float(mad[k, i[k]]), "nearest_mad": f"{tr[0][i[k]]}_{tr[1][i[k]]:06d}",
                 "min_hamming": int(ham[k, j[k]])} for k, (s, f) in enumerate(zip(sc, fr))]
        res[part] = rows
        print(f"{part}: {len(rows)} sampled frames; min MAD to training {mad.min(1).min():.2f}..{mad.min(1).max():.2f}, "
              f"min Hamming {ham.min(1).min()}..{ham.min(1).max()} (of 256)")
        for s in split[part]:
            r = [x for x in rows if x["scene"] == s]
            print(f"   {s}: min MAD {min(x['min_mad'] for x in r):.2f}, min Hamming {min(x['min_hamming'] for x in r)}")
    # context: adjacent frames inside a training scene, and nearest frame of another scene
    adj, other = [], []
    rng = np.random.default_rng(0)
    for k in rng.choice(len(tr[0]), 300, replace=False):
        same = tr[0] == tr[0][k]
        nb = np.nonzero(same & (np.abs(tr[1].astype(int) - tr[1][k]) == 1))[0]
        if len(nb):
            adj.append(float(np.abs(tr[2][nb[0]] - tr[2][k]).mean()))
        d = np.abs(tr[2][~same] - tr[2][k]).mean(axis=(1, 2))
        other.append(float(d.min()))
    ctx = {"adjacent_frame_mad_median": float(np.median(adj)), "adjacent_frame_mad_max": float(np.max(adj)),
           "other_scene_nearest_mad_min": float(np.min(other)), "other_scene_nearest_mad_median": float(np.median(other))}
    print("context:", ctx)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"context": ctx, **res}, open(os.path.join(OUT, "duplicate_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
