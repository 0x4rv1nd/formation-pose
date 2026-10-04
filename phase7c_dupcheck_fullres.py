"""
Phase 7c, step 1 (second part): the thumbnails of phase7c_dupcheck.py are dominated by the sky, so the nearest training frame
of a val / test frame is often a frame of another scene with the same sky. Here every sampled val / test frame is compared at
full resolution with its 5 nearest training frames (by thumbnail): the number of pixels that differ by more than 12 grey levels
(a duplicate has about 0; a different aircraft position with the same sky has the aircraft's pixels), and the ground-truth pose
difference (rotation angle, translation distance) to the nearest of those. Writes results/phase7c/duplicate_check_fullres.json.
"""
import json
import os
import zipfile

import cv2
import numpy as np

import fw_uav_io as io
from phase7c_dupcheck import TRAIN_ZIP, CACHE, OUT
from phase7c_model_check import load_train_scene


def grey(zf, name):
    return cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.int16)


def rot_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main():
    split = json.load(open("fw_uav_split.json"))
    z = np.load(CACHE)
    sc, fr, th = z["sc"], z["fr"], z["t"]
    poses = {s: load_train_scene(s) for s in np.unique(sc)}
    rows = []
    with zipfile.ZipFile(TRAIN_ZIP) as ztr, zipfile.ZipFile(io.ZIP_PATH) as zva:
        for part in ("test", "val"):
            for scene in split[part]:
                ids, R, t, K = io.load_scene(scene)
                for frame in ids[::10]:
                    a = grey(zva, f"val/{scene}/rgb/{int(frame):06d}.png")
                    ta = cv2.resize(a.astype(np.uint8), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
                    cand = np.argsort(np.abs(th - ta).mean(axis=(1, 2)))[:5]
                    best = None
                    for c in cand:
                        b = grey(ztr, f"training/{sc[c]}/rgb/{int(fr[c]):06d}.png")
                        n = int((np.abs(a - b) > 12).sum())
                        if best is None or n < best[0]:
                            best = (n, f"{sc[c]}_{int(fr[c]):06d}")
                    i = list(ids).index(frame)
                    dpose = min((rot_deg(R[i], Rt[j]), float(np.linalg.norm(t[i] - tt[j])))
                                for _, Rt, tt, _ in poses.values() for j in range(len(Rt)))
                    rows.append({"part": part, "scene": scene, "frame": int(frame), "changed_px": best[0], "nearest": best[1]})
                    # pose distance in the same units as step 4 is reported there; here only an exact-duplicate test
                    rows[-1]["min_rot_deg"], rows[-1]["trans_m_at_min_rot"] = dpose
    for part in ("test", "val"):
        r = [x for x in rows if x["part"] == part]
        ch = [x["changed_px"] for x in r]
        print(f"{part}: changed pixels vs nearest-sky training frame: min {min(ch)}, median {int(np.median(ch))}, max {max(ch)}")
        print(f"   smallest pose rotation difference to ANY training frame: {min(x['min_rot_deg'] for x in r):.2f} deg")
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "duplicate_check_fullres.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
