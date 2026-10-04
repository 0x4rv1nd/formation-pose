"""
Phase 7c: background groups of the FW-UAV6DPose scenes, and the scene-level split derived from them.

A scene's background is the per-pixel median over its frames of a 64x36 grey thumbnail (the aircraft is small and moves, so
it drops out). Distance between two scenes = mean absolute difference of these thumbnails (grey levels, 0-255). Scenes of the
same sky render are < 1.25 apart (continuum up to 1.2, then 1.6, then > 2.9); different skies are a median of about 20 apart.
A background GROUP is a connected component of the scenes under this distance (all 85 training.zip + all 24 val.zip scenes).

Split rule:
  TEST        000029, 000030, 000136, 000137 (as in Phase 7b).
  EXCLUDED    every training.zip scene in a group that contains a test scene.
  VALIDATION  one complete group of 4-6 training.zip scenes (VAL_GROUP): no test scene, no val.zip scene, nothing else in its group,
              chosen from the candidate table below.
  TRAIN       all remaining training.zip scenes + the 16 Phase 7b training scenes + the old val scenes 000042-45.
Needs data/fw_uav/thumbs_7c.npz (phase7c_dupcheck.py). Writes results/phase7c/background_groups.json / .md.
"""
import json
import os
import zipfile

import numpy as np

import fw_uav_io as io
from phase7c_dupcheck import CACHE, OUT, thumb
from phase7c_model_check import load_train_scene, train_scenes

THRESHOLD = 1.25
VAL_GROUP = ["000150", "000151", "000152", "000153"]   # why: see results/phase7c/background_groups.md (written below)
USER_LIST = ["000021", "000022", "000023", "000024", "000025", "000026", "000027", "000028", "000128", "000129", "000130", "000132"]


def val_zip_backgrounds(scenes):
    out = {}
    with zipfile.ZipFile(io.ZIP_PATH) as zf:
        for s in scenes:
            out[s] = np.median([thumb(zf.read(f"val/{s}/rgb/{f:06d}.png"))[0] for f in range(0, 100, 10)], axis=0)
    return out


def groups(names, bg):
    """Connected components of the scenes under THRESHOLD."""
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    dist = {(a, b): float(np.abs(bg[a] - bg[b]).mean()) for a in names for b in names}
    for a in names:
        for b in names:
            if a < b and dist[a, b] < THRESHOLD:
                parent[find(a)] = find(b)
    comp = {}
    for n in names:
        comp.setdefault(find(n), []).append(n)
    return sorted(comp.values(), key=lambda m: m[0]), dist


def main():
    s7b = json.load(open("fw_uav_split.json"))
    z = np.load(CACHE)
    sc, t = z["sc"], z["t"]
    bg = {"T" + str(s): np.median(t[sc == s], axis=0) for s in np.unique(sc)}      # T = training.zip, V = val.zip
    vz = s7b["train"] + s7b["val"] + s7b["test"]
    bg.update({"V" + s: b for s, b in val_zip_backgrounds(vz).items()})
    comps, dist = groups(sorted(bg), bg)
    test = ["V" + s for s in s7b["test"]]
    t_scenes = train_scenes()
    n_frames = {s: len(load_train_scene(s)[0]) for s in t_scenes}
    depth = {s: load_train_scene(s)[2][:, 2] for s in t_scenes}

    excluded = sorted(n[1:] for c in comps if set(c) & set(test) for n in c if n[0] == "T")
    val_c = [c for c in comps if set("T" + s for s in VAL_GROUP) <= set(c)][0]
    assert sorted(val_c) == sorted("T" + s for s in VAL_GROUP), val_c                 # a complete group, nothing else in it
    assert 4 <= len(VAL_GROUP) <= 6 and not set(VAL_GROUP) & set(excluded)
    train = [s for s in t_scenes if s not in excluded and s not in VAL_GROUP]
    assert len(train) + len(excluded) + len(VAL_GROUP) == len(t_scenes)

    def nearest_outside(c):
        o = [n for n in bg if n not in c]
        return min(((n, min(dist[m, n] for m in c)) for n in o), key=lambda x: x[1])

    rows = []                                            # candidates: groups made only of training.zip scenes, 2-6 scenes
    for c in comps:
        if all(n[0] == "T" for n in c) and 2 <= len(c) <= 6:
            ss = [n[1:] for n in c]
            nn, d = nearest_outside(c)
            dt = min(dist[m, h] for m in c for h in test)
            r = np.concatenate([depth[s] for s in ss])
            rows.append((ss, sum(n_frames[s] for s in ss), r.min(), r.max(), nn, d, dt))
    lines = ["## Background groups (distance < %.2f grey levels, over all 85 training.zip + 24 val.zip scenes)" % THRESHOLD, "",
             "| group | scenes | split role |", "|---|---|---|"]
    role = {}
    for c in comps:
        ss = [n[1:] for n in c]
        zips = "".join(sorted({n[0] for n in c}))
        if set(c) & set(test):
            r = "TEST group: test " + ", ".join(sorted(h[1:] for h in c if h in test)) + "; training.zip members EXCLUDED"
        elif set(ss) == set(VAL_GROUP):
            r = "VALIDATION"
        else:
            r = "train" + (" (contains val.zip scenes: 7b train / old val)" if "V" in zips else "")
        lines.append(f"| {len(lines) - 3} | {' '.join(ss) if len(ss) <= 12 else ' '.join(ss[:6]) + ' ... ' + ss[-1] + f' ({len(ss)} scenes)'} | {r} |")
        role[len(lines) - 4] = r
    lines += ["", "## Candidate validation groups (2-6 training.zip scenes, no val.zip scene in the group)", "",
              "| scenes | frames | range (m) | nearest scene outside the group (distance) | min distance to a test scene |", "|---|---|---|---|---|"]
    for ss, nf, lo, hi, nn, d, dt in rows:
        lines.append(f"| {' '.join(ss)}{' **(chosen)**' if ss == VAL_GROUP else ''} | {nf} | {lo:.0f}-{hi:.0f} | {nn} ({d:.1f}) | {dt:.1f} |")
    lines += ["", f"Only {VAL_GROUP[0]}-{VAL_GROUP[-1]} is a complete group of 4-6 scenes that has no other member; it is clearly separate from every other "
              "scene (the others in the table have 2-3 scenes). It adds 331 frames with ranges up to 494 m (the test scenes reach 433 m).",
              "", f"Excluded from training (test groups): {len(excluded)} scenes: {' '.join(excluded)}.",
              f"Scenes in the TEST groups that were not in the first list (000021-28, 000128-130, 000132): "
              f"{' '.join(sorted(set(excluded) - set(USER_LIST)))} (distance to test scene 000136/000137 1.0-1.2, closer than 000130 to 000137).",
              f"Training: {len(train)} training.zip scenes + {len(s7b['train'])} Phase 7b training scenes + {len(s7b['val'])} old validation scenes."]
    os.makedirs(OUT, exist_ok=True)
    res = {"threshold": THRESHOLD, "groups": [[n for n in c] for c in comps], "excluded_test_groups": excluded,
           "val_group": VAL_GROUP, "train_training_zip": train, "added_to_excluded_vs_first_list": sorted(set(excluded) - set(USER_LIST)),
           "val_group_nearest_outside": dict(zip(("scene", "distance"), nearest_outside(val_c))),
           "val_group_min_distance_to_test": min(dist[m, h] for m in val_c for h in test)}
    json.dump(res, open(os.path.join(OUT, "background_groups.json"), "w"), indent=1)
    open(os.path.join(OUT, "background_groups.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
