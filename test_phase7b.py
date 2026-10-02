"""Sanity checks for Phase 7b part A (prints PASS/FAIL). Needs `python phase7b_prepare.py` to have been run."""

import json
import os
import sys
import zipfile

import numpy as np

import fw_uav_config as cfg
import fw_uav_io as io
import phase7b_prepare as prep

results = []


def check(name, ok):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


split = prep.load_split()
train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
check(f"split by scene: {len(train)} train / {len(val)} val / {len(test)} test, disjoint, covering all {len(io.scenes())} scenes",
      (len(train), len(val), len(test)) == (16, 4, 4) and not (train & val or train & test or val & test)
      and train | val | test == set(io.scenes()))
check("scene 000125 (sequence cut) is not in the test split", "000125" not in test)
groups = split["background_groups"]
group_split = {g: {k for k, s in (("train", train), ("val", val), ("test", test)) if set(v) & s}
               for g, v in groups.items() if g != "note"}
check("no background group is shared between splits", all(len(v) == 1 for v in group_split.values()))
ranges = np.concatenate([np.linalg.norm(io.load_scene(s)[2], axis=1) for s in test])
check(f"test scenes cover near and far distances ({ranges.min():.0f}-{ranges.max():.0f} m)", ranges.min() < 200 and ranges.max() > 400)

flip = prep.flip_idx_from_names(cfg.KEYPOINT_NAMES)
mismatch = prep.check_flip_idx(flip)
check(f"flip_idx {flip} is an involution that pairs mirror-image keypoints (max mirror mismatch {mismatch.max():.2f} m)",
      flip == [0, 1, 2, 4, 3, 9, 10, 11, 12, 5, 6, 7, 8])
yaml_text = open(os.path.join(prep.YOLO_DIR, "data.yaml")).read()
check("data.yaml: kpt_shape [13, 3], the same flip_idx, relative paths (no absolute 'path')",
      "kpt_shape: [13, 3]" in yaml_text and f"flip_idx: {flip}" in yaml_text and "train: images/train" in yaml_text
      and "val: images/val" in yaml_text and "path:" not in yaml_text)

cache = io.build_keypoint_cache(cfg.KEYPOINTS)
ok_format = ok_values = ok_box = ok_gt = True
n_img = {}
for part, scenes in (("train", split["train"]), ("val", split["val"])):
    imgs = sorted(os.listdir(os.path.join(prep.YOLO_DIR, "images", part)))
    labs = sorted(os.listdir(os.path.join(prep.YOLO_DIR, "labels", part)))
    n_img[part] = len(imgs)
    ok_format &= [i[:-4] for i in imgs] == [l[:-4] for l in labs] and all(i.endswith(".jpg") for i in imgs)
    for name in imgs:
        scene, frame = name[:-4].split("_")
        assert scene in scenes
        box, kp = prep.read_label(os.path.join(prep.YOLO_DIR, "labels", part, name[:-4] + ".txt"))
        raw = open(os.path.join(prep.YOLO_DIR, "labels", part, name[:-4] + ".txt")).read().split()
        ok_format &= len(raw) == 5 + 3 * 13 and raw[0] == "0"
        ok_values &= bool(((np.r_[box, kp[:, :2].ravel()] >= 0) & (np.r_[box, kp[:, :2].ravel()] <= 1)).all()
                          and set(kp[:, 2]) <= {0.0, 2.0} and np.all(kp[kp[:, 2] == 0, :2] == 0))
        vis = kp[:, 2] > 0
        x0, x1, y0, y1 = box[0] - box[2] / 2, box[0] + box[2] / 2, box[1] - box[3] / 2, box[1] + box[3] / 2
        ok_box &= bool(np.all((kp[vis, 0] >= x0 - 1e-6) & (kp[vis, 0] <= x1 + 1e-6) & (kp[vis, 1] >= y0 - 1e-6) & (kp[vis, 1] <= y1 + 1e-6)))
        c = cache[scene]
        i = int(np.flatnonzero(c["frames"] == int(frame))[0])
        ok_gt &= bool(np.array_equal(vis, c["vis_dilated"][i]) and np.allclose(kp[vis, 0] * cfg.IMAGE_W, c["uv"][i][vis, 0], atol=1e-3)
                      and np.allclose(kp[vis, 1] * cfg.IMAGE_H, c["uv"][i][vis, 1], atol=1e-3))
check(f"{n_img['train']} train + {n_img['val']} val images, one label file each (class 0, 13 x (x, y, v))",
      ok_format and (n_img["train"], n_img["val"]) == (1600, 400))
check("label values: everything in [0, 1], v in {0, 2}, hidden keypoints at (0, 0)", ok_values)
check("every visible keypoint lies inside its box", ok_box)
check("keypoints and visibility equal the ground-truth projection with the 2 px dilated mask", ok_gt)

with zipfile.ZipFile(prep.ZIP_OUT) as zf:
    names = zf.namelist()
size_mb = os.path.getsize(prep.ZIP_OUT) / 1e6
check(f"zip ({size_mb:.0f} MB): data.yaml + 2000 images + 2000 labels, no test-scene file",
      "yolo_fw_uav/data.yaml" in names and sum(n.endswith(".jpg") for n in names) == 2000 and sum(n.endswith(".txt") for n in names) == 2000
      and not any(os.path.basename(n).split("_")[0] in test for n in names))

nb = json.load(open(os.path.join("notebooks", "train_yolo_pose_colab.ipynb")))
src = "\n".join("".join(c["source"]) for c in nb["cells"])
check("notebook: GPU check, Drive checkpoints and resume, imgsz=1280, epochs=100, patience=20, pose validation",
      all(s in src for s in ("cuda.is_available", "resume=True", "imgsz=1280", "epochs=100", "patience=20", "drive.mount", "pose.map50")))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
