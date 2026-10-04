"""
Phase 7c, part A: YOLO-pose dataset from the full FW-UAV6DPose training split (the only change from Phase 7b is the training data).

Split by background group (phase7c_background_groups.py): train = training.zip scenes outside the test groups and the validation
group + the 16 Phase 7b training scenes + the old val scenes 000042-45, EVERY 2ND FRAME of every scene (consecutive frames are
nearly identical). Validation: one complete background group of training.zip (000150-53), all frames. The 4 test scenes are never read.
Labels, keypoints, visibility, flip_idx, box margin and JPEG quality are those of phase7b_prepare.py (its functions are reused).

    python phase7c_prepare.py --stage split      # fw_uav_split_7c.json and the pose-coverage table (no images)
    python phase7c_prepare.py --stage build      # data/fw_uav/yolo_fw_uav_7c/ and yolo_fw_uav_7c.zip, label-check figures
"""
import argparse
import json
import os
import shutil
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

import fw_uav_config as cfg
import fw_uav_io as io
import phase7b_prepare as p7b
from phase7c_model_check import TRAIN_ZIP, load_train_scene, read_train_mask, train_scenes

SPLIT_OUT = "fw_uav_split_7c.json"
YOLO_DIR = os.path.join(io.DATA_DIR, "yolo_fw_uav_7c")
ZIP_OUT = os.path.join(io.DATA_DIR, "yolo_fw_uav_7c.zip")
OUT_DIR = os.path.join("results", "phase7c")
STEP = 2


def make_split():
    s7b = json.load(open("fw_uav_split.json"))
    bg = json.load(open(os.path.join(OUT_DIR, "background_groups.json")))
    new = bg["train_training_zip"]
    assert not set(new) & (set(bg["excluded_test_groups"]) | set(bg["val_group"]))
    assert len(new) + len(bg["excluded_test_groups"]) + len(bg["val_group"]) == len(train_scenes())
    split = {
        "description": "Phase 7c scene-level split by background group (phase7c_background_groups.py). Training = training.zip scenes outside the "
                       "test groups and the validation group + the 16 Phase 7b training scenes + the old validation scenes 000042-45 of val.zip, "
                       f"every {STEP}nd frame (frames 0, 2, 4, ...). Validation = one complete background group of training.zip, all frames. "
                       "Test = the 4 Phase 7b test scenes, all frames, never used for training or tuning.",
        "frame_step_train": STEP,
        "train_training_zip": new,
        "train_val_zip": s7b["train"] + s7b["val"],
        "val_training_zip": bg["val_group"],
        "test": s7b["test"],
        "excluded_training_zip": bg["excluded_test_groups"],
        "rules": "Background group = connected component of scenes whose median background thumbnails differ by < 1.25 grey levels. "
                 "Excluded: training.zip scenes in a group with a test scene. Validation: one complete group of 4 training.zip scenes with no "
                 "test or val.zip scene. Everything else is training.",
    }
    json.dump(split, open(SPLIT_OUT, "w"), indent=1)
    return split


def rot_deg(Ra, Rb):
    """Geodesic angle (deg) between every rotation in Ra (n,3,3) and every one in Rb (m,3,3): (n, m)."""
    tr = np.einsum("nij,mij->nm", Ra, Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def poses(scene):
    return load_train_scene(scene) if scene in set(train_scenes()) else io.load_scene(scene)


def coverage(split):
    """Median (and share above 30 deg) of the rotation angle from every TEST frame to the nearest training frame."""
    s7b = json.load(open("fw_uav_split.json"))

    def stack(scenes, step):
        return np.concatenate([poses(s)[1][::step] for s in scenes])
    sets = {"7b (16 scenes, all frames)": stack(s7b["train"], 1),
            "7c (65+20 scenes, every 2nd)": stack(split["train_training_zip"] + split["train_val_zip"], STEP)}
    out = {"metric": "rotation angle (deg) of the ground-truth model-to-camera rotation to the nearest training frame",
           "n_train_frames": {k: len(v) for k, v in sets.items()}, "scenes": {}}
    for scene in split["test"]:
        Rq = io.load_scene(scene)[1]
        out["scenes"][scene] = {}
        for k, Rt in sets.items():
            d = rot_deg(Rq, Rt).min(axis=1)
            out["scenes"][scene][k] = {"median": float(np.median(d)), "p90": float(np.percentile(d, 90)), "share_above_30deg": float((d > 30).mean())}
    json.dump(out, open(os.path.join(OUT_DIR, "pose_coverage.json"), "w"), indent=1)
    names = list(sets)
    lines = ["| test scene | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for scene, r in out["scenes"].items():
        lines.append(f"| {scene} | " + " | ".join(f"{r[k]['median']:.1f} ({100 * r[k]['share_above_30deg']:.0f} %)" for k in names) + " |")
    lines.append("")
    lines.append("Median rotation angle in degrees to the nearest training frame (share of the scene's frames farther than 30 deg). "
                 "Training frames: " + ", ".join(f"{k}: {v}" for k, v in out["n_train_frames"].items()))
    open(os.path.join(OUT_DIR, "pose_coverage.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def scene_job(args):
    """Write the JPEGs and labels of one scene. kind: 'training' (training.zip) or 'val' (val.zip)."""
    scene, kind, part, step = args
    zpath = TRAIN_ZIP if kind == "training" else io.ZIP_PATH
    ids, R, t, K = load_train_scene(scene) if kind == "training" else io.load_scene(scene)
    kernel = np.ones((2 * io.MASK_DILATE_PX + 1,) * 2, np.uint8)
    n_vis = []
    with zipfile.ZipFile(zpath) as zf:
        for i in range(0, len(ids), step):
            frame = int(ids[i])
            name = f"{scene}_{frame:06d}"
            img = cv2.imdecode(np.frombuffer(zf.read(f"{kind}/{scene}/rgb/{frame:06d}.png"), np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(YOLO_DIR, "images", part, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, p7b.JPEG_QUALITY])
            mask = read_train_mask(zf, scene, frame) if kind == "training" else io.read_mask(scene, frame, zf)
            uv, depth = io.project_points(cfg.KEYPOINTS, R[i], t[i], K)
            mask_d = cv2.dilate(mask.astype(np.uint8), kernel) > 0
            u, v = np.round(uv).astype(int).T
            inside = (u >= 0) & (u < io.IMAGE_SIZE[0]) & (v >= 0) & (v < io.IMAGE_SIZE[1]) & (depth > 0)
            vis = np.zeros(len(uv), bool)
            vis[inside] = mask_d[v[inside], u[inside]]
            with open(os.path.join(YOLO_DIR, "labels", part, name + ".txt"), "w") as f:
                f.write(p7b.yolo_label(mask, uv, vis) + "\n")
            n_vis.append(int(vis.sum()))
    return scene, part, n_vis


def build(split):
    if os.path.exists(YOLO_DIR):
        shutil.rmtree(YOLO_DIR)
    for part in ("train", "val"):
        os.makedirs(os.path.join(YOLO_DIR, "images", part))
        os.makedirs(os.path.join(YOLO_DIR, "labels", part))
    jobs = [(s, "training", "train", STEP) for s in split["train_training_zip"]] + \
           [(s, "val", "train", STEP) for s in split["train_val_zip"]] + [(s, "training", "val", 1) for s in split["val_training_zip"]]
    stats = {"train": [], "val": []}
    with Pool(6) as pool:
        for scene, part, n_vis in pool.imap_unordered(scene_job, jobs):
            stats[part] += n_vis
            print(f"  {part}: scene {scene} done ({len(n_vis)} frames)", flush=True)
    flip_idx = p7b.flip_idx_from_names(cfg.KEYPOINT_NAMES)
    p7b.check_flip_idx(flip_idx)
    p7b.write_yaml(os.path.join(YOLO_DIR, "data.yaml"), flip_idx)
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(YOLO_DIR):
            for fn in sorted(files):
                full = os.path.join(root, fn)
                zf.write(full, os.path.join("yolo_fw_uav_7c", os.path.relpath(full, YOLO_DIR)))
    size = os.path.getsize(ZIP_OUT) / 1e6
    summary = {k: {"images": len(v), "mean_visible_keypoints": float(np.mean(v))} for k, v in stats.items()}
    summary.update(zip_mb=round(size, 1), flip_idx=flip_idx, jpeg_quality=p7b.JPEG_QUALITY, box_margin_px=p7b.BOX_MARGIN_PX)
    json.dump(summary, open(os.path.join(OUT_DIR, "dataset_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return flip_idx


def check_same_as_7b(split):
    """The labels of the val.zip training scenes (even frames) must equal the Phase 7b label files (7b train and val parts)."""
    n = bad = 0
    for fn in os.listdir(os.path.join(YOLO_DIR, "labels", "train")):
        if fn.split("_")[0] in split["train_val_zip"]:
            old = [p for p in (os.path.join(p7b.YOLO_DIR, "labels", part, fn) for part in ("train", "val")) if os.path.exists(p)]
            assert len(old) == 1, fn
            n += 1
            bad += open(old[0]).read() != open(os.path.join(YOLO_DIR, "labels", "train", fn)).read()
    print(f"labels identical to Phase 7b for {n - bad} of {n} files")
    assert bad == 0


def label_checks(split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT_DIR, exist_ok=True)
    new = split["train_training_zip"]
    pick = [("train", new[k], f) for k, f in zip([0, 6, 12, 18, 24, 30, 36, 42], [20, 40, 60, 10, 30, 50, 70, 80])]
    pick += [("val", s, f) for s, f in zip(split["val_training_zip"], [30, 50, 70, 20])]
    pick = [(part, s, f if f % 2 == 0 or part == "val" else f + 1) for part, s, f in pick]
    for k in range(3):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (part, scene, frame) in zip(axes.ravel(), pick[4 * k:4 * k + 4]):
            name = f"{scene}_{frame:06d}"
            ip, lp = (os.path.join(YOLO_DIR, d, part, name + e) for d, e in (("images", ".jpg"), ("labels", ".txt")))
            if not os.path.exists(ip):                                  # a scene shorter than the picked frame
                ip = sorted(f for f in os.listdir(os.path.dirname(ip)) if f.startswith(scene))[-1]
                name = ip[:-4]
                ip, lp = (os.path.join(YOLO_DIR, d, part, name + e) for d, e in (("images", ".jpg"), ("labels", ".txt")))
            p7b.draw(ax, ip, lp)
            n = int((p7b.read_label(lp)[1][:, 2] > 0).sum())
            ax.set_title(f"{part} {name}  ({n}/13 visible)", fontsize=10)
        fig.suptitle("Phase 7c label check: box (yellow) and keypoint indices; blue = left, red = right, green = centreline", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"label_check_{k + 1}.png"), dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["split", "build"], required=True)
    stage = ap.parse_args().stage
    os.makedirs(OUT_DIR, exist_ok=True)
    if stage == "split":
        split = make_split()
        print(f"train: {len(split['train_training_zip'])} training.zip + {len(split['train_val_zip'])} val.zip scenes; excluded {len(split['excluded_training_zip'])}")
        coverage(split)
    else:
        split = json.load(open(SPLIT_OUT))
        build(split)
        check_same_as_7b(split)
        label_checks(split)


if __name__ == "__main__":
    main()
