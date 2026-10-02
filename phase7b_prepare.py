"""
Phase 7b, part A: prepare a YOLO-pose dataset from the FW-UAV6DPose train / val scenes (fw_uav_split.json).

For every frame of a train or val scene (test scenes are never read):
    image   RGB frame, saved as JPEG (quality 95)
    label   class 0 ("uav"), box from the visible-object mask (+ margin), the 13 keypoints of fw_uav_config.py
            projected with the ground-truth pose (v = 2 if inside the image and inside the mask dilated by 2 px,
            otherwise v = 0 with x = y = 0)
Writes data/fw_uav/yolo_fw_uav/ (images/{train,val}, labels/{train,val}, data.yaml with relative paths), zips it to
data/fw_uav/yolo_fw_uav.zip and draws results/phase7b/label_check_*.png and label_check_flip.png.

    python phase7b_prepare.py
"""

import json
import os
import shutil
import zipfile

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fw_uav_config as cfg
import fw_uav_io as io

SPLIT_PATH = "fw_uav_split.json"
YOLO_DIR = os.path.join(io.DATA_DIR, "yolo_fw_uav")
ZIP_OUT = os.path.join(io.DATA_DIR, "yolo_fw_uav.zip")
OUT_DIR = os.path.join("results", "phase7b")
JPEG_QUALITY = 95
BOX_MARGIN_PX = 4          # box = mask bounding box grown by this many pixels on every side


def load_split():
    with open(SPLIT_PATH) as f:
        return json.load(f)


def flip_idx_from_names(names):
    """Index of the mirror partner of every keypoint (left <-> right, centreline keypoints map to themselves)."""
    idx = []
    for n in names:
        partner = n.replace("left", "right") if "left" in n else n.replace("right", "left") if "right" in n else n
        idx.append(names.index(partner))
    return idx


def check_flip_idx(flip_idx):
    """
    flip_idx must be an involution, and mirroring the keypoints about the aircraft's centreline plane (y = nose y)
    must map every keypoint closest to its partner (the aircraft is left-right symmetric, so a horizontally flipped
    image shows the same aircraft with left and right swapped). Returns the mirror mismatch of every keypoint in metres:
    the wing-root keypoints were picked at |y| = 4.2 m rather than as exact mirror images, so they are 0.4-0.7 m off.
    """
    kp = cfg.KEYPOINTS
    assert sorted(flip_idx) == list(range(len(kp))) and all(flip_idx[flip_idx[i]] == i for i in range(len(kp))), \
        "flip_idx is not an involution"
    y0 = kp[cfg.KEYPOINT_NAMES.index("nose"), 1]
    mirrored = kp.copy()
    mirrored[:, 1] = 2 * y0 - kp[:, 1]
    dist = np.linalg.norm(mirrored[:, None, :] - kp[None, :, :], axis=2)       # mirrored i to every keypoint j
    assert (dist.argmin(axis=1) == np.array(flip_idx)).all(), "flip_idx pairs keypoints that are not mirror images"
    return dist[np.arange(len(kp)), flip_idx]


def yolo_label(mask, uv, vis):
    """YOLO-pose label line: class, normalised box (cx, cy, w, h), then (x, y, v) per keypoint."""
    ys, xs = np.nonzero(mask)
    x0, x1 = max(xs.min() - BOX_MARGIN_PX, 0), min(xs.max() + 1 + BOX_MARGIN_PX, cfg.IMAGE_W)
    y0, y1 = max(ys.min() - BOX_MARGIN_PX, 0), min(ys.max() + 1 + BOX_MARGIN_PX, cfg.IMAGE_H)
    vals = [0, (x0 + x1) / 2 / cfg.IMAGE_W, (y0 + y1) / 2 / cfg.IMAGE_H, (x1 - x0) / cfg.IMAGE_W, (y1 - y0) / cfg.IMAGE_H]
    for (u, v), visible in zip(uv, vis):
        if visible:
            vals += [min(max(u / cfg.IMAGE_W, 0.0), 1.0), min(max(v / cfg.IMAGE_H, 0.0), 1.0), 2]
        else:
            vals += [0.0, 0.0, 0]
    return format_label(vals)


def format_label(vals):
    out = [str(int(vals[0]))] + [f"{x:.6f}" for x in vals[1:5]]
    for i in range(5, len(vals), 3):
        out += [f"{vals[i]:.6f}", f"{vals[i + 1]:.6f}", str(int(vals[i + 2]))]
    return " ".join(out)


def write_yaml(path, flip_idx):
    lines = ["# YOLO-pose dataset: FW-UAV6DPose train / val scenes (Phase 7b). Paths are relative to this file.",
             "train: images/train", "val: images/val", "",
             "kpt_shape: [13, 3]   # 13 keypoints, (x, y, visibility)",
             f"flip_idx: {flip_idx}   # horizontal flip swaps left and right keypoints",
             "names:", "  0: uav", "", "# keypoint order: " + ", ".join(cfg.KEYPOINT_NAMES)]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def build(split):
    cache = io.build_keypoint_cache(cfg.KEYPOINTS)
    if os.path.exists(YOLO_DIR):
        shutil.rmtree(YOLO_DIR)
    stats = {}
    with zipfile.ZipFile(io.ZIP_PATH) as zf:
        for part in ("train", "val"):
            os.makedirs(os.path.join(YOLO_DIR, "images", part))
            os.makedirs(os.path.join(YOLO_DIR, "labels", part))
            n_vis = []
            for scene in split[part]:
                c = cache[scene]
                for i, frame in enumerate(c["frames"]):
                    name = f"{scene}_{int(frame):06d}"
                    img = cv2.imdecode(np.frombuffer(zf.read(f"val/{scene}/rgb/{int(frame):06d}.png"), np.uint8), cv2.IMREAD_COLOR)
                    cv2.imwrite(os.path.join(YOLO_DIR, "images", part, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    mask = io.read_mask(scene, int(frame), zf)
                    with open(os.path.join(YOLO_DIR, "labels", part, name + ".txt"), "w") as f:
                        f.write(yolo_label(mask, c["uv"][i], c["vis_dilated"][i]) + "\n")
                    n_vis.append(int(c["vis_dilated"][i].sum()))
                print(f"  {part}: scene {scene} done")
            stats[part] = {"scenes": len(split[part]), "frames": len(n_vis), "mean_visible_keypoints": float(np.mean(n_vis))}
    return stats


def make_zip():
    if os.path.exists(ZIP_OUT):
        os.remove(ZIP_OUT)
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_STORED) as zf:      # JPEGs do not compress further
        for root, _, files in os.walk(YOLO_DIR):
            for fn in sorted(files):
                full = os.path.join(root, fn)
                zf.write(full, os.path.join("yolo_fw_uav", os.path.relpath(full, YOLO_DIR)))
    return os.path.getsize(ZIP_OUT) / 1e6


def read_label(path):
    v = np.array(open(path).read().split(), dtype=float)
    return v[1:5], v[5:].reshape(-1, 3)


def draw(ax, img_path, label_path, flip_idx=None, flip=False, pad=60):
    """Crop around the labelled box and draw the box and keypoints (blue = left, red = right, green = centreline)."""
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    box, kp = read_label(label_path)
    if flip:
        img = img[:, ::-1]
        box = box.copy()
        box[0] = 1 - box[0]
        kp = kp[flip_idx].copy()
        kp[kp[:, 2] > 0, 0] = 1 - kp[kp[:, 2] > 0, 0]
    W, H = cfg.IMAGE_W, cfg.IMAGE_H
    cx, cy, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    half = max(w, h) / 2 + pad
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half, W))
    y0, y1 = int(max(cy - half, 0)), int(min(cy + half, H))
    ax.imshow(img[y0:y1, x0:x1], extent=(x0, x1, y1, y0))
    ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec="yellow", lw=1))
    for i, (x, y, v) in enumerate(kp):
        if v > 0:
            name = cfg.KEYPOINT_NAMES[i]
            col = "dodgerblue" if "left" in name else "red" if "right" in name else "lime"
            ax.plot(x * W, y * H, "o", ms=5, mfc="none", mec=col, mew=1.3)
            ax.annotate(str(i), (x * W, y * H), xytext=(3, 3), textcoords="offset points", color=col, fontsize=9, weight="bold")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])


def label_checks(split, flip_idx):
    os.makedirs(OUT_DIR, exist_ok=True)
    pick = [("train", s, f) for s, f in [("000064", 30), ("000010", 40), ("000087", 50), ("000124", 20),
                                         ("000096", 60), ("000011", 80), ("000123", 70), ("000125", 10)]]
    pick += [("val", s, f) for s, f in [("000043", 30), ("000042", 50), ("000044", 70), ("000045", 20)]]
    for k in range(3):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (part, scene, frame) in zip(axes.ravel(), pick[4 * k:4 * k + 4]):
            name = f"{scene}_{frame:06d}"
            draw(ax, os.path.join(YOLO_DIR, "images", part, name + ".jpg"), os.path.join(YOLO_DIR, "labels", part, name + ".txt"))
            n = int(sum(1 for _ in read_label(os.path.join(YOLO_DIR, "labels", part, name + ".txt"))[1] if _[2] > 0))
            ax.set_title(f"{part} {scene} f{frame}  ({n}/13 visible)", fontsize=10)
        fig.suptitle("Label check: box (yellow) and keypoint indices; blue = left, red = right, green = centreline", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"label_check_{k + 1}.png"), dpi=130)
        plt.close(fig)
    # flip check: the horizontally flipped image with the labels transformed by flip_idx must still sit on the same parts
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, (part, scene, frame) in enumerate([("train", "000064", 30), ("train", "000010", 40), ("val", "000043", 30)]):
        name = f"{scene}_{frame:06d}"
        ip, lp = (os.path.join(YOLO_DIR, d, part, name + e) for d, e in (("images", ".jpg"), ("labels", ".txt")))
        draw(axes[0, col], ip, lp)
        draw(axes[1, col], ip, lp, flip_idx=flip_idx, flip=True)
        axes[0, col].set_title(f"original: {scene} f{frame}", fontsize=10)
        axes[1, col].set_title("flipped image, labels flipped with flip_idx\n(same index must stay on the same part, left/right colours swap)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "label_check_flip.png"), dpi=130)
    plt.close(fig)


def main():
    split = load_split()
    assert not set(split["train"]) & set(split["val"]) and not set(split["train"]) & set(split["test"]) \
        and not set(split["val"]) & set(split["test"]), "scenes appear in more than one split"
    assert "000125" not in split["test"]
    flip_idx = flip_idx_from_names(cfg.KEYPOINT_NAMES)
    mismatch = check_flip_idx(flip_idx)
    print(f"flip_idx = {flip_idx}; mirror mismatch per keypoint [m]: {np.round(mismatch, 2).tolist()}")
    stats = build(split)
    write_yaml(os.path.join(YOLO_DIR, "data.yaml"), flip_idx)
    label_checks(split, flip_idx)
    size = make_zip()
    print(json.dumps(stats, indent=1))
    print(f"wrote {ZIP_OUT}: {size:.0f} MB")
    with open(os.path.join(OUT_DIR, "dataset_summary.json"), "w") as f:
        json.dump({"split": {k: split[k] for k in ("train", "val", "test")}, "flip_idx": flip_idx,
                   "flip_mirror_mismatch_m": [round(float(x), 3) for x in mismatch], "jpeg_quality": JPEG_QUALITY,
                   "box_margin_px": BOX_MARGIN_PX, "stats": stats, "zip_mb": round(size, 1)}, f, indent=1)


if __name__ == "__main__":
    main()
