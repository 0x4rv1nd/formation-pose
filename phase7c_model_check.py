"""
Phase 7c, step 2: the Phase 7a model check (corrected 3D model projected with the ground-truth pose vs the visible mask)
on the middle frame of every training.zip scene. Writes results/phase7c/model_check_training.json and
model_vs_mask_training_scenes.png. Reuses silhouette() and iou() of phase7a_model_check.py.
"""
import json
import os
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2

import fw_uav_config as cfg
import fw_uav_io as io
from phase7a_model_check import silhouette, iou

TRAIN_ZIP = os.path.join(io.DATA_DIR, "training.zip")
TRAIN_META = os.path.join(io.DATA_DIR, "train_meta", "training")
OUT_DIR = os.path.join("results", "phase7c")


def train_scenes():
    return sorted(d for d in os.listdir(TRAIN_META) if d.isdigit())


def load_train_scene(scene):
    """Same as fw_uav_io.load_scene, for a training.zip scene (JSON copied to data/fw_uav/train_meta/)."""
    with open(os.path.join(TRAIN_META, scene, "scene_gt.json")) as f:
        gt = json.load(f)
    with open(os.path.join(TRAIN_META, scene, "scene_camera.json")) as f:
        cam = json.load(f)
    ids = np.array(sorted(gt, key=int), dtype=int)
    R = np.array([gt[str(i)][0]["cam_R_m2c"] for i in ids]).reshape(-1, 3, 3)
    t = np.array([gt[str(i)][0]["cam_t_m2c"] for i in ids]) / io.MM_PER_M
    K = np.array(cam[str(ids[0])]["cam_K"]).reshape(3, 3)
    return ids, R, t, K


def read_train_mask(zf, scene, frame):
    img = cv2.imdecode(np.frombuffer(zf.read(f"training/{scene}/mask_visib/{frame:06d}_000000.png"), np.uint8), cv2.IMREAD_UNCHANGED)
    return (img if img.ndim == 2 else img[..., :3].max(axis=2)) > 0


def main():
    V, F = io.load_obj_vertices()
    Vc = V * cfg.MODEL_UNIT_TO_M @ cfg.MODEL_ROTATION.T
    names = train_scenes()
    rows = []
    fig, axes = plt.subplots(10, 9, figsize=(24, 28))
    with zipfile.ZipFile(TRAIN_ZIP) as zf:
        for ax, scene in zip(axes.ravel(), names):
            ids, R, t, K = load_train_scene(scene)
            i = len(ids) // 2
            mask = read_train_mask(zf, scene, int(ids[i]))
            sil = silhouette(Vc, F, R[i], t[i], K)
            ys, xs = np.nonzero(mask)
            pad = 25 + 0.25 * max(np.ptp(xs), np.ptp(ys))
            x0, x1 = int(max(xs.min() - pad, 0)), int(min(xs.max() + pad, cfg.IMAGE_W))
            y0, y1 = int(max(ys.min() - pad, 0)), int(min(ys.max() + pad, cfg.IMAGE_H))
            rgb = np.zeros(mask.shape + (3,), np.float32)
            rgb[..., 1], rgb[..., 0] = mask * 0.9, sil * 0.9
            ax.imshow(rgb[y0:y1, x0:x1])
            ax.set_xticks([]); ax.set_yticks([])
            v = iou(sil, mask)
            ax.set_title(f"{scene} f{int(ids[i])}, {np.linalg.norm(t[i]):.0f} m\nIoU {v:.2f}", fontsize=8)
            rows.append({"scene": scene, "frame": int(ids[i]), "range_m": float(np.linalg.norm(t[i])), "mask_px": int(mask.sum()), "iou": v})
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Corrected model silhouette (red) vs visible mask (green), middle frame of every training.zip scene")
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "model_vs_mask_training_scenes.png"), dpi=70)
    ious = np.array([r["iou"] for r in rows])
    dist = {"min": float(ious.min()), "median": float(np.median(ious)), "max": float(ious.max())}
    print(f"{len(ious)} training.zip scenes: IoU min {dist['min']:.2f} / median {dist['median']:.2f} / max {dist['max']:.2f}")
    for r in sorted(rows, key=lambda r: r["iou"])[:6]:
        print(f"  lowest: {r['scene']} f{r['frame']} IoU {r['iou']:.2f} ({r['range_m']:.0f} m, {r['mask_px']} px)")
    print("scenes with IoU < 0.5:", [r["scene"] for r in rows if r["iou"] < 0.5])
    json.dump({"scenes": rows, "iou_distribution": dist}, open(os.path.join(OUT_DIR, "model_check_training.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
