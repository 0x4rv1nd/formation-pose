"""
Phase 7a: verify that the 3D model matches the dataset's ground-truth poses.

For a few frames from different scenes the model's triangles are projected with the ground-truth pose and the
frame's intrinsics and the silhouette is compared with the visible-object mask (IoU). Two model conventions are
compared: the OBJ as it is (centimetres converted to metres only) and with the correction found in
fw_uav_config.py (180 degree rotation about x). Writes results/phase7a/model_vs_mask.png and model_check.json. It then checks one frame (frame 50) of EVERY scene
with the corrected model: results/phase7a/model_vs_mask_all_scenes.png and the IoU distribution in model_check.json.
"""

import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fw_uav_config as cfg
import fw_uav_io as io

OUT_DIR = os.path.join("results", "phase7a")
FRAMES = [("000064", 30), ("000009", 0), ("000043", 10), ("000087", 50), ("000122", 50), ("000137", 99)]


def silhouette(V_obj, F, R, t, K):
    """Filled projection of all triangles (bool image)."""
    X = V_obj @ R.T + t
    uv = X @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    img = np.zeros((cfg.IMAGE_H, cfg.IMAGE_W), np.uint8)
    cv2.fillPoly(img, list(np.round(uv).astype(np.int32)[F]), 255)
    return img > 0


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def main():
    V, F = io.load_obj_vertices()
    models = {"as in file (cm to m only)": (V * cfg.MODEL_UNIT_TO_M, np.eye(3)),
              "corrected (x-axis rotation by 180 deg)": (V * cfg.MODEL_UNIT_TO_M @ cfg.MODEL_ROTATION.T, None)}
    fig, axes = plt.subplots(2, len(FRAMES), figsize=(3.2 * len(FRAMES), 6.6))
    report = []
    for j, (scene, frame) in enumerate(FRAMES):
        ids, R, t, K = io.load_scene(scene)
        i = list(ids).index(frame)
        mask = io.read_mask(scene, frame)
        ys, xs = np.nonzero(mask)
        pad = 25 + 0.25 * max(np.ptp(xs), np.ptp(ys))
        x0, x1 = int(max(xs.min() - pad, 0)), int(min(xs.max() + pad, cfg.IMAGE_W))
        y0, y1 = int(max(ys.min() - pad, 0)), int(min(ys.max() + pad, cfg.IMAGE_H))
        row = {"scene": scene, "frame": frame, "range_m": float(np.linalg.norm(t[i])), "mask_px": int(mask.sum())}
        for k, (name, (Vo, _)) in enumerate(models.items()):
            sil = silhouette(Vo, F, R[i], t[i], K)
            row[name] = iou(sil, mask)
            rgb = np.zeros(mask.shape + (3,), np.float32)
            rgb[..., 1] = mask * 0.9            # mask in green, model silhouette in red (overlap = yellow)
            rgb[..., 0] = sil * 0.9
            ax = axes[k, j]
            ax.imshow(rgb[y0:y1, x0:x1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{scene} f{frame}, {row['range_m']:.0f} m\nIoU {row[name]:.2f}", fontsize=9)
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
        report.append(row)
    fig.suptitle("3D model silhouette (red) vs ground-truth visible mask (green); yellow = overlap")
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "model_vs_mask.png"), dpi=150)
    plt.close(fig)
    for k in list(models):
        print(f"{k}: mean IoU {np.mean([r[k] for r in report]):.3f}, per frame {[round(r[k], 2) for r in report]}")

    # ---- one frame from every scene, corrected model ----
    Vc = V * cfg.MODEL_UNIT_TO_M @ cfg.MODEL_ROTATION.T
    all_scenes = []
    names = io.scenes()
    fig, axes = plt.subplots(4, 6, figsize=(19, 11.5))
    for ax, scene in zip(axes.ravel(), names):
        ids, R, t, K = io.load_scene(scene)
        i = len(ids) // 2
        mask = io.read_mask(scene, int(ids[i]))
        sil = silhouette(Vc, F, R[i], t[i], K)
        ys, xs = np.nonzero(mask)
        pad = 25 + 0.25 * max(np.ptp(xs), np.ptp(ys))
        x0, x1 = int(max(xs.min() - pad, 0)), int(min(xs.max() + pad, cfg.IMAGE_W))
        y0, y1 = int(max(ys.min() - pad, 0)), int(min(ys.max() + pad, cfg.IMAGE_H))
        rgb = np.zeros(mask.shape + (3,), np.float32)
        rgb[..., 1], rgb[..., 0] = mask * 0.9, sil * 0.9
        ax.imshow(rgb[y0:y1, x0:x1])
        ax.set_xticks([])
        ax.set_yticks([])
        v = iou(sil, mask)
        ax.set_title(f"{scene} f{int(ids[i])}, {np.linalg.norm(t[i]):.0f} m, {int(mask.sum())} px\nIoU {v:.2f}", fontsize=9)
        all_scenes.append({"scene": scene, "frame": int(ids[i]), "range_m": float(np.linalg.norm(t[i])),
                           "mask_px": int(mask.sum()), "iou": v})
    fig.suptitle("Corrected model silhouette (red) vs visible mask (green), middle frame of every scene")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "model_vs_mask_all_scenes.png"), dpi=110)
    plt.close(fig)
    ious = np.array([r["iou"] for r in all_scenes])
    dist = {"min": float(ious.min()), "median": float(np.median(ious)), "max": float(ious.max())}
    print(f"all {len(ious)} scenes, corrected model: IoU min {dist['min']:.2f} / median {dist['median']:.2f} / "
          f"max {dist['max']:.2f}")
    for r in sorted(all_scenes, key=lambda r: r["iou"])[:5]:
        print(f"  lowest: {r['scene']} f{r['frame']} IoU {r['iou']:.2f} ({r['range_m']:.0f} m, {r['mask_px']} px)")
    with open(os.path.join(OUT_DIR, "model_check.json"), "w") as f:
        json.dump({"selected_frames": report, "all_scenes": all_scenes, "iou_distribution": dist}, f, indent=1)


if __name__ == "__main__":
    main()
