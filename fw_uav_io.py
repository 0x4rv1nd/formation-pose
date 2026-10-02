"""
Phase 7a: loading the FW-UAV6DPose validation split (BOP format) and its 3D model.

Poses are model-to-camera (cam_R_m2c, cam_t_m2c). Translations are in millimetres in the files and are
returned in metres here. Only the small JSON files and the few masks that are needed are read from the zip.
"""

import json
import os
import zipfile

import cv2
import numpy as np

DATA_DIR = os.path.join("data", "fw_uav")
ZIP_PATH = os.path.join(DATA_DIR, "val.zip")
META_DIR = os.path.join(DATA_DIR, "val_meta", "val")
MASK_DIR = os.path.join(DATA_DIR, "val_masks")
OBJ_PATH = os.path.join(DATA_DIR, "models", "D0108-C20821-FixedWings-Merge.OBJ")
IMAGE_SIZE = (1920, 1080)        # width, height
MM_PER_M = 1000.0


def scenes():
    return sorted(d for d in os.listdir(META_DIR) if d.isdigit())


def load_scene(scene):
    """-> (frame ids (T,), R (T,3,3) model-to-camera, t (T,3) in METRES, K (3,3)). One object per frame."""
    with open(os.path.join(META_DIR, scene, "scene_gt.json")) as f:
        gt = json.load(f)
    with open(os.path.join(META_DIR, scene, "scene_camera.json")) as f:
        cam = json.load(f)
    ids = np.array(sorted(gt, key=int), dtype=int)
    R = np.array([gt[str(i)][0]["cam_R_m2c"] for i in ids]).reshape(-1, 3, 3)
    t = np.array([gt[str(i)][0]["cam_t_m2c"] for i in ids]) / MM_PER_M
    K = np.array(cam[str(ids[0])]["cam_K"]).reshape(3, 3)
    return ids, R, t, K


def load_obj_vertices(path=OBJ_PATH):
    """Unique vertices of the OBJ (in the file's own units) and the triangle index array."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
    return np.array(verts), np.array(faces)


def read_mask(scene, frame, zf=None):
    """Visible-object mask (H, W) bool for a frame; cached as a small PNG in data/fw_uav/val_masks/."""
    cache = os.path.join(MASK_DIR, f"{scene}_{frame:06d}.png")
    if os.path.exists(cache):
        img = cv2.imread(cache, cv2.IMREAD_UNCHANGED)
    else:
        os.makedirs(MASK_DIR, exist_ok=True)
        own = zf is None
        zf = zf or zipfile.ZipFile(ZIP_PATH)
        raw = zf.read(f"val/{scene}/mask_visib/{frame:06d}_000000.png")
        if own:
            zf.close()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(cache, (img[..., :3].max(axis=2) > 0).astype(np.uint8) * 255)
    return (img if img.ndim == 2 else img[..., :3].max(axis=2)) > 0


# ---------------------------------------------------------------------------
# Keypoint projection and visibility (ground-truth poses, no noise)
# ---------------------------------------------------------------------------
CACHE_PATH = os.path.join(DATA_DIR, "val_keypoints.npz")
MASK_DILATE_PX = 2          # a keypoint counts as visible if it lies inside the mask dilated by this many pixels


def project_points(points, R, t, K):
    """Pixels (N, 2) and camera-frame depths (N,) of object-frame points (N, 3) for one pose."""
    X = points @ R.T + t
    uv = X @ K.T
    return uv[:, :2] / uv[:, 2:3], X[:, 2]


def build_keypoint_cache(points, force=False):
    """
    Project the keypoints with the ground-truth pose of every frame and decide their visibility:
        vis_strict  inside the image and inside the visible-object mask
        vis_dilated inside the image and inside the mask dilated by MASK_DILATE_PX pixels
    (keypoints on the silhouette boundary can fall one pixel outside the rasterised mask). Self-occlusion inside
    the silhouette is ignored. Cached in data/fw_uav/val_keypoints.npz (rebuilt if the keypoints change).
    Returns {scene: dict(frames, R, t, K, uv, vis_strict, vis_dilated)}.
    """
    if os.path.exists(CACHE_PATH) and not force:
        z = np.load(CACHE_PATH)
        if z["points"].shape == points.shape and np.allclose(z["points"], points):
            return {s: {k: z[f"{s}/{k}"] for k in ("frames", "R", "t", "K", "uv", "vis_strict", "vis_dilated")}
                    for s in scenes()}
    kernel = np.ones((2 * MASK_DILATE_PX + 1,) * 2, np.uint8)
    out, flat = {}, {"points": points}
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for scene in scenes():
            ids, R, t, K = load_scene(scene)
            uv = np.zeros((len(ids), len(points), 2))
            strict = np.zeros((len(ids), len(points)), bool)
            dil = np.zeros_like(strict)
            for i, frame in enumerate(ids):
                uv[i], depth = project_points(points, R[i], t[i], K)
                mask = read_mask(scene, int(frame), zf)
                mask_d = cv2.dilate(mask.astype(np.uint8), kernel) > 0
                u, v = np.round(uv[i]).astype(int).T
                inside = (u >= 0) & (u < IMAGE_SIZE[0]) & (v >= 0) & (v < IMAGE_SIZE[1]) & (depth > 0)
                strict[i][inside] = mask[v[inside], u[inside]]
                dil[i][inside] = mask_d[v[inside], u[inside]]
            out[scene] = dict(frames=ids, R=R, t=t, K=K, uv=uv, vis_strict=strict, vis_dilated=dil)
            for k, a in out[scene].items():
                flat[f"{scene}/{k}"] = a
            print(f"  cached keypoints of scene {scene}")
    np.savez_compressed(CACHE_PATH, **flat)
    return out
