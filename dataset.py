"""
Synthetic dataset generation: random poses -> keypoint features + grid labels.

Usage:
    python dataset.py --n_train 100000 --n_val 20000 --noise_px 0
"""

import argparse
import os
import time

import numpy as np

import config
import geometry
import simulator


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def pose_to_label(poses):
    """(n, 6) poses -> (n,) labels: nearest grid point in x, y, z, roll."""
    poses = np.atleast_2d(poses)
    idx = [np.abs(poses[:, [d]] - grid[None, :]).argmin(axis=1)
           for d, grid in enumerate(config.GRID)]
    return np.ravel_multi_index(idx, config.GRID_SHAPE)


def label_to_pose(labels):
    """(n,) labels -> (n, 6) grid-point poses with pitch = yaw = 0."""
    idx = np.unravel_index(np.asarray(labels), config.GRID_SHAPE)
    x, y, z, roll = (grid[i] for grid, i in zip(config.GRID, idx))
    zeros = np.zeros_like(x)
    return np.column_stack([x, y, z, roll, zeros, zeros])


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def make_features(uv, vis, boresight):
    """
    45 numbers: (u, v, visible) per keypoint (42), then the 3-component gimbal
    boresight unit vector in follower body axes. u, v are normalised to
    [-1, 1] about the image centre. Hidden keypoints get u = v = 0, visible = 0.

    The gimbal centres the leader in the image, so the pixels alone lose the
    viewing direction; a real gimbal knows where it points, so we add it.
    """
    u = (uv[:, 0] - config.CX) / config.CX
    v = (uv[:, 1] - config.CY) / config.CY
    feats = np.column_stack([u, v, vis.astype(float)])
    feats[~vis, :2] = 0.0
    return np.concatenate([feats.ravel(), boresight])


def generate(n, noise_px=0.0, seed=config.SEED, gimbal_noise_deg=config.GIMBAL_NOISE_DEG):
    """Returns X (n, 45), y (n,), poses (n, 6)."""
    rng = np.random.default_rng(seed)
    poses = simulator.sample_random_poses(n, rng)
    X = np.empty((n, config.N_FEATURES))
    for i, pose in enumerate(poses):
        uv, vis = geometry.observe(pose, noise_px, rng)
        boresight = geometry.boresight_unit(pose, gimbal_noise_deg, rng)
        X[i] = make_features(uv, vis, boresight)
    y = pose_to_label(poses)
    return X, y, poses


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
# Pairs of keypoint names joined by lines to form a rough wireframe
WIREFRAME = [
    ("nose", "canopy"), ("canopy", "fin_root"), ("fin_root", "fin_top"),
    ("fin_top", "tail_cone"), ("nose", "intake"),
    ("nose", "left_wing_root"), ("left_wing_root", "left_wingtip"),
    ("left_wingtip", "tail_cone"), ("tail_cone", "left_hstab_tip"),
    ("left_wing_root", "left_store"),
    ("nose", "right_wing_root"), ("right_wing_root", "right_wingtip"),
    ("right_wingtip", "tail_cone"), ("tail_cone", "right_hstab_tip"),
    ("right_wing_root", "right_store"),
]


def plot_samples(poses=None, path=None, seed=config.SEED):
    """2x3 grid of validation samples saved to results/sample_keypoints.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if poses is None:
        poses = np.load(os.path.join(config.DATA_DIR, "val.npz"))["poses"]
    if path is None:
        path = os.path.join(config.RESULTS_DIR, "sample_keypoints.png")
    rng = np.random.default_rng(seed)
    chosen = poses[rng.choice(len(poses), 6, replace=False)]
    name_idx = {name: i for i, name in enumerate(config.KEYPOINT_NAMES)}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, pose in zip(axes.ravel(), chosen):
        uv, vis = geometry.observe(pose)
        for a, b in WIREFRAME:
            i, j = name_idx[a], name_idx[b]
            ax.plot(uv[[i, j], 0], uv[[i, j], 1], "-", color="0.6", lw=0.8)
        ax.scatter(uv[vis, 0], uv[vis, 1], s=25, c="red", zorder=3, label="visible")
        ax.scatter(uv[~vis, 0], uv[~vis, 1], s=25, facecolors="none",
                   edgecolors="grey", zorder=3, label="hidden")
        ax.plot([0, config.IMAGE_WIDTH, config.IMAGE_WIDTH, 0, 0],
                [0, 0, config.IMAGE_HEIGHT, config.IMAGE_HEIGHT, 0], "k-", lw=0.5)

        # Zoom on the aircraft (it is small in a 1280x960 image at long range)
        centre = uv.mean(axis=0)
        half = max(np.ptp(uv, axis=0).max() * 0.65, 20)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] + half, centre[1] - half)   # image y-axis inverted
        ax.set_aspect("equal")
        x, y, z, r, p, w = pose
        ax.set_title(f"x={x:.0f} y={y:.0f} z={z:.0f} m\n"
                     f"roll={r:.0f} pitch={p:.0f} yaw={w:.0f} deg | "
                     f"{vis.sum()}/{config.N_KEYPOINTS} visible", fontsize=9)
        ax.set_xlabel("u [px]")
        ax.set_ylabel("v [px]")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("--n_val", type=int, default=20000)
    parser.add_argument("--noise_px", type=float, default=config.KEYPOINT_NOISE_PX)
    parser.add_argument("--gimbal_noise_deg", type=float, default=config.GIMBAL_NOISE_DEG)
    args = parser.parse_args()

    os.makedirs(config.DATA_DIR, exist_ok=True)
    splits = [("train", args.n_train, config.SEED), ("val", args.n_val, config.SEED + 1)]
    for name, n, seed in splits:
        t0 = time.time()
        X, y, poses = generate(n, args.noise_px, seed, args.gimbal_noise_deg)
        path = os.path.join(config.DATA_DIR, f"{name}.npz")
        np.savez_compressed(path, X=X, y=y, poses=poses)
        n_visible = X[:, 2:3 * config.N_KEYPOINTS:3].sum(axis=1)
        print(f"{name}: {n} samples in {time.time() - t0:.1f} s -> {path}")
        print(f"  X {X.shape}, distinct labels used: {len(np.unique(y))}/{config.N_LABELS}")
        print(f"  mean visible keypoints: {n_visible.mean():.2f}/{config.N_KEYPOINTS}")

    plot_samples()


if __name__ == "__main__":
    main()
