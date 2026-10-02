"""
Phase 7a: choose keypoints on the FW-UAV model from the mesh geometry, write them to fw_uav_config.py and
draw results/phase7a/uav_keypoints.png (top, side and front views).

The model is a VTOL fixed-wing UAV: one fuselage with a nose propeller, a straight tapered wing, two booms with
four lift rotors, and a T-tail. Keypoints are picked by simple extreme-point rules in the dataset object frame
(x forward, y right, z down, metres):
    nose            max x on the centreline
    tail            min x on the centreline (trailing edge of the tail)
    fin top         highest vertex
    stabiliser tips left / right: extreme |y| of the tail plane (x < -15)
    wing tips       leading / trailing edge corner of each wing tip (largest |y| of the wing)
    wing roots      leading / trailing edge of each wing at |y| = 4.2 m (just outside the fuselage)
13 keypoints (the rotors are not used: their blades are thin and spin).
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fw_uav_config as cfg
import fw_uav_io as io

OUT_DIR = os.path.join("results", "phase7a")


def object_frame_vertices():
    V, _ = io.load_obj_vertices()
    V = np.unique(V.round(2), axis=0)
    return (V * cfg.MODEL_UNIT_TO_M) @ cfg.MODEL_ROTATION.T


def select_keypoints(P):
    height = -P[:, 2]
    names, pts = [], []

    def add(name, mask, key):
        idx = np.flatnonzero(mask)
        pts.append(P[idx[np.argmax(key[idx])]])
        names.append(name)

    centre = np.abs(P[:, 1]) < 0.8
    add("nose", centre & (height < 2.0), P[:, 0])
    add("tail", centre, -P[:, 0])
    add("fin_top", np.ones(len(P), bool), height)
    tail = P[:, 0] < -15.0
    add("stabiliser_tip_left", tail & (P[:, 1] < 0), -P[:, 1])
    add("stabiliser_tip_right", tail & (P[:, 1] > 0), P[:, 1])
    wing = (height > 0.3) & (height < 1.6) & (P[:, 0] > -13.0)
    for side, sign in (("left", -1), ("right", 1)):
        s = sign * P[:, 1]
        tip = wing & (s > s[wing].max() - 0.7)
        add(f"wingtip_{side}_leading", tip, P[:, 0])
        add(f"wingtip_{side}_trailing", tip, -P[:, 0])
        root = wing & (np.abs(s - 4.2) < 0.3)
        add(f"wing_root_{side}_leading", root, P[:, 0])
        add(f"wing_root_{side}_trailing", root, -P[:, 0])
    return names, np.array(pts)


def write_config(names, pts):
    path = "fw_uav_config.py"
    src = open(path).read()
    head = src[:src.index("KEYPOINT_NAMES = ")]
    body = "KEYPOINT_NAMES = [\n" + "".join(f'    "{n}",\n' for n in names) + "]\nKEYPOINTS = np.array([\n"
    body += "".join(f"    [{x:.3f}, {y:.3f}, {z:.3f}],\n" for x, y, z in pts) + "])\n"
    open(path, "w").write(head + body)


def plot(P, names, pts):
    rng = np.random.default_rng(0)
    S = P[rng.choice(len(P), min(40000, len(P)), replace=False)]
    views = [("Top view (x forward, y right)", 0, 1, 1), ("Side view (x forward, up)", 0, 2, -1),
             ("Front view (y right, up)", 1, 2, -1)]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw={"width_ratios": [1, 1.3, 1.3]})
    for ax, (title, a, b, sb) in zip(axes, views):
        ax.scatter(S[:, a], sb * S[:, b], s=0.4, color="0.6")
        ax.scatter(pts[:, a], sb * pts[:, b], s=40, color="tab:red", zorder=3)
        for i, n in enumerate(names):
            ax.annotate(f"{i + 1}", (pts[i, a], sb * pts[i, b]), xytext=(4, 4), textcoords="offset points",
                        fontsize=11, color="tab:red", weight="bold")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel(["x [m]", "y [m]", "z [m]"][a])
        ax.set_ylabel(["x [m]", "y [m]", "height (-z) [m]" if sb < 0 else "z [m]"][b] if b else "y [m]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("y [m]")
    axes[1].set_ylabel("height, -z [m]")
    axes[2].set_ylabel("height, -z [m]")
    legend = "   ".join(f"{i + 1} {n}" for i, n in enumerate(names))
    fig.suptitle("FW-UAV model (object frame, metres) with the 13 chosen keypoints\n" + legend, fontsize=10)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "uav_keypoints.png"), dpi=150)
    plt.close(fig)


def main():
    P = object_frame_vertices()
    names, pts = select_keypoints(P)
    write_config(names, pts)
    plot(P, names, pts)
    print("model extent x/y/z [m]:", np.ptp(P, axis=0).round(2))
    for n, p in zip(names, pts):
        print(f"{n:26s} {p.round(2)}")


if __name__ == "__main__":
    main()
