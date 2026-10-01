"""Phase 1 sanity checks. Run: python test_phase1.py"""

import cv2
import numpy as np

import config
import dataset
import geometry

results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append(bool(condition))
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


def kp(name):
    return config.KEYPOINT_NAMES.index(name)


# --- Follower directly behind the leader ------------------------------------
behind = [-60, 0, 0, 0, 0, 0]
cg = geometry.project_points(behind, np.zeros((1, 3)))[0]
check("leader CG projects to image centre (640, 480)",
      np.allclose(cg, [640, 480], atol=1e-9), f"got {cg}")

uv, vis = geometry.observe(behind)
check("left wingtip u < right wingtip u",
      uv[kp("left_wingtip"), 0] < uv[kp("right_wingtip"), 0],
      f"{uv[kp('left_wingtip'), 0]:.1f} vs {uv[kp('right_wingtip'), 0]:.1f}")
check("fin top v < 480", uv[kp("fin_top"), 1] < 480, f"v = {uv[kp('fin_top'), 1]:.1f}")
check("nose hidden when viewed from behind", not vis[kp("nose")])
check("tail cone visible when viewed from behind", vis[kp("tail_cone")])

# --- Convention: position is in FOLLOWER axes (distinguishes from leader axes)
yawed = [-60, 0, 0, 0, 0, 30]
R_cam, t_cam = geometry.pose_to_camera(yawed)
leader_cg_follower = -np.array(yawed[:3])
check("leader CG in follower frame is [60, 0, 0]",
      np.allclose(leader_cg_follower, [60, 0, 0]))
R_c = R_cam @ geometry.euler_to_R(0, 0, 30).T     # follower body -> camera
boresight = R_c.T @ np.array([0, 0, 1.0])
check("boresight points along follower x-axis with yaw = 30",
      np.allclose(boresight, [1, 0, 0]), f"got {boresight}")
check("CG camera coords = [0, 0, 60] with yaw = 30",
      np.allclose(R_c @ leader_cg_follower, [0, 0, 60]))
cg = geometry.project_points(yawed, np.zeros((1, 3)))[0]
check("yawed pose: leader CG projects to (640, 480)",
      np.allclose(cg, [640, 480], atol=1e-9), f"got {cg}")

# --- Gimbal boresight --------------------------------------------------------
b = geometry.boresight_unit([-60, 0, 0, 0, 0, 0])
check("boresight for [-60,0,0,0,0,0] is [1,0,0]", np.allclose(b, [1, 0, 0]), f"got {b}")
b = geometry.boresight_unit([-60, 0, 0, 0, 0, 30])
check("boresight for yaw = 30 is still [1,0,0]", np.allclose(b, [1, 0, 0]), f"got {b}")
b = geometry.boresight_unit([30, 40, -10, 0, 0, 0], 2.0, np.random.default_rng(0))
check("noisy boresight is a unit vector", np.isclose(np.linalg.norm(b), 1.0))

# --- Follower above and behind -----------------------------------------------
above = [-40, 0, -20, 0, 0, 0]
_, vis = geometry.observe(above)
hidden = [n for n in ("left_store", "right_store", "intake") if not vis[kp(n)]]
check("stores and intake hidden from above", len(hidden) == 3, f"hidden: {hidden}")
check("fin top visible from above", vis[kp("fin_top")])

# --- Agreement with OpenCV -------------------------------------------------
pose = [-55, 12, -8, 20, -7, 9]
R_cam, t_cam = geometry.pose_to_camera(pose)
rvec, _ = cv2.Rodrigues(R_cam)
uv_cv, _ = cv2.projectPoints(config.KEYPOINTS, rvec, t_cam, config.K, None)
err = np.abs(uv_cv.reshape(-1, 2) - geometry.project_points(pose)).max()
check("project_points matches cv2.projectPoints", err < 1e-6, f"max err {err:.2e} px")

# --- Euler round trip --------------------------------------------------------
angles = geometry.R_to_euler(geometry.euler_to_R(20, -7, 9))
check("euler_to_R / R_to_euler round trip", np.allclose(angles, [20, -7, 9]))

# --- Labels -----------------------------------------------------------------
rng = np.random.default_rng(0)
p = np.column_stack([rng.uniform(-100, 50, 500), rng.uniform(-50, 50, 500),
                     rng.uniform(-20, 20, 500), rng.uniform(-45, 45, 500),
                     rng.uniform(-10, 10, (500, 2))])
back = dataset.label_to_pose(dataset.pose_to_label(p))
nearest = np.column_stack([g[np.abs(p[:, [d]] - g).argmin(axis=1)]
                           for d, g in enumerate(config.GRID)])
check("label_to_pose(pose_to_label(p)) is the nearest grid point",
      np.allclose(back[:, :4], nearest) and np.all(back[:, 4:] == 0))

# --- Dataset ----------------------------------------------------------------
X, y, poses = dataset.generate(200, 0.0, seed=123)
check("dataset X has shape (n, 45)", X.shape == (200, 45), f"{X.shape}")
check("last 3 features are unit boresight = -pos/|pos|",
      np.allclose(X[:, 42:], -poses[:, :3] / np.linalg.norm(poses[:, :3], axis=1, keepdims=True)))
check("labels in [0, 4800)", y.min() >= 0 and y.max() < config.N_LABELS)
check("all sampled ranges >= 25 m",
      np.linalg.norm(poses[:, :3], axis=1).min() >= config.MIN_RANGE_M)

print(f"\n{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
