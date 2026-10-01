"""Sanity checks for Phase 4 (prints PASS/FAIL). Run after `python classifier.py`."""

import sys

import numpy as np
from scipy.spatial.transform import Rotation

import classifier
import config
import filters
import geometry
import pnp
import simulator

results = []


def check(name, ok):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


def pose_errors(a, b):
    """Position error (m) and geodesic attitude error (deg) between two poses."""
    pos = np.linalg.norm(a[:3] - b[:3])
    ra = Rotation.from_euler("ZYX", a[[5, 4, 3]], degrees=True)
    rb = Rotation.from_euler("ZYX", b[[5, 4, 3]], degrees=True)
    return pos, np.degrees((ra.inv() * rb).magnitude())


rng = np.random.default_rng(0)
model = classifier.load_model()

# --- camera_to_pose inverts pose_to_camera ----------------------------------
max_pos = max_ang = 0.0
for pose in simulator.sample_random_poses(20, rng):
    R_cam, t_cam = geometry.pose_to_camera(pose)
    back = geometry.camera_to_pose(R_cam, t_cam, geometry.boresight_unit(pose))
    max_pos = max(max_pos, np.abs(back[:3] - pose[:3]).max())
    max_ang = max(max_ang, pose_errors(pose, back)[1])
check(f"camera_to_pose(pose_to_camera(p)) recovers p for 20 poses "
      f"(max {max_pos:.1e} m, {max_ang:.1e} deg)", max_pos < 1e-6 and max_ang < 1e-6)

# --- zero noise: both PnP methods are exact ---------------------------------
worst = {"pnp_only": (0.0, 0.0), "classifier_pnp": (0.0, 0.0)}
n_tested = 0
for pose in simulator.sample_random_poses(300, rng):
    obs = filters.make_observation(pose)
    if obs.visible.sum() < 6:
        continue
    n_tested += 1
    coarse = classifier.predict_pose(model, obs.features[None])[0]
    for name, est in [("pnp_only", pnp.pnp_only(obs)[0]),
                      ("classifier_pnp", pnp.classifier_pnp(obs, coarse)[0])]:
        if est is None:
            worst[name] = (np.inf, np.inf)
        else:
            e = pose_errors(pose, est)
            worst[name] = (max(worst[name][0], e[0]), max(worst[name][1], e[1]))
for name, (ep, ea) in worst.items():
    check(f"zero noise, {n_tested} poses with >= 6 visible: {name} exact "
          f"(max {ep:.1e} m, {ea:.1e} deg)", ep < 1e-3 and ea < 1e-3)

# --- too few keypoints -------------------------------------------------------
pose = simulator.sample_random_poses(1, rng)[0]
obs = filters.make_observation(pose)
coarse = classifier.predict_pose(model, obs.features[None])[0]
for n_vis in (0, 3):
    few = filters.make_observation(pose)
    few.visible[:] = False
    few.visible[:n_vis] = True
    try:
        est, info = pnp.pnp_only(few)
        check(f"pnp_only returns None with {n_vis} visible keypoints", est is None and info["failed"])
    except Exception as e:
        check(f"pnp_only does not crash with {n_vis} visible keypoints ({e!r})", False)
    try:
        est, info = pnp.classifier_pnp(few, coarse)
        check(f"classifier_pnp falls back to the coarse pose with {n_vis} visible keypoints",
              info["fallback"] and np.array_equal(est, coarse))
    except Exception as e:
        check(f"classifier_pnp does not crash with {n_vis} visible keypoints ({e!r})", False)

# --- plausibility ------------------------------------------------------------
check("plausibility: behind the camera is rejected", not pnp._plausible([0, 0, -50]))
check("plausibility: range outside [10, 300] m is rejected",
      not pnp._plausible([0, 0, 5]) and not pnp._plausible([0, 0, 400]) and pnp._plausible([0, 0, 50]))

# --- no NaNs in classifier+PnP on noisy random poses and the trajectory -----
poses = np.vstack([simulator.sample_random_poses(300, rng), simulator.approach_trajectory()[1]])
est = np.empty_like(poses)
for i, pose in enumerate(poses):
    obs = filters.make_observation(pose, noise_px=5.0, rng=rng)
    coarse = classifier.predict_pose(model, obs.features[None])[0]
    est[i] = pnp.classifier_pnp(obs, coarse)[0]
check("classifier+PnP results have no NaNs (noise 5 px, 400 frames)", np.isfinite(est).all())

# --- Phase 4b: hybrid and the dropout model ------------------------------------
worst_gap, zero_noise_worst, n_h = -np.inf, 0.0, 0
for noise in (0.0, 2.0):
    for pose in simulator.sample_random_poses(150, rng):
        obs = filters.make_observation(pose, noise, rng)
        coarse = classifier.predict_pose(model, obs.features[None])[0]
        _, info_h = pnp.hybrid_pnp(obs, coarse)
        _, info_sq = pnp.pnp_only(obs)
        _, info_cl = pnp.classifier_pnp(obs, coarse)
        cands = ([info_sq["rmse"]] if not info_sq["failed"] else []) + \
                ([info_cl["rmse"]] if not info_cl["fallback"] else [])
        if cands:
            worst_gap = max(worst_gap, info_h["rmse"] - min(cands))
        if noise == 0.0 and obs.visible.sum() >= 6:
            n_h += 1
            est = pnp.hybrid_pnp(obs, coarse)[0]
            zero_noise_worst = max(zero_noise_worst, *pose_errors(pose, est))
check(f"hybrid RMSE never exceeds the better candidate (worst gap {worst_gap:.1e} px)", worst_gap <= 1e-9)
check(f"hybrid with zero noise is exact on {n_h} poses (worst {zero_noise_worst:.1e})",
      zero_noise_worst < 1e-3)

few = filters.make_observation(pose)
few.visible[:] = False
est, info = pnp.hybrid_pnp(few, coarse)
check("hybrid falls back to the coarse pose when both methods fail",
      info["fallback"] and info["source"] == "coarse" and np.array_equal(est, coarse))

import os
import train_dropout
check("models/classifier_dropout.pt exists", os.path.exists(train_dropout.DROPOUT_MODEL_PATH))
dmodel = classifier.load_model(train_dropout.DROPOUT_MODEL_PATH)
pred = classifier.predict_pose(dmodel, filters.make_observation(poses[0]).features[None])
check("dropout model loads and predicts a finite (1, 6) pose", pred.shape == (1, 6) and np.isfinite(pred).all())

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
