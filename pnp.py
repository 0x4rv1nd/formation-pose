"""
Phase 4: PnP pose estimation from the keypoints visible in one frame.

Two single-frame methods (plus the classifier alone, which lives in classifier.py):
    pnp_only(obs)                   SQPnP, no initial guess, no classifier
    classifier_pnp(obs, coarse)     iterative PnP started from the classifier's pose, then LM

cv2.solvePnP returns (rvec, tvec) with X_cam = R @ P_leader + t, which is what
geometry.pose_to_camera produces, so geometry.camera_to_pose converts back to
our pose. The camera frame is turned into the follower frame with the MEASURED
boresight (the gimbal direction), never with the estimate itself.
"""

import cv2
import numpy as np

import config
import geometry

MIN_POINTS = 4                  # solvePnP needs at least 4 point correspondences here
MIN_RANGE_M, MAX_RANGE_M = 10.0, 300.0   # plausible distance to the leader


def _visible_points(obs):
    """3D keypoints (leader frame) and their pixels, for the visible ones only."""
    vis = obs.visible
    return (np.ascontiguousarray(config.KEYPOINTS[vis], dtype=np.float64),
            np.ascontiguousarray(obs.uv[vis], dtype=np.float64))


def _plausible(tvec):
    """Leader must be in front of the camera and at a sensible range."""
    t = np.asarray(tvec, dtype=float).ravel()
    return bool(np.all(np.isfinite(t)) and t[2] > 0 and
                MIN_RANGE_M <= np.linalg.norm(t) <= MAX_RANGE_M)


def _rmse(obj, img, rvec, tvec):
    """Reprojection error in pixels: sqrt(mean over points of the squared 2D distance)."""
    proj, _ = cv2.projectPoints(obj, rvec, tvec, config.K, None)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))


def pnp_only(obs):
    """
    SQPnP on the visible keypoints, no initial guess.
    Returns (pose or None, info). None if < 4 keypoints are visible, the solver
    fails, or the solution is implausible (behind the camera / range outside [10, 300] m).
    info: n_visible, rmse (px, nan on failure), failed.
    """
    obj, img = _visible_points(obs)
    info = {"n_visible": len(obj), "rmse": np.nan, "failed": True}
    if len(obj) < MIN_POINTS:
        return None, info
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, config.K, None, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None, info
    if not ok or not _plausible(tvec):
        return None, info

    R_cam = cv2.Rodrigues(rvec)[0]
    info["rmse"] = _rmse(obj, img, rvec, tvec)
    info["failed"] = False
    return geometry.camera_to_pose(R_cam, tvec, obs.boresight), info


def classifier_pnp(obs, coarse_pose):
    """
    Refine the classifier's coarse pose: iterative PnP started from it
    (useExtrinsicGuess), then a Levenberg-Marquardt polish.
    Falls back to coarse_pose (info["fallback"] = True) if < 4 keypoints are
    visible, the solver fails, or the solution is implausible.
    Returns (pose, info); info: n_visible, rmse (px), fallback.
    """
    obj, img = _visible_points(obs)
    coarse_pose = np.asarray(coarse_pose, dtype=float)

    # Initial guess: the coarse pose seen through the MEASURED boresight
    R0, t0 = geometry.pose_to_camera(coarse_pose, obs.boresight)
    rvec0 = cv2.Rodrigues(R0)[0]
    tvec0 = t0.reshape(3, 1).copy()

    def fallback():
        rmse = _rmse(obj, img, rvec0, tvec0) if len(obj) > 0 else np.nan
        return coarse_pose.copy(), {"n_visible": len(obj), "rmse": rmse, "fallback": True}

    if len(obj) < MIN_POINTS:
        return fallback()
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, config.K, None, rvec0.copy(), tvec0.copy(),
                                      useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return fallback()
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, config.K, None, rvec, tvec)
    except cv2.error:
        return fallback()
    if not _plausible(tvec) or not np.all(np.isfinite(rvec)):
        return fallback()

    R_cam = cv2.Rodrigues(rvec)[0]
    pose = geometry.camera_to_pose(R_cam, tvec, obs.boresight)
    return pose, {"n_visible": len(obj), "rmse": _rmse(obj, img, rvec, tvec), "fallback": False}


def hybrid_pnp(obs, coarse_pose):
    """
    Phase 4b. Run BOTH pnp_only (SQPnP, no guess) and classifier_pnp (iterative PnP from the
    coarse pose) and keep the plausible solution with the lower reprojection RMSE. Falls
    back to coarse_pose only if both fail, so a failed classifier+PnP no longer drags the
    result to the coarse pose when SQPnP succeeded.
    Returns (pose, info); info: n_visible, rmse, fallback, source ("sqpnp" | "classifier" | "coarse").
    """
    pose_sq, info_sq = pnp_only(obs)
    pose_cl, info_cl = classifier_pnp(obs, coarse_pose)

    candidates = []                      # (rmse, source, pose)
    if pose_sq is not None:
        candidates.append((info_sq["rmse"], "sqpnp", pose_sq))
    if not info_cl["fallback"]:
        candidates.append((info_cl["rmse"], "classifier", pose_cl))

    if not candidates:                   # both failed: coarse pose (info_cl holds its RMSE)
        return pose_cl, {"n_visible": info_cl["n_visible"], "rmse": info_cl["rmse"],
                         "fallback": True, "source": "coarse"}
    rmse, source, pose = min(candidates, key=lambda c: c[0])
    return pose, {"n_visible": info_cl["n_visible"], "rmse": rmse,
                  "fallback": False, "source": source}
