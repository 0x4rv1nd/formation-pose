"""
Phase 7a: pose methods for the FW-UAV evaluation, working in the CAMERA frame.

    pnp_measurement      SQPnP on the visible keypoints (no classifier is used for this UAV)
    sample_synthetic_poses / calibrate_range_noise
                         PnP error covariance per range bin on SYNTHETIC poses made with this UAV's model,
                         intrinsics and distance range (never on the validation ground truth)
    UAVKalman            the Phase 5b range-calibrated Kalman filter (filters.RangeKalmanFilter) fed with those
                         PnP measurements; same dynamics, gating and re-initialisation, optionally another dt

Pose vector (as in the rest of the project): [x, y, z, roll, pitch, yaw], here the UAV's position in the camera
frame (m, x right / y down / z forward) and the ZYX Euler angles (deg) of R_m2c.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import config
import filters
import fw_uav_config as cfg
import fw_uav_io as io

MIN_POINTS = 4
MIN_RANGE_M, MAX_RANGE_M = 50.0, 1000.0       # plausible distance of a solution (the dataset spans 164-518 m)
GROSS_REL_POS, GROSS_ROT_DEG = 0.10, 20.0     # failure: position error > 10 % of the range or rotation > 20 deg


def rotation_to_pose_angles(R):
    yaw, pitch, roll = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)
    return np.array([roll, pitch, yaw])


def pose_to_rotation(pose):
    return Rotation.from_euler("ZYX", pose[[5, 4, 3]], degrees=True).as_matrix()


def observe(R, t, K, noise_px, rng, points=None):
    """Noisy pixels of all keypoints for one pose."""
    uv, _ = io.project_points(cfg.KEYPOINTS if points is None else points, R, t, K)
    return uv + (rng.normal(0.0, noise_px, uv.shape) if noise_px > 0 else 0.0)


def pnp_measurement(uv, visible, K):
    """SQPnP pose (6,) from the visible keypoints, or None (too few points, solver failure, implausible)."""
    if visible.sum() < MIN_POINTS:
        return None
    obj = np.ascontiguousarray(cfg.KEYPOINTS[visible], dtype=np.float64)
    img = np.ascontiguousarray(uv[visible], dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None
    t = tvec.ravel()
    if not ok or not np.all(np.isfinite(t)) or t[2] <= 0 or not MIN_RANGE_M <= np.linalg.norm(t) <= MAX_RANGE_M:
        return None
    return np.concatenate([t, rotation_to_pose_angles(cv2.Rodrigues(rvec)[0])])


def pose_errors(pose, R_true, t_true):
    """Translation error (m) and geodesic rotation error (deg) of a pose vector against the ground truth."""
    dt = np.linalg.norm(pose[:3] - t_true)
    dr = np.degrees((Rotation.from_matrix(R_true).inv() * Rotation.from_matrix(pose_to_rotation(pose))).magnitude())
    return dt, dr


# ---------------------------------------------------------------------------
# Synthetic calibration of the measurement noise (this UAV, its camera, its distance range)
# ---------------------------------------------------------------------------
CALIB_SEED = 7_000_007
CALIB_N = 4000
CALIB_MAX_ABS_PITCH = 80.0        # Euler angles are ill-defined near +/-90 deg pitch: such poses are not used for R
CALIB_MIN_N = 100


def sample_synthetic_poses(n, rng, K):
    """
    n poses: range uniform in cfg.CALIB_RANGE_M, direction through a pixel uniform inside the image (100 px
    margin), orientation uniform on SO(3). Returns R (n,3,3) and t (n,3).
    """
    r = rng.uniform(*cfg.CALIB_RANGE_M, n)
    u = rng.uniform(100, cfg.IMAGE_W - 100, n)
    v = rng.uniform(100, cfg.IMAGE_H - 100, n)
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(n)], axis=1)
    t = ray / np.linalg.norm(ray, axis=1, keepdims=True) * r[:, None]
    return Rotation.random(n, random_state=int(rng.integers(2 ** 31))).as_matrix(), t


def calibrate_range_noise(noise_px, K, n=CALIB_N, seed=CALIB_SEED):
    """
    PnP error covariance (6x6, position in m, angles in deg) in range bins on synthetic poses, the same recipe as
    filters.calibrate_pnp_noise_by_range: frames without a solution and gross failures are left out, a floor of
    0.01 m / 0.01 deg std is added, bins are linked by linear interpolation between their centres.
    The bins are the evaluation bins with the lowest one starting at the shortest calibration range.
    """
    rng = np.random.default_rng(seed)
    R, t = sample_synthetic_poses(n, rng, K)
    ranges, errs, n_missing, n_gross, n_lock = [], [], 0, 0, 0
    for i in range(n):
        uv = observe(R[i], t[i], K, noise_px, rng)
        z = pnp_measurement(uv, np.ones(len(uv), bool), K)
        if z is None:
            n_missing += 1
            continue
        r = np.linalg.norm(t[i])
        dt, dr = pose_errors(z, R[i], t[i])
        if dt > GROSS_REL_POS * r or dr > GROSS_ROT_DEG:
            n_gross += 1
            continue
        if abs(rotation_to_pose_angles(R[i])[1]) > CALIB_MAX_ABS_PITCH:
            n_lock += 1
            continue
        e = filters.pose_error_vector(z, np.concatenate([t[i], rotation_to_pose_angles(R[i])]))
        ranges.append(r)
        errs.append(e)
    ranges, errs = np.array(ranges), np.array(errs)
    floor = np.diag([filters.R_FLOOR_POS_M ** 2] * 3 + [filters.R_FLOOR_ATT_DEG ** 2] * 3)
    edges = [cfg.CALIB_RANGE_M[0]] + list(cfg.RANGE_BINS_M[1:-1]) + [np.inf]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ranges >= lo) & (ranges < hi)
        if m.sum() < CALIB_MIN_N:
            raise RuntimeError(f"range bin {lo}-{hi} m has only {m.sum()} samples at {noise_px} px")
        cov = errs[m].T @ errs[m] / m.sum() + floor
        bins.append({"range_lo_m": float(lo), "range_hi_m": None if np.isinf(hi) else float(hi), "n": int(m.sum()),
                     "centre_m": float(ranges[m].mean()), "std": np.sqrt(np.diag(cov)).tolist(),
                     "cov": cov.tolist()})
    return {"noise_px": noise_px, "n_poses": n, "n_no_solution": n_missing, "n_gross_failures": n_gross,
            "n_excluded_near_pitch_90": n_lock, "n_used": len(errs),
            "std_labels": ["x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"], "bins": bins}


# ---------------------------------------------------------------------------
# Kalman filter: filters.RangeKalmanFilter with PnP measurements given directly and a selectable dt
# ---------------------------------------------------------------------------
class UAVKalman(filters.RangeKalmanFilter):
    """
    The Phase 5b range-calibrated Kalman filter, unchanged apart from (1) taking the PnP measurement (6,) or
    None instead of an observation and (2) the time step dt, which enters F and Q only (A_MAX, W_MAX unchanged).
    """

    def __init__(self, range_bins, dt=cfg.DT, reinit_mode="full"):
        super().__init__(None, range_bins, reinit_mode=reinit_mode)
        self.F[0:3, 3:6] = dt * np.eye(3)
        self.F[6:9, 9:12] = dt * np.eye(3)
        q_block = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        self.Q = np.zeros((12, 12))
        for i in range(3):
            for base, a_max in [(0, config.A_MAX), (6, config.W_MAX)]:
                idx = [base + i, base + 3 + i]
                self.Q[np.ix_(idx, idx)] = q_block * (a_max ** 2 / 3.0)

    def initialise(self, z):
        self.n_gated = self.n_missing = self.n_reinit = self.consec_gated = 0
        self.x = self.P = None
        return self.step(z, first=True)

    def step(self, z, first=False):
        if self.x is None:                                   # still waiting for a first measurement
            if z is None:
                self.n_missing += 1
                return np.full(6, np.nan)
            self._set_from_measurement(z)
            return self._pose()
        self.predict()
        if z is None:
            self.n_missing += 1
        else:
            self.update(z)
        return self._pose()
