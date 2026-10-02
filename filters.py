"""
Filters that track the leader's relative pose over time.

Phase 3: ParticleFilter, the BASELINE particle filter of the paper (Section IV-E).
Phase 5 adds KalmanFilter and ImprovedParticleFilter (both work on PnP measurements). Any filter only has to provide
    initialise(observation) -> pose (6,)
    step(observation)       -> pose (6,)
so run_filter() works with both.

Particle state: [x, y, z, roll, pitch, yaw, vx, vy, vz]
(pose as in config.py, velocity = d(x, y, z)/dt in m/s).
"""

import time
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

import classifier
import config
import dataset
import geometry
import pnp
import simulator


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
@dataclass
class Observation:
    uv: np.ndarray          # (14, 2) keypoint pixels
    visible: np.ndarray     # (14,) bool
    boresight: np.ndarray   # (3,) gimbal direction, follower body axes
    features: np.ndarray    # (45,) classifier input


def make_observation(pose, noise_px=0.0, rng=None):
    """What the follower sees when the leader is at `pose` (simulated detector + gimbal)."""
    uv, vis = geometry.observe(pose, noise_px, rng)
    boresight = geometry.boresight_unit(pose)
    return Observation(uv, vis, boresight, dataset.make_features(uv, vis, boresight))


def hide_all_keypoints(obs):
    """Copy of obs in which no keypoint is visible (measurement outage); features rebuilt."""
    vis = np.zeros_like(obs.visible)
    return replace(obs, visible=vis, features=dataset.make_features(obs.uv, vis, obs.boresight))


# ---------------------------------------------------------------------------
# Alpha schedules: step index k (1, 2, ...) -> fraction of particles that are resampled
# ---------------------------------------------------------------------------
def alpha_every_step(alpha):
    return lambda k: alpha


def alpha_every_nth(alpha, n):
    """alpha on every n-th step, plain resampling (alpha = 1) otherwise."""
    return lambda k: alpha if k % n == 0 else 1.0


# ---------------------------------------------------------------------------
# Baseline particle filter
# ---------------------------------------------------------------------------
class ParticleFilter:
    def __init__(self, model, n_particles, alpha_schedule, seed=0):
        self.model = model
        self.n = n_particles
        self.alpha_schedule = alpha_schedule
        self.rng = np.random.default_rng(seed)
        # (true pose - classifier pose) for validation samples: the classifier's error distribution
        self.error_samples = np.load(classifier.ERROR_SAMPLES_PATH)
        self.particles = np.zeros((n_particles, 9))
        self.weights = np.full(n_particles, 1.0 / n_particles)
        self.k = 0

    # -- helpers ----------------------------------------------------------
    def _classifier_pose(self, obs):
        return classifier.predict_pose(self.model, obs.features[None])[0]

    def _draw_from_classifier(self, obs, m):
        """m particle poses = classifier pose + random classifier error samples (paper step 1)."""
        base = self._classifier_pose(obs)
        rows = self.error_samples[self.rng.integers(len(self.error_samples), size=m)]
        return base + rows

    def _estimate(self):
        """Weighted mean state -> pose (6,) and velocity (3,)."""
        w = self.weights
        pos_vel = w @ self.particles[:, [0, 1, 2, 6, 7, 8]]
        rots = Rotation.from_euler("ZYX", self.particles[:, [5, 4, 3]], degrees=True)
        yaw, pitch, roll = rots.mean(weights=w).as_euler("ZYX", degrees=True)
        return np.array([*pos_vel[:3], roll, pitch, yaw]), pos_vel[3:]

    # -- paper step 1 -----------------------------------------------------
    def initialise(self, obs):
        self.k = 0
        self.particles[:, :6] = self._draw_from_classifier(obs, self.n)
        self.particles[:, 6:] = self.rng.uniform(
            -config.V_INIT_MAX, config.V_INIT_MAX, size=(self.n, 3))
        self.weights = np.full(self.n, 1.0 / self.n)
        return self._estimate()[0]

    # -- one filter step --------------------------------------------------
    def propagate(self):
        """Constant-velocity model with uniformly random acceleration and attitude rate."""
        dt = config.DT
        p = self.particles
        p[:, :3] += p[:, 6:] * dt
        p[:, 6:] += self.rng.uniform(-config.A_MAX, config.A_MAX, size=(self.n, 3)) * dt
        p[:, 3:6] += self.rng.uniform(-config.W_MAX, config.W_MAX, size=(self.n, 3)) * dt

    def update_weights(self, obs):
        """w_i = 1 / (MSE_i + eps) over the keypoints visible in the observation."""
        vis = obs.visible
        if vis.sum() < config.PF_MIN_VISIBLE:
            return  # too few keypoints: keep the weights as they are
        # Particles do not compute occlusion; we simply compare the observed visible ones.
        proj = geometry.project_points_batch(self.particles[:, :6], obs.boresight)
        err = proj[:, vis, :] - obs.uv[vis][None]              # (N, n_vis, 2)
        mse = np.mean(err ** 2, axis=(1, 2))
        w = 1.0 / (mse + config.PF_EPS)
        self.weights = w / w.sum()

    def resample(self, obs, mean_velocity):
        """floor(alpha*N) by systematic resampling, the rest fresh from the classifier."""
        alpha = self.alpha_schedule(self.k)
        n_keep = int(np.floor(alpha * self.n + 1e-9))
        n_fresh = self.n - n_keep

        # Systematic resampling: one random offset, N evenly spaced pointers
        pointers = (self.rng.random() + np.arange(n_keep)) / max(n_keep, 1)
        idx = np.searchsorted(np.cumsum(self.weights), pointers)
        idx = np.minimum(idx, self.n - 1)
        new = self.particles[idx]

        if n_fresh > 0:
            fresh = np.empty((n_fresh, 9))
            fresh[:, :6] = self._draw_from_classifier(obs, n_fresh)
            fresh[:, 6:] = mean_velocity
            new = np.vstack([new, fresh])

        self.particles = new
        self.weights = np.full(self.n, 1.0 / self.n)

    def step(self, obs):
        self.k += 1
        self.propagate()
        self.update_weights(obs)
        pose, mean_velocity = self._estimate()   # estimate BEFORE resampling
        self.resample(obs, mean_velocity)
        return pose


# ---------------------------------------------------------------------------
# Phase 5: PnP measurement and its calibrated noise
# ---------------------------------------------------------------------------
REF_RANGE_M = 65.0           # reference range for R (about the middle of the approach, 90 -> 40 m)
CALIB_SEED = 424242          # calibration poses and noise: not used for training, validation or experiments
CALIB_N = 1000
R_FLOOR_POS_M = 0.01         # floor on the measurement std (m, deg): keeps R invertible at 0 px noise
R_FLOOR_ATT_DEG = 0.01
CHI2_999_6DOF = 22.4577      # chi-squared 99.9 % quantile, 6 degrees of freedom (scipy.stats.chi2.ppf(0.999, 6))


def pnp_measurement(model, obs):
    """
    The "PnP measurement": hybrid_pnp started from the classifier's coarse pose.
    Returns a pose (6,), or None when there is no real PnP solution (fewer than 4 visible
    keypoints, or both solvers failed and hybrid_pnp fell back to the coarse pose). A coarse
    classifier pose is not a measurement we want to trust, so it counts as missing.
    """
    if obs.visible.sum() < pnp.MIN_POINTS:
        return None
    coarse = classifier.predict_pose(model, obs.features[None])[0]
    pose, info = pnp.hybrid_pnp(obs, coarse)
    return None if info["fallback"] else pose


def wrap_deg(a):
    """Wrap angle(s) in degrees to [-180, 180)."""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def pose_error_vector(est, true):
    """est - true as a 6-vector, with the three angle differences wrapped."""
    e = np.asarray(est, dtype=float) - np.asarray(true, dtype=float)
    e[..., 3:] = wrap_deg(e[..., 3:])
    return e


def scaled_R(R_ref, position):
    """
    Measurement covariance at the range of `position`. PnP position error grows with
    distance, so the position variances scale with (range / REF_RANGE_M)^2:
    R = D R_ref D with D = diag(s, s, s, 1, 1, 1), s = range / REF_RANGE_M.
    """
    s = np.linalg.norm(position) / REF_RANGE_M
    D = np.diag([s, s, s, 1.0, 1.0, 1.0])
    return D @ R_ref @ D


def calibrate_pnp_noise(model, noise_px, n=CALIB_N, seed=CALIB_SEED):
    """
    Measure the PnP error covariance on n random poses at the given keypoint noise.
    Position errors are divided by (range / REF_RANGE_M) first so that one covariance
    at the reference range describes all ranges (see scaled_R). Frames with no PnP
    solution or a gross failure (position error > 10 m or attitude error > 20 deg, the
    Phase 4 definition) are left out: the gate rejects those at run time, and including
    them would inflate R for every normal frame.
    Returns a dict with the 6x6 covariance (list), per-axis std and the counts.
    """
    rng = np.random.default_rng(seed)
    poses = simulator.sample_random_poses(n, rng)
    errs, n_missing, n_gross = [], 0, 0
    for pose in poses:
        obs = make_observation(pose, noise_px, rng)
        z = pnp_measurement(model, obs)
        if z is None:
            n_missing += 1
            continue
        e = pose_error_vector(z, pose)
        if np.linalg.norm(e[:3]) > 10.0 or np.linalg.norm(e[3:]) > 20.0:
            n_gross += 1
            continue
        e[:3] /= np.linalg.norm(pose[:3]) / REF_RANGE_M
        errs.append(e)
    errs = np.array(errs)
    cov = errs.T @ errs / len(errs)          # covariance about zero (the PnP error is ~unbiased)
    cov = cov + np.diag([R_FLOOR_POS_M ** 2] * 3 + [R_FLOOR_ATT_DEG ** 2] * 3)
    return {"noise_px": noise_px, "ref_range_m": REF_RANGE_M, "n_poses": n,
            "n_no_solution": n_missing, "n_gross_failures": n_gross, "n_used": len(errs),
            "std": np.sqrt(np.diag(cov)).tolist(),
            "std_labels": ["x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"],
            "cov": cov.tolist()}


# Phase 5b (fixed in advance, not tuned)
RANGE_BIN_EDGES = [25.0, 50.0, 75.0, 100.0, np.inf]   # m; poses never get beyond ~113 m, so no 150+ bin
RANGE_CALIB_N = 2000         # more poses than REF-range calibration so that every bin has enough samples
RANGE_BIN_MIN_N = 30
REINIT_AFTER = 3             # consecutive gated measurements before the attitude is re-initialised


def calibrate_pnp_noise_by_range(model, noise_px, n=RANGE_CALIB_N, seed=CALIB_SEED):
    """
    PnP error covariance per range bin, measured on random poses (calibration set only).
    Same exclusions and floor as calibrate_pnp_noise, but the position errors are NOT rescaled:
    each bin holds the raw 6x6 covariance of its own poses. The bin centre is the mean range of
    the samples in the bin (used for interpolation).
    """
    rng = np.random.default_rng(seed)
    poses = simulator.sample_random_poses(n, rng)
    ranges, errs, n_missing, n_gross = [], [], 0, 0
    for pose in poses:
        obs = make_observation(pose, noise_px, rng)
        z = pnp_measurement(model, obs)
        if z is None:
            n_missing += 1
            continue
        e = pose_error_vector(z, pose)
        if np.linalg.norm(e[:3]) > 10.0 or np.linalg.norm(e[3:]) > 20.0:
            n_gross += 1
            continue
        ranges.append(np.linalg.norm(pose[:3]))
        errs.append(e)
    ranges, errs = np.array(ranges), np.array(errs)
    floor = np.diag([R_FLOOR_POS_M ** 2] * 3 + [R_FLOOR_ATT_DEG ** 2] * 3)
    bins = []
    for lo, hi in zip(RANGE_BIN_EDGES[:-1], RANGE_BIN_EDGES[1:]):
        m = (ranges >= lo) & (ranges < hi)
        if m.sum() < RANGE_BIN_MIN_N:
            raise RuntimeError(f"range bin {lo}-{hi} m has only {m.sum()} samples at {noise_px} px")
        cov = errs[m].T @ errs[m] / m.sum() + floor
        bins.append({"range_lo_m": lo, "range_hi_m": None if np.isinf(hi) else hi,
                     "n": int(m.sum()), "centre_m": float(ranges[m].mean()),
                     "std": np.sqrt(np.diag(cov)).tolist(), "cov": cov.tolist()})
    return {"noise_px": noise_px, "n_poses": n, "n_no_solution": n_missing,
            "n_gross_failures": n_gross, "n_used": len(errs),
            "std_labels": ["x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"], "bins": bins}


def range_R(bins, r):
    """Covariance at range r: linear interpolation between the bin covariances at the bin
    centres (a convex combination of positive definite matrices), clamped at the ends."""
    centres = np.array([b["centre_m"] for b in bins])
    covs = np.array([b["cov"] for b in bins])
    if r <= centres[0]:
        return covs[0]
    if r >= centres[-1]:
        return covs[-1]
    j = np.searchsorted(centres, r) - 1
    w = (r - centres[j]) / (centres[j + 1] - centres[j])
    return (1 - w) * covs[j] + w * covs[j + 1]


# ---------------------------------------------------------------------------
# Single-frame PnP as a "filter" (so run_filter can run it)
# ---------------------------------------------------------------------------
class PnPOnly:
    """No filtering: the PnP measurement of the current frame, NaN when there is none."""

    def __init__(self, model):
        self.model = model

    def initialise(self, obs):
        return self.step(obs)

    def step(self, obs):
        z = pnp_measurement(self.model, obs)
        return np.full(6, np.nan) if z is None else z


# ---------------------------------------------------------------------------
# Kalman filter on PnP measurements
# ---------------------------------------------------------------------------
class KalmanFilter:
    """
    Linear Kalman filter. State (12): [x, y, z, vx, vy, vz, roll, pitch, yaw, rr, pr, yr]
    (positions in m, angles in deg, rates per second). Constant-velocity model for both.
    Measurement: the 6-number PnP pose (H picks the position and angle entries).

    Process noise (discrete white-noise-acceleration model). For one axis with state
    [p, v] and an unknown acceleration a of std sigma_a that is constant within a step:
        F = [[1, dt], [0, 1]]
        Q = sigma_a^2 * [[dt^4/4, dt^3/2],
                         [dt^3/2, dt^2  ]]
    The Phase 3 particle filter adds a uniform random acceleration in +/-A_MAX, whose
    standard deviation is A_MAX / sqrt(3), so sigma_a = A_MAX / sqrt(3) for the three
    position axes. For the angles we treat W_MAX the same way (W_MAX deg/s of rate change
    per second): sigma_a = W_MAX / sqrt(3) deg/s^2. The same A_MAX and W_MAX as the
    particle filter are used; nothing is tuned.

    Measurement noise R: calibrated (calibrate_pnp_noise) and scaled with range (scaled_R).
    Gating: an update is skipped when the squared Mahalanobis distance of the innovation
    exceeds the chi-squared 99.9 % threshold (6 dof). Skipped updates are counted in n_gated.
    With no PnP measurement the filter only predicts (counted in n_missing).
    """

    def __init__(self, model, R_ref):
        self.model = model
        self.R_ref = np.asarray(R_ref, dtype=float)
        dt = config.DT

        self.F = np.eye(12)
        self.F[0:3, 3:6] = dt * np.eye(3)
        self.F[6:9, 9:12] = dt * np.eye(3)
        self.H = np.zeros((6, 12))
        self.H[0:3, 0:3] = np.eye(3)
        self.H[3:6, 6:9] = np.eye(3)

        # Q: the 2x2 white-noise-acceleration block per axis, placed at its (p, v) indices
        q_block = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        self.Q = np.zeros((12, 12))
        for i in range(3):
            for base, a_max in [(0, config.A_MAX), (6, config.W_MAX)]:
                idx = [base + i, base + 3 + i]
                self.Q[np.ix_(idx, idx)] = q_block * (a_max ** 2 / 3.0)

        self.x = None            # state; None until the first PnP measurement
        self.P = None
        self.n_gated = 0
        self.n_missing = 0
        self.last_d2 = np.nan    # squared Mahalanobis distance of the last innovation

    # -- helpers ------------------------------------------------------------
    def _R(self, z_pos, est_pos):
        """Measurement covariance for a measurement at z_pos (est_pos: current estimated position)."""
        return scaled_R(self.R_ref, z_pos)

    def _pose(self):
        return self.x[[0, 1, 2, 6, 7, 8]].copy()

    def _set_from_measurement(self, z):
        """Start from z with zero velocities and a large velocity uncertainty."""
        pose_idx = [0, 1, 2, 6, 7, 8]
        self.x = np.zeros(12)
        self.x[pose_idx] = z
        self.P = np.zeros((12, 12))
        self.P[np.ix_(pose_idx, pose_idx)] = self._R(z[:3], z[:3])
        self.P[3:6, 3:6] = config.V_INIT_MAX ** 2 * np.eye(3)    # +/- 5 m/s, as the particle filter
        self.P[9:12, 9:12] = config.W_MAX ** 2 * np.eye(3)       # +/- 20 deg/s

    # -- filter steps ---------------------------------------------------------
    def initialise(self, obs):
        self.n_gated = self.n_missing = 0
        self.x = self.P = None
        z = pnp_measurement(self.model, obs)
        if z is None:
            self.n_missing += 1
            return np.full(6, np.nan)
        self._set_from_measurement(z)
        return self._pose()

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """Measurement update with the pose z (6,). Returns True if used, False if gated out."""
        R = self._R(z[:3], self.x[:3])
        nu = pose_error_vector(z, self.H @ self.x)           # innovation, angles wrapped
        S = self.H @ self.P @ self.H.T + R
        self.last_d2 = float(nu @ np.linalg.solve(S, nu))
        if self.last_d2 > CHI2_999_6DOF:
            self.n_gated += 1
            return False
        K = np.linalg.solve(S, self.H @ self.P).T            # P H^T S^-1 (S and P are symmetric)
        self.x = self.x + K @ nu
        # Joseph form keeps P symmetric and positive definite
        A = np.eye(12) - K @ self.H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def step(self, obs):
        z = pnp_measurement(self.model, obs)
        if self.x is None:                                   # still waiting for a first measurement
            if z is None:
                self.n_missing += 1
                return np.full(6, np.nan)
            self._set_from_measurement(z)
            return self._pose()
        self.predict()
        if z is None:
            self.n_missing += 1                              # predict only
        else:
            self.update(z)
        return self._pose()


class RangeKalmanFilter(KalmanFilter):
    """
    Phase 5b Kalman filter: the Phase 5 filter with two changes and nothing else.
      1. R comes from the range-binned calibration (calibrate_pnp_noise_by_range), interpolated
         at the estimated range (the filter's predicted position at update time) instead of the
         Phase 5 rule "one covariance, position variances scaled with range^2".
      2. After REINIT_AFTER consecutive gated measurements the filter is re-initialised from the
         current PnP measurement. Steps without a measurement do not change the count. n_reinit counts
         the events. reinit_mode selects what is reset:
           "attitude" (Phase 5b): attitude = measurement, rates 0 with std W_MAX, attitude covariance = R;
                      position and velocity keep their estimates.
           "full" (Phase 7a-fix, default): the whole state as at the first measurement (position and attitude =
                      measurement with covariance R, velocities and rates 0 with std V_INIT_MAX / W_MAX).
    """

    def __init__(self, model, range_bins, reinit_after=REINIT_AFTER, reinit_mode="full"):
        if reinit_mode not in ("attitude", "full"):
            raise ValueError(f"unknown reinit_mode {reinit_mode!r}")
        mid = np.array(range_bins[len(range_bins) // 2]["cov"])
        super().__init__(model, mid)
        self.range_bins = range_bins
        self.reinit_mode = reinit_mode
        self.reinit_after = reinit_after
        self.n_reinit = 0
        self.consec_gated = 0

    def _R(self, z_pos, est_pos):
        return range_R(self.range_bins, np.linalg.norm(est_pos))

    def initialise(self, obs):
        self.n_reinit = 0
        self.consec_gated = 0
        return super().initialise(obs)

    def update(self, z):
        used = super().update(z)
        if used:
            self.consec_gated = 0
        else:
            self.consec_gated += 1
            if self.consec_gated >= self.reinit_after:
                if self.reinit_mode == "full":
                    self._set_from_measurement(z)
                else:
                    self._reinit_attitude(z)
                self.n_reinit += 1
                self.consec_gated = 0
        return used

    def _reinit_attitude(self, z):
        R = self._R(z[:3], self.x[:3])
        self.x[6:9] = z[3:]
        self.x[9:12] = 0.0
        self.P[6:12, :] = 0.0
        self.P[:, 6:12] = 0.0
        self.P[6:9, 6:9] = R[3:, 3:]
        self.P[9:12, 9:12] = config.W_MAX ** 2 * np.eye(3)


# ---------------------------------------------------------------------------
# Improved particle filter
# ---------------------------------------------------------------------------
class ImprovedParticleFilter(ParticleFilter):
    """
    Phase 3 particle filter with exactly two changes (dynamics, measured boresight and
    systematic resampling are inherited unchanged):
      1. Gaussian likelihood  w_i ~ exp(-SSE_i / (2 sigma^2)), sigma = max(noise_px, 1) px,
         SSE = squared pixel error summed over the visible keypoints (computed in log space).
      2. Initial and fresh particles are drawn around the current PnP pose using the
         calibrated PnP error covariance (scaled to the range), not the classifier bin.
    If there is no PnP measurement (outage), the "fresh" particles are copied from the
    current particle cloud (plain resampling), since there is nothing new to draw around.
    """

    def __init__(self, model, n_particles, alpha_schedule, noise_px, R_ref, seed=0):
        super().__init__(model, n_particles, alpha_schedule, seed)
        self.sigma = max(noise_px, 1.0)
        self.R_ref = np.asarray(R_ref, dtype=float)
        self.pnp_pose = None

    def _draw_from_classifier(self, obs, m):
        """Replaces the classifier draw: m poses ~ N(PnP pose, R)."""
        if self.pnp_pose is None:
            if self.k == 0:                                  # no PnP at the start: use the classifier
                return super()._draw_from_classifier(obs, m)
            idx = self.rng.choice(self.n, size=m, p=self.weights)
            return self.particles[idx, :6]
        R = scaled_R(self.R_ref, self.pnp_pose[:3])
        return self.pnp_pose + self.rng.multivariate_normal(np.zeros(6), R, size=m)

    def update_weights(self, obs):
        vis = obs.visible
        if vis.sum() < config.PF_MIN_VISIBLE:
            return
        proj = geometry.project_points_batch(self.particles[:, :6], obs.boresight)
        sse = np.sum((proj[:, vis, :] - obs.uv[vis][None]) ** 2, axis=(1, 2))   # (N,)
        log_w = -sse / (2.0 * self.sigma ** 2)
        w = np.exp(log_w - log_w.max())                      # subtract the max: no underflow
        self.weights = w / w.sum()

    def initialise(self, obs):
        self.pnp_pose = pnp_measurement(self.model, obs)
        return super().initialise(obs)

    def step(self, obs):
        self.pnp_pose = pnp_measurement(self.model, obs)
        return super().step(obs)


# ---------------------------------------------------------------------------
# Running a filter on a trajectory
# ---------------------------------------------------------------------------
def run_filter(times, true_poses, filt, noise_px=0.0, seed=0, outage=None):
    """
    Feed the filter the simulated observations of true_poses.
    outage=(t_start, t_end): for t_start <= t < t_end every keypoint is hidden (Phase 5).
    Returns estimated poses (T, 6) and the runtime of each filter call in seconds (T,).
    """
    rng = np.random.default_rng(seed)
    est = np.empty((len(times), 6))
    runtime = np.empty(len(times))
    for i, pose in enumerate(true_poses):
        obs = make_observation(pose, noise_px, rng)   # same random numbers with or without outage
        if outage is not None and outage[0] - 1e-9 <= times[i] < outage[1] - 1e-9:
            obs = hide_all_keypoints(obs)
        t0 = time.perf_counter()
        est[i] = filt.initialise(obs) if i == 0 else filt.step(obs)
        runtime[i] = time.perf_counter() - t0
    return est, runtime
