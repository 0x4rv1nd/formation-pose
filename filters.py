"""
Filters that track the leader's relative pose over time.

Phase 3: ParticleFilter, the BASELINE particle filter of the paper (Section IV-E).
Phase 5 will add an improved filter in this file. Any filter only has to provide
    initialise(observation) -> pose (6,)
    step(observation)       -> pose (6,)
so run_filter() works with both.

Particle state: [x, y, z, roll, pitch, yaw, vx, vy, vz]
(pose as in config.py, velocity = d(x, y, z)/dt in m/s).
"""

import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

import classifier
import config
import dataset
import geometry


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
# Running a filter on a trajectory
# ---------------------------------------------------------------------------
def run_filter(times, true_poses, filt, noise_px=0.0, seed=0):
    """
    Feed the filter the simulated observations of true_poses.
    Returns estimated poses (T, 6) and the runtime of each filter call in seconds (T,).
    """
    rng = np.random.default_rng(seed)
    est = np.empty((len(times), 6))
    runtime = np.empty(len(times))
    for i, pose in enumerate(true_poses):
        obs = make_observation(pose, noise_px, rng)
        t0 = time.perf_counter()
        est[i] = filt.initialise(obs) if i == 0 else filt.step(obs)
        runtime[i] = time.perf_counter() - t0
    return est, runtime
