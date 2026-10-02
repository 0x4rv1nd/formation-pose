"""Sanity checks for Phase 5 (prints PASS/FAIL). Needs models/classifier.pt from `python classifier.py`."""

import sys

import numpy as np

import classifier
import config
import filters
import simulator

results = []


def check(name, ok):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


model = classifier.load_model()
times, true = simulator.approach_trajectory()

# Smaller calibration set than run_phase5 (same seed) to keep the test fast
R0 = np.array(filters.calibrate_pnp_noise(model, 0.0, n=200)["cov"])
R2 = np.array(filters.calibrate_pnp_noise(model, 2.0, n=200)["cov"])
check("calibrated R is symmetric positive definite",
      np.allclose(R2, R2.T) and np.linalg.eigvalsh(R2).min() > 0)

# --- Kalman filter, zero noise -----------------------------------------------
kf = filters.KalmanFilter(model, R0)
est, _ = filters.run_filter(times, true, kf, noise_px=0.0)
pos_err = np.linalg.norm(est[:, :3] - true[:, :3], axis=1)
check(f"KF with zero noise tracks the trajectory (mean position error {pos_err.mean():.4f} m < 0.1)",
      np.isfinite(est).all() and pos_err.mean() < 0.1)


# --- covariance stays symmetric positive definite -------------------------------
def covariance_ok(kf, noise_px, outage=None):
    """Run the filter step by step, checking P after every step."""
    rng = np.random.default_rng(1)
    ok = True
    for i, pose in enumerate(true):
        obs = filters.make_observation(pose, noise_px, rng)
        if outage and outage[0] <= times[i] < outage[1]:
            obs = filters.hide_all_keypoints(obs)
        kf.initialise(obs) if i == 0 else kf.step(obs)
        P = kf.P
        ok &= np.allclose(P, P.T, atol=1e-9 * np.abs(P).max()) and np.linalg.eigvalsh(P).min() > 0
    return bool(ok)


check("KF covariance symmetric and positive definite throughout (0 px)",
      covariance_ok(filters.KalmanFilter(model, R0), 0.0))
check("KF covariance symmetric and positive definite throughout (2 px, with outage)",
      covariance_ok(filters.KalmanFilter(model, R2), 2.0, outage=(4.0, 5.0)))

# --- outage: finite estimates every step ------------------------------------------
kf = filters.KalmanFilter(model, R2)
est, _ = filters.run_filter(times, true, kf, noise_px=2.0, outage=(4.0, 5.0))
check(f"KF returns finite estimates at every step during an outage ({kf.n_missing} missing steps predicted through)",
      np.isfinite(est).all() and kf.n_missing >= 10)
pf = filters.ImprovedParticleFilter(model, 300, filters.alpha_every_step(0.9), 2.0, R2, seed=0)
est_pf, _ = filters.run_filter(times, true, pf, noise_px=2.0, outage=(4.0, 5.0))
check("improved PF also returns finite estimates through the outage", np.isfinite(est_pf).all())

# --- gating -----------------------------------------------------------------------
kf = filters.KalmanFilter(model, R2)
filters.run_filter(times[:30], true[:30], kf, noise_px=2.0)
n_before = kf.n_gated                              # real measurements may have been gated already
kf.predict()
x_before = kf.x.copy()
bad = true[30].copy()
bad[:3] += 50.0                                    # corrupted measurement: true pose + 50 m
accepted = kf.update(bad)
check(f"measurement with +50 m error is gated out (d^2 = {kf.last_d2:.0f} > {filters.CHI2_999_6DOF:.1f}), "
      "state unchanged", (not accepted) and kf.n_gated == n_before + 1 and np.allclose(kf.x, x_before))
good = true[30].copy()
check("a good measurement is accepted", kf.update(good))

# --- improved particle filter weights --------------------------------------------------
obs = [filters.make_observation(p, 0.0) for p in true[:3]]
pf = filters.ImprovedParticleFilter(model, 500, filters.alpha_every_step(0.9), 0.0, R0, seed=0)
pf.initialise(obs[0])
pf.propagate()
pf.update_weights(obs[1])
w = pf.weights
check("improved PF weights are finite, non-negative and sum to 1 at 0 px noise (no underflow)",
      np.isfinite(w).all() and (w >= 0).all() and abs(w.sum() - 1) < 1e-9 and w.max() > 0)
# Even with particles that are hopelessly wrong (huge SSE) the log-space weights must not underflow
pf.particles[:, :3] += 80.0
pf.update_weights(obs[1])
check("weights stay valid even when every particle is far off",
      np.isfinite(pf.weights).all() and abs(pf.weights.sum() - 1) < 1e-9)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
