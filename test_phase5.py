"""Sanity checks for Phase 5 (prints PASS/FAIL). Needs models/classifier.pt from `python classifier.py`."""

import sys

import numpy as np

import classifier
import config
import filters
import run_phase5
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

# --- Phase 5b: range-calibrated R and re-initialisation (2 px, 5 seeds, same settings as run_phase5b.py) ---
bins = filters.calibrate_pnp_noise_by_range(model, 2.0)["bins"]
check("range-binned R is symmetric positive definite in every bin and interpolates between bins",
      all(np.allclose(b["cov"], np.array(b["cov"]).T) and np.linalg.eigvalsh(b["cov"]).min() > 0 for b in bins)
      and np.allclose(filters.range_R(bins, 1e3), bins[-1]["cov"])
      and np.allclose(filters.range_R(bins, 0.5 * (bins[0]["centre_m"] + bins[1]["centre_m"])),
                      0.5 * (np.array(bins[0]["cov"]) + np.array(bins[1]["cov"]))))
check("position std in the calibration grows with range", bins[-1]["std"][0] > bins[0]["std"][0])

def run_tracking_reinit(seed, outage):
    """Like filters.run_filter, but also records the step index of each re-initialisation."""
    kf = filters.RangeKalmanFilter(model, bins, reinit_mode="attitude")
    rng = np.random.default_rng(seed)
    est, reinit_steps = np.empty((len(times), 6)), []
    for i, pose in enumerate(true):
        obs = filters.make_observation(pose, 2.0, rng)
        if outage and outage[0] - 1e-9 <= times[i] < outage[1] - 1e-9:
            obs = filters.hide_all_keypoints(obs)
        n = kf.n_reinit
        est[i] = kf.initialise(obs) if i == 0 else kf.step(obs)
        if kf.n_reinit > n:
            reinit_steps.append(i)
    return kf, est, reinit_steps


i_back = int(np.argmax(times >= 5.0 - 1e-9))        # first step with measurements after the outage
normal_rate, delays, att_rec, att_rec_normal = [], [], [], []
for seed in range(5):
    kf, est, _ = run_tracking_reinit(seed, None)
    normal_rate.append(100 * kf.n_gated / (len(times) - 1 - kf.n_missing))
    normal_att = run_phase5.errors(true, est)[1]
    kf, est, steps = run_tracking_reinit(seed, (4.0, 5.0))
    steps = [i for i in steps if i >= i_back]
    delays.append(steps[0] - i_back + 1 if steps else None)    # 1 = at the first measurement back
    if steps:
        w = np.arange(len(times))
        w = (w >= steps[0]) & (times < 6.0 - 1e-9)
        att_rec.append(run_phase5.errors(true, est)[1][w].mean())
        att_rec_normal.append(normal_att[w].mean())
rate = np.mean(normal_rate)
check(f"fixed KF normal gate rejection rate {rate:.2f} % < 1 % (nominal 0.1 %)", rate < 1.0)
check(f"(a) fixed KF re-initialises within {filters.REINIT_AFTER} steps after measurements return "
      f"(steps taken per seed: {delays})",
      all(d is not None and d <= filters.REINIT_AFTER for d in delays))
check(f"(b) from the re-initialisation to t = 6 s the attitude error ({np.mean(att_rec):.2f} deg) is within 2x "
      f"its normal level over the same steps ({np.mean(att_rec_normal):.2f} deg)",
      len(att_rec) == 5 and np.mean(att_rec) <= 2 * np.mean(att_rec_normal))

# --- Phase 7a-fix: full-state re-initialisation vs the Phase 5b attitude-only one ---------------------------
z0 = np.array([-60.0, 20.0, 10.0, 5.0, 0.0, 0.0])
z_far = z0 + np.array([200.0, 0.0, 0.0, 30.0, 0.0, 0.0])      # a measurement the filter cannot explain
state = {}
for mode in ("attitude", "full"):
    kf = filters.RangeKalmanFilter(model, bins, reinit_mode=mode)
    kf._set_from_measurement(z0)
    for _ in range(filters.REINIT_AFTER):
        kf.predict()
        kf.update(z_far)
    state[mode] = kf
full, att = state["full"], state["attitude"]
check("full re-init after 3 gated measurements: position and attitude = measurement, velocities and rates reset to 0",
      full.n_reinit == 1 and np.allclose(full.x[[0, 1, 2, 6, 7, 8]], z_far) and np.allclose(full.x[[3, 4, 5, 9, 10, 11]], 0.0)
      and np.allclose(full.P[3:6, 3:6], config.V_INIT_MAX ** 2 * np.eye(3)))
check("attitude-only re-init (Phase 5b) still available: attitude reset, position kept",
      att.n_reinit == 1 and np.allclose(att.x[6:9], z_far[3:]) and not np.allclose(att.x[:3], z_far[:3], atol=1.0))
check("both re-initialised covariances stay symmetric positive definite",
      all(np.allclose(k.P, k.P.T) and np.linalg.eigvalsh(k.P).min() > 0 for k in (full, att)))

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
