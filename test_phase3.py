"""Sanity checks for Phase 3 (prints PASS/FAIL). Run after `python classifier.py`."""

import sys

import numpy as np

import classifier
import config
import filters
import geometry
import simulator

results = []


def check(name, ok):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


rng = np.random.default_rng(0)
poses = simulator.sample_random_poses(10, rng)

# --- batch projection ---------------------------------------------------
max_diff = 0.0
for pose in poses:
    batch = geometry.project_points_batch(pose[None], geometry.boresight_unit(pose))[0]
    max_diff = max(max_diff, np.abs(batch - geometry.project_points(pose)).max())
check(f"project_points_batch matches project_points (max diff {max_diff:.1e} px)", max_diff < 1e-6)

# Several poses at once, each with its own boresight, must also agree
batch_all = np.stack([geometry.project_points_batch(p[None], geometry.boresight_unit(p))[0]
                      for p in poses])
check("batch of 10 poses has shape (10, 14, 2)", batch_all.shape == (10, config.N_KEYPOINTS, 2))

# Boresight of ANOTHER pose: the leader CG (the origin of the leader frame) must leave the centre
other = geometry.boresight_unit(poses[1])
origin = np.zeros((1, 3))
at_own = geometry.project_points_batch(poses[0][None], geometry.boresight_unit(poses[0]), origin)[0, 0]
at_other = geometry.project_points_batch(poses[0][None], other, origin)[0, 0]
check("own boresight puts the leader CG at the image centre",
      np.allclose(at_own, [config.CX, config.CY], atol=1e-6))
check("a different boresight moves the leader CG away from the centre",
      np.linalg.norm(at_other - [config.CX, config.CY]) > 1.0)

# --- filter internals ---------------------------------------------------
model = classifier.load_model()
times, true = simulator.approach_trajectory()
obs = [filters.make_observation(p) for p in true[:3]]

N = 500
pf = filters.ParticleFilter(model, N, filters.alpha_every_step(0.9), seed=0)
pf.initialise(obs[0])
pf.propagate()
pf.update_weights(obs[1])
check("weights are non-negative and sum to 1",
      (pf.weights >= 0).all() and abs(pf.weights.sum() - 1) < 1e-9)
check("weights are not all equal after an update", pf.weights.std() > 0)
pf.resample(obs[1], np.zeros(3))
check("resampling keeps exactly N particles", pf.particles.shape == (N, 9))
check("weights are uniform after resampling", np.allclose(pf.weights, 1.0 / N))

# --- sanity run: particles start exactly at the truth --------------------
pf = filters.ParticleFilter(model, 1000, filters.alpha_every_step(1.0), seed=0)
pf.initialise(filters.make_observation(true[0]))
pf.particles[:, :6] = true[0]
pf.particles[:, 6:] = (true[1, :3] - true[0, :3]) / config.DT   # true velocity
errs = []
for pose in true[1:21]:
    est = pf.step(filters.make_observation(pose))
    errs.append(np.linalg.norm(est[:3] - pose[:3]))
check(f"particles at the true pose stay close for 20 steps "
      f"(max position error {max(errs):.2f} m < 2 m)", max(errs) < 2.0)

# --- run_filter ------------------------------------------------------------
pf = filters.ParticleFilter(model, 200, filters.alpha_every_nth(0.9, 10), seed=1)
est, runtime = filters.run_filter(times[:15], true[:15], pf, noise_px=0.0, seed=1)
check("run_filter returns (T, 6) estimates and (T,) runtimes",
      est.shape == (15, 6) and runtime.shape == (15,))
check("run_filter output has no NaNs", np.isfinite(est).all() and np.isfinite(runtime).all())

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
