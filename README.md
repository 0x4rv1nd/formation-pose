# formation-pose

A student reproduction of **"Vision-Based Precision Pose Estimation for Autonomous
Formation Flying"** (Punnoose, Stanford).

A follower aircraft carries a gimballed camera that always points at a leader aircraft.
The goal is to estimate the leader's relative pose (position + attitude) from the pixel
locations of 14 keypoints on the leader. Like the paper, we assume a perfect keypoint
detector and use purely synthetic data.

The project is built in 6 phases; this repository currently contains Phases 1 to 4.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Phase 1: simulation and dataset generation

| File | Purpose |
|------|---------|
| `config.py` | Conventions, camera intrinsics, keypoints, occlusion shapes, label grid |
| `geometry.py` | Pose -> camera transform, pinhole projection, occlusion (the only pose-to-pixel code) |
| `simulator.py` | Random pose sampling and the approach trajectory used in later phases |
| `dataset.py` | Features, labels, dataset generation CLI, sample plot |
| `test_phase1.py` | Sanity checks (prints PASS/FAIL) |

Run:

```bash
python test_phase1.py
python dataset.py                 # 100k train + 20k val -> data/*.npz
python dataset.py --noise_px 1.0  # same, with 1 px Gaussian keypoint noise
python dataset.py --gimbal_noise_deg 0.5  # same, with gimbal pointing error
```

This writes `data/train.npz` and `data/val.npz` (keys `X`, `y`, `poses`) and
`results/sample_keypoints.png`.

- **Pose**: `[x, y, z, roll, pitch, yaw]` - follower position relative to the leader
  (metres, leader body axes, x forward / y right / z down) and leader attitude relative
  to the follower (degrees, ZYX Euler). See the docstring in `config.py`.
- **Features**: 45 numbers. 42 keypoint features, `(u, v, visible)` per keypoint, with
  `u, v` normalised about the image centre (hidden keypoints are zeroed), plus 3 numbers
  for the gimbal direction: the unit vector from the follower to the leader in follower
  body axes (`geometry.boresight_unit`). The gimbal centres the leader in the image, so
  the pixel features alone lose the viewing direction, which a real gimbal knows. A 3D
  unit vector is used rather than azimuth/elevation because azimuth wraps at +/-180 deg
  when the leader is behind the follower. `config.GIMBAL_NOISE_DEG` (default 0) can
  perturb it for later phases.
- **Labels**: nearest point of the paper's Table I grid over x, y, z and roll
  (10 x 10 x 8 x 6 = 4800 classes).

![sample keypoints](results/sample_keypoints.png)

### Simplifications

- The 14 keypoints are an approximate F-16 layout, not measured from a real model.
- Occlusion uses a simple fuselage ellipsoid plus two flat wing triangles; the tail,
  canopy and stores do not occlude anything.
- Pitch and yaw are sampled (+/-10 deg) but are not part of the classification label.

## Phase 2: pose classifier

Implements the paper's classifier (Section IV-D): an MLP that maps the features to one of
the 4800 grid cells. Its output is the coarse pose used to start the particle filter in
Phase 3.

| File | Purpose |
|------|---------|
| `classifier.py` | Model, training, evaluation, figures, and the API below |
| `test_phase2.py` | Sanity checks (prints PASS/FAIL) |
| `models/classifier.pt` | Trained weights (tracked in git, about 2 MB) |
| `models/error_samples.npy` | (true pose - predicted bin pose) for every validation sample, shape (20000, 6); used by the particle filter in Phase 3 |

Run (needs `data/*.npz` from Phase 1):

```bash
python classifier.py
python test_phase2.py
```

Later phases use `load_model()`, `predict_probs(model, X)`, `predict_label(model, X)`
and `predict_pose(model, X)`.

**Architecture**: 45 -> 100 (ReLU) -> 100 (ReLU) -> 4800, softmax / cross-entropy.
Adam, lr 1e-3, batch size 256, up to 40 epochs on `data/train.npz`, best checkpoint by
validation accuracy, early stopping after 5 epochs without improvement. The model
standardises its input with the training-set mean and std (stored with the weights).

### Results (20,000 validation samples)

| Metric | Ours | Paper |
|--------|------|-------|
| Top-1 accuracy | 59.2 % | 54 % |
| Top-5 accuracy | 95.5 % | - |
| Neighbour accuracy (within one grid step in x, y, z, roll) | 99.9 % | - |
| Mean abs. error x / y / z | 4.59 / 3.16 / 1.63 m | - |
| Mean abs. error roll | 4.82 deg | - |

Errors are between the predicted bin's pose and the true pose. Raw numbers are in
`results/classifier_metrics.json`.

Predicted bin pose minus true pose: mostly discretisation error.

![error vs true pose](results/classifier_error_vs_true.png)

Predicted bin pose minus correct bin pose: a tall spike at zero plus small bars for
adjacent-bin mistakes.

![error vs correct bin](results/classifier_error_vs_label.png)

Training curve: `results/training_curve.png`.

### Design note: why the gimbal direction is an input

The first version used only the 42 keypoint features and reached just 16.0 % top-1. The
gimbal always centres the leader, so the pixels do not contain the viewing direction, and
the random +/-10 deg pitch and yaw look like a change of viewing direction. A real gimbal
knows where it points, so the classifier now also gets that direction (45 features).
Diagnostic runs with the same architecture and training recipe:

| Features | Pitch / yaw | Top-1 | Top-5 | Neighbour |
|----------|-------------|-------|-------|-----------|
| 42 (original) | +/-10 deg | 16.0 % | 52.2 % | 71.6 % |
| 42 | 0 | 60.2 % | 94.9 % | 98.5 % |
| **45 (used)** | +/-10 deg | 59.2 % | 95.5 % | 99.9 % |
| 45 | 0 | 65.1 % | 97.3 % | 100.0 % |

The 45-feature, +/-10 deg row is the final model. Pitch and yaw stay at +/-10 deg in the
actual design because the later phases must handle them.

## Phase 3: baseline particle filter

A faithful implementation of the paper's particle filter (Section IV-E), **without any
improvements** (those are Phases 4-5). It is meant as a fair reproduction, including the
paper's weaknesses.

| File | Purpose |
|------|---------|
| `filters.py` | `ParticleFilter`, observations, alpha schedules, `run_filter` |
| `geometry.py` | `project_points_batch`: vectorised projection of many particles |
| `run_phase3.py` | Experiments and figures |
| `test_phase3.py` | Sanity checks (prints PASS/FAIL) |
| `results/pf_baseline_metrics.json` | Numbers in the table below |

```bash
python test_phase3.py
python run_phase3.py     # about 6 s for everything
```

### Algorithm

Particle state `[x, y, z, roll, pitch, yaw, vx, vy, vz]` (pose as in `config.py`, relative velocity in m/s).

1. **Initialise**: the classifier's pose for the first observation plus random rows of
   `models/error_samples.npy` (its validation errors). Velocities uniform in +/-5 m/s.
2. **Propagate** every step (dt = 0.1 s): position += velocity * dt, velocity += uniform
   random acceleration in +/-`A_MAX` (2 m/s^2) * dt, roll/pitch/yaw += uniform random rate
   in +/-`W_MAX` (20 deg/s) * dt.
3. **Weights**: `w_i = 1 / (MSE_i + eps)`, where MSE_i is the mean squared pixel error between
   the observed visible keypoints and particle i's projection (`eps` = 1e-3 px^2, `PF_EPS`),
   then normalised. With fewer than 3 visible keypoints the weights are not updated.
4. **Estimate** (before resampling): weighted mean of position and velocity; weighted mean
   of attitude with `scipy Rotation.mean`.
5. **Resample**: `floor(alpha * N)` particles by systematic resampling; the other
   `N - floor(alpha * N)` are drawn fresh from the *current* classifier prediction plus
   random error samples, with the current weighted mean velocity. `alpha` follows a
   schedule (alpha = 1 is plain resampling).

### Our simplifications

- **Constant-velocity dynamics.** The paper treats the leader's unknown controls as a
  random disturbance but does not fully specify the dynamics, so this is our own
  constant-velocity approximation with uniform random accelerations and attitude rates.
- **No occlusion for particles** (as in the paper, it is too expensive): only keypoints that
  are visible in the actual observation are compared.
- **Measured boresight.** The gimbal points at the *true* leader and its direction is known.
  Particles are projected with that measured boresight, not with their own direction to
  their own leader; otherwise every particle would put the leader at the image centre and
  the weights could not tell them apart.
- `eps`, the velocity range and when the weights are skipped are our choices; the paper
  gives no values.

### Results

Approach trajectory, no keypoint noise, 5 seeds (mean +/- std over seeds of the
time-averaged error). The classifier-only estimate is deterministic. Attitude error is the
geodesic angle between the true and estimated rotations.

| Method | Position error [m] | Attitude error [deg] | Runtime [ms/step] |
|--------|-------------------|----------------------|-------------------|
| Classifier only | 7.05 | 5.03 | - |
| Config A (N = 1000, alpha = 0.9 every 10th step) | 10.27 +/- 2.67 | 8.73 +/- 0.69 | 1.8 |
| Config B (N = 5000, alpha = 0.9 every step) | 4.91 +/- 1.14 | 5.19 +/- 0.55 | 8.5 |

Errors over time (seed 0):

![errors](results/pf_baseline_errors.png)

True vs estimated position, Config B (seed 0):

![trajectory](results/pf_baseline_trajectory.png)

### Comparison with the paper

Like the paper, the baseline filter does **not** clearly beat the classifier. Config A is
*worse* than the classifier alone in both position and attitude: with only 1000 particles and
fresh classifier particles on just every 10th step, the cloud drifts away from the truth
between refreshes (the error grows, then drops when the particles are replaced). Config B
reduces the mean position error (4.9 m vs 7.1 m, with a large spread between seeds), but its
attitude error is no better than the classifier's, and it needs about 5x the particles.
Neither configuration gets near an accurate estimate, which is the weakness Phases 4-5
address. Only the trends should be compared with the paper: our keypoints, camera and
occlusion model differ from the paper's, and the dynamics are our approximation.

## Phase 4: PnP refinement

Three single-frame estimators, compared on the same noisy keypoints (the classifier is the
Phase 2 model, not retrained, and sees the noisy features):

1. **Classifier only**: the grid-cell pose from Phase 2.
2. **PnP only**: `cv2.solvePnP(..., flags=cv2.SOLVEPNP_SQPNP)`, no initial guess, no classifier.
3. **Classifier + PnP** (our improvement): the classifier's pose, converted to `rvec/tvec`, is
   the initial guess for `cv2.solvePnP(..., useExtrinsicGuess=True, flags=SOLVEPNP_ITERATIVE)`,
   followed by `cv2.solvePnPRefineLM`.

| File | Purpose |
|------|---------|
| `pnp.py` | `pnp_only(obs)`, `classifier_pnp(obs, coarse_pose)` |
| `geometry.py` | added `camera_to_pose` (inverse of `pose_to_camera`); `pose_to_camera` got an optional `boresight` argument |
| `run_phase4.py` | Experiments, tables, figures |
| `test_phase4.py` | Sanity checks (prints PASS/FAIL) |
| `results/pnp_metrics.json` | All numbers |

```bash
python test_phase4.py
python run_phase4.py
```

### Details

- Only keypoints that are visible in the observation are used, with `config.KEYPOINTS` and `config.K`.
- **Why at least 4 points**: each keypoint gives 2 equations (u, v) and a pose has 6 unknowns, so 3
  points are the mathematical minimum, but 3 points give up to 4 solutions. We require 4 (as
  OpenCV's iterative solver needs) and treat fewer as "no solution".
- **Camera frame to pose**: PnP returns the leader in the *camera* frame. `camera_to_pose` rotates
  it into the follower frame using the *measured* boresight (the gimbal direction), not the
  estimate. For the initial guess, `pose_to_camera(coarse_pose, boresight)` also uses the measured
  boresight, so the guess is consistent with how the image was formed.
- **Fallback**: if fewer than 4 keypoints are visible or the solver fails, `classifier_pnp` returns
  the coarse pose and sets `info["fallback"] = True`; `pnp_only` returns `None`.
- **Plausibility**: a solution is rejected if the leader is behind the camera (`t_cam` z <= 0) or the
  range is outside [10, 300] m (failure for PnP only, fallback for classifier + PnP).
- `info` holds the number of visible keypoints, the reprojection RMSE in px
  (sqrt of the mean squared 2D distance) and whether it fell back.
- **Metrics**: position error (m), attitude error (geodesic angle, deg), mean and median. A frame is
  a *failure* if there is no solution or position > 10 m or attitude > 20 deg. Runtime of
  classifier + PnP includes the classifier call. In the trajectory experiment, PnP-only frames
  with no solution would be left out of the error averages (and counted); there were none.
  Gross errors are kept in the means.

### Experiments

1. **Approach trajectory** (`simulator.approach_trajectory`, 101 steps), noise 0, 1, 2, 5 px, 5 seeds
   each, plus Phase 3 Config B (N = 5000, alpha = 0.9) as the reference. The observations are
   identical for all methods.
2. **Random poses**: 2000 poses per noise level (seed 12345; training uses 0, validation 1),
   split by number of visible keypoints.

### Results: approach trajectory

Mean +/- std over 5 seeds of the time-averaged error; "med" is the median over frames.

| Noise | Method | Pos mean [m] | Pos med [m] | Att mean [deg] | Att med [deg] | Fail % | ms/frame |
|------|--------|-------------|------------|---------------|--------------|--------|----------|
| 0 px | Classifier only | 7.05 | 7.52 | 5.03 | 4.96 | 2.0 | 0.05 |
| | PnP only | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.08 |
| | Classifier + PnP | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.21 |
| | Phase 3 Config B | 4.91 +/- 1.14 | 4.68 | 5.19 +/- 0.55 | 4.64 | 6.7 | 8.9 |
| 1 px | Classifier only | 7.19 +/- 0.05 | 7.64 | 5.14 +/- 0.07 | 4.96 | 6.3 | 0.05 |
| | PnP only | 0.44 +/- 0.02 | 0.32 | 0.75 +/- 0.04 | 0.72 | 0.0 | 0.08 |
| | Classifier + PnP | 0.43 +/- 0.02 | 0.31 | 0.74 +/- 0.04 | 0.72 | 0.0 | 0.22 |
| | Phase 3 Config B | 6.77 +/- 0.83 | 6.47 | 5.90 +/- 0.26 | 5.18 | 18.6 | 9.0 |
| 2 px | Classifier only | 7.26 +/- 0.11 | 7.49 | 5.24 +/- 0.12 | 4.98 | 9.1 | 0.05 |
| | PnP only | 0.89 +/- 0.05 | 0.64 | 1.50 +/- 0.08 | 1.45 | 0.0 | 0.09 |
| | Classifier + PnP | 0.87 +/- 0.04 | 0.63 | 1.49 +/- 0.08 | 1.44 | 0.0 | 0.23 |
| | Phase 3 Config B | 6.35 +/- 1.03 | 6.07 | 6.20 +/- 0.20 | 5.64 | 13.3 | 9.1 |
| 5 px | Classifier only | 8.08 +/- 0.23 | 7.86 | 5.77 +/- 0.19 | 5.19 | 22.0 | 0.05 |
| | PnP only | 2.76 +/- 0.14 | 1.94 | 4.35 +/- 0.63 | 3.66 | 1.2 | 0.09 |
| | Classifier + PnP | 2.17 +/- 0.10 | 1.60 | 3.72 +/- 0.20 | 3.56 | 0.8 | 0.22 |
| | Phase 3 Config B | 7.91 +/- 0.43 | 7.70 | 7.45 +/- 0.46 | 7.12 | 30.7 | 9.1 |

Errors over time at 2 px (seed 0):

![trajectory errors](results/pnp_trajectory_errors.png)

Error vs noise level:

![noise sweep](results/pnp_noise_sweep.png)

### Results: random poses (2000 per noise level)

| Noise | Visible | n | Method | Pos mean / med [m] | Att mean / med [deg] | Fail % | Fallback % |
|------|---------|---|--------|-------------------|---------------------|--------|-----------|
| 2 px | all | 2000 | Classifier only | 6.73 / 6.47 | 9.62 / 9.76 | 12.6 | - |
| | | | PnP only | 0.71 / 0.41 | 1.37 / 1.23 | 0.0 | - |
| | | | Classifier + PnP | 0.68 / 0.41 | 1.37 / 1.22 | 0.0 | 0.0 |
| | 7-9 | 399 | Classifier only | 6.50 / 6.31 | 9.61 / 9.78 | 9.5 | - |
| | | | PnP only | 0.51 / 0.30 | 1.47 / 1.37 | 0.0 | - |
| | | | Classifier + PnP | 0.50 / 0.30 | 1.47 / 1.35 | 0.0 | 0.0 |
| | 10-14 | 1601 | Classifier only | 6.78 / 6.54 | 9.62 / 9.75 | 13.3 | - |
| | | | PnP only | 0.76 / 0.45 | 1.35 / 1.18 | 0.0 | - |
| | | | Classifier + PnP | 0.73 / 0.44 | 1.34 / 1.17 | 0.0 | 0.0 |
| 5 px | all | 2000 | Classifier only | 7.65 / 6.99 | 9.95 / 9.98 | 20.5 | - |
| | | | PnP only | 2.17 / 1.14 | 3.60 / 3.06 | 2.4 | - |
| | | | Classifier + PnP | 1.90 / 0.99 | 3.38 / 2.98 | 1.1 | 0.1 |
| | 7-9 | 399 | Classifier only | 7.33 / 6.82 | 9.87 / 9.87 | 17.0 | - |
| | | | PnP only | 1.45 / 0.82 | 3.88 / 3.15 | 0.8 | - |
| | | | Classifier + PnP | 1.22 / 0.78 | 3.46 / 3.05 | 0.0 | 0.0 |
| | 10-14 | 1601 | Classifier only | 7.73 / 7.04 | 9.97 / 10.01 | 21.4 | - |
| | | | PnP only | 2.35 / 1.24 | 3.53 / 2.99 | 2.8 | - |
| | | | Classifier + PnP | 2.07 / 1.07 | 3.36 / 2.97 | 1.3 | 0.2 |

The 0 px and 1 px tables are in the script output and `results/pnp_metrics.json`.
**The requested 4-6 group is empty**: with our occlusion model no random pose (and no trajectory
frame) has fewer than 8 visible keypoints, so the benchmark says nothing about few-keypoint cases.

![by visibility](results/pnp_by_visibility.png)

### What this shows

- **Zero-noise PnP is near-exact (about 1e-10 m) because the keypoints are ideal.** That is expected
  and not a result; the noise sweep and visibility breakdown are the meaningful parts.
- **Both PnP methods are far better than the classifier alone and than the Phase 3 particle filter**
  (about 0.9 m / 1.5 deg vs 7 m / 5 deg at 2 px), at a fraction of the cost of the filter.
- **Does the classifier add anything over pure geometry?** Very little in these experiments.
  Up to 2 px, classifier + PnP and PnP only agree to within a few percent; sometimes PnP only is
  fractionally better (for example the median position error at 2 px over all random poses,
  0.4129 vs 0.4131 m), which is a tie. SQPnP finds the global solution without any guess, so a good
  starting point is not needed when 8 or more keypoints are visible.
- **At 5 px the classifier helps modestly**: on the trajectory, position 2.17 vs 2.76 m (-21 %),
  attitude 3.72 vs 4.35 deg (-14 %, with a much smaller spread across seeds), and gross failures
  0.8 % vs 1.2 %. On random poses, 1.90 vs 2.17 m, and 1.1 % vs 2.4 % failures. The coarse pose
  steers the iterative solver away from the occasional wrong PnP solution when the noise is large.
- **What is not tested here**: the few-keypoint regime (4-6 visible), where a global solver and a
  guess-based solver could differ most. The 4-6 column of the benchmark is empty, so we make no claim
  about it.
- The classifier still provides a coarse pose when PnP cannot run (fewer than 4 visible keypoints;
  this is what the fallback returns) and a prior for a filter, which is Phase 5.
