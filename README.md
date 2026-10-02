# formation-pose

A student reproduction of **"Vision-Based Precision Pose Estimation for Autonomous
Formation Flying"** (Punnoose, Stanford).

A follower aircraft carries a gimballed camera that always points at a leader aircraft.
The goal is to estimate the leader's relative pose (position + attitude) from the pixel
locations of 14 keypoints on the leader. Like the paper, we assume a perfect keypoint
detector and use purely synthetic data.

The project is built in 6 phases; this repository currently contains Phases 1 to 4b.

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

## Phase 4b: hybrid PnP and keypoint dropout

Phase 4 could not test few-keypoint cases (no random pose has fewer than 8 visible keypoints),
and a quick check suggested the classifier hurt there. This phase tests it properly.

| File | Purpose |
|------|---------|
| `pnp.py` | added `hybrid_pnp(obs, coarse_pose)`; `classifier_pnp` is unchanged |
| `train_dropout.py` | trains `models/classifier_dropout.pt` (same architecture and settings; 0-6 extra visible keypoints hidden per sample, re-drawn every batch). `models/classifier.pt` is untouched |
| `run_phase4b.py` | dropout sweep, figure and metrics |
| `results/pnp_dropout.png`, `pnp_dropout_metrics.json`, `classifier_dropout_metrics.json` | outputs |

```bash
python train_dropout.py
python run_phase4b.py
```

- **Hybrid**: runs SQPnP (no guess) and classifier-initialised PnP, keeps the plausible solution with
  the lower reprojection RMSE, and returns the coarse pose only if both fail.
- **Classifier accuracy (top-1, validation)**:

| Model | Normal validation | Validation with random dropout (0-6 hidden) |
|-------|------|------|
| Original | 59.2 % | 18.3 % |
| Dropout-trained | 56.3 % | 52.5 % |

  The dropout model gives up about 3 points on normal data and is far better when keypoints are missing.
  (The dropout model trained for all 40 epochs without early stopping.)
- **Sweep**: 2000 random poses, noise 2 and 5 px, 5 seeds. Extra keypoints are hidden so that about
  4-6, 7-9 or 10+ stay visible (a random target in the range; keypoints are never added, so the 10+
  group only uses poses that really have at least 10 visible), and the classifier features are
  rebuilt from the reduced set. Cells are mean over seeds; "pos" in m, "att" in deg, shown as mean / median.
  Gross failure = no solution, position > 10 m or attitude > 20 deg.

| Noise | Visible | Method | Pos mean / med | Att mean / med | Fail % | Fallback % |
|------|------|--------|------|------|------|------|
| 2 px | 4-6 | PnP only | 1.45 / 0.73 | 5.44 / 2.14 | 2.8 | 0.0 |
|  |  | Classifier + PnP (orig clf) | 5.71 / 0.73 | 4.46 / 2.18 | 5.8 | 4.7 |
|  |  | Classifier + PnP (dropout clf) | 1.35 / 0.70 | 2.64 / 2.11 | 0.8 | 0.0 |
|  |  | Hybrid (dropout clf) | 1.37 / 0.70 | 3.99 / 2.12 | 1.7 | 0.0 |
|  | 7-9 | PnP only | 0.89 / 0.48 | 1.66 / 1.45 | 0.0 | 0.0 |
|  |  | Classifier + PnP (orig clf) | 5.58 / 0.49 | 2.94 / 1.47 | 4.3 | 4.3 |
|  |  | Classifier + PnP (dropout clf) | 0.87 / 0.48 | 1.65 / 1.44 | 0.0 | 0.0 |
|  |  | Hybrid (dropout clf) | 0.87 / 0.48 | 1.65 / 1.44 | 0.0 | 0.0 |
|  | 10+ | PnP only | 0.80 / 0.44 | 1.37 / 1.21 | 0.0 | 0.0 |
|  |  | Classifier + PnP (orig clf) | 1.31 / 0.43 | 1.53 / 1.20 | 0.5 | 0.5 |
|  |  | Classifier + PnP (dropout clf) | 0.80 / 0.43 | 1.37 / 1.20 | 0.0 | 0.0 |
|  |  | Hybrid (dropout clf) | 0.77 / 0.43 | 1.37 / 1.20 | 0.0 | 0.0 |
| 5 px | 4-6 | PnP only | 3.67 / 1.94 | 13.98 / 5.55 | 14.4 | 0.0 |
|  |  | Classifier + PnP (orig clf) | 7.53 / 1.89 | 8.43 / 5.55 | 13.1 | 4.8 |
|  |  | Classifier + PnP (dropout clf) | 3.35 / 1.79 | 6.80 / 5.36 | 8.7 | 0.0 |
|  |  | Hybrid (dropout clf) | 3.39 / 1.82 | 11.46 / 5.46 | 11.8 | 0.0 |
|  | 7-9 | PnP only | 2.50 / 1.30 | 4.87 / 3.63 | 4.2 | 0.0 |
|  |  | Classifier + PnP (orig clf) | 6.96 / 1.23 | 5.27 / 3.66 | 6.6 | 4.5 |
|  |  | Classifier + PnP (dropout clf) | 2.17 / 1.18 | 4.11 / 3.58 | 2.3 | 0.0 |
|  |  | Hybrid (dropout clf) | 2.14 / 1.18 | 4.37 / 3.58 | 2.5 | 0.0 |
|  | 10+ | PnP only | 2.35 / 1.30 | 3.49 / 2.98 | 2.7 | 0.0 |
|  |  | Classifier + PnP (orig clf) | 2.59 / 1.10 | 3.58 / 2.96 | 1.6 | 0.6 |
|  |  | Classifier + PnP (dropout clf) | 1.90 / 1.10 | 3.37 / 2.95 | 1.0 | 0.0 |
|  |  | Hybrid (dropout clf) | 1.88 / 1.10 | 3.37 / 2.95 | 1.0 | 0.0 |

Classifier-only numbers and no-solution counts are in `results/pnp_dropout_metrics.json`.

![dropout sweep](results/pnp_dropout.png)

### Conclusion

- **The Phase 4 hint was a training-distribution problem, not a PnP problem.** With the original
  classifier, classifier + PnP falls back to the coarse pose on about 4-5 % of frames when keypoints
  are dropped (even at 7-9 visible), giving mean position errors of 5-6 m. With the dropout-trained
  classifier the fallback rate is about 0.
- **Medians barely change between methods** (all within a few percent): for typical frames,
  PnP-only is already as accurate as anything else.
- **The classifier helps in the tails.** With the dropout-trained classifier, classifier + PnP has the
  lowest (or tied-lowest) gross-failure rate in every cell except 10+ visible at 2 px, where failures are
  ~0 for every method (PnP-only and the hybrid 0.0 %, classifier + PnP 0.02 %): at 4-6 visible, 0.8 % vs 2.8 % for PnP-only at 2 px and
  8.7 % vs 14.4 % at 5 px, and it halves the mean attitude error at 4-6 / 2 px (2.64 vs 5.44 deg). The gain
  shrinks as more keypoints are visible and noise is low, and is zero in the median.
- **The hybrid did not beat classifier + PnP (dropout clf).** It is equal or marginally better at 7+
  visible, but worse at 4-6 visible (attitude mean 3.99 vs 2.64 deg and 1.7 % vs 0.8 % failures at
  2 px), though still better than PnP-only. Choosing the solution with the lowest reprojection
  error does not choose the one closest to the truth: with few noisy points, the wrong pose
  can fit the pixels equally well, and the classifier's pose is the better tie-breaker. The hybrid's
  advantage is robustness to a bad classifier (it is the safe choice with the original model).
- **Summary**: the classifier is useful only as an initial guess that has seen missing keypoints, and
  then it mainly removes occasional gross errors; it does not improve typical accuracy. These are
  simulated, ideal-detector results (our occlusion model, Gaussian noise only).

## Phase 5: filtering on top of PnP

Two filters that use the PnP pose as their measurement, compared with the Phase 3 baseline
particle filter (untouched) and with single-frame PnP.

| File | Purpose |
|------|---------|
| `filters.py` | added `KalmanFilter`, `ImprovedParticleFilter`, `PnPOnly`, `pnp_measurement`, `calibrate_pnp_noise`; `run_filter` got an optional `outage` argument (default behaviour unchanged) |
| `run_phase5.py` | calibration, experiments A and B, figures, metrics |
| `run_phase5_ablation.py` | the likelihood / particle-source ablation table in the Discussion |
| `test_phase5.py` | sanity checks (prints PASS/FAIL) |
| `results/kf_noise_calibration.json`, `filter_metrics.json` | calibrated R and all numbers |
| `results/filter_errors_2px.png`, `filter_noise_sweep.png`, `filter_outage.png`, `filter_trajectory.png` | figures |

```bash
python test_phase5.py
python run_phase5.py     # about 50 s
```

### The PnP measurement

`hybrid_pnp` (Phase 4b) started from the **original** classifier (the same one the Phase 3 filter uses).
If fewer than 4 keypoints are visible, or both solvers fail and the hybrid falls back to the coarse
classifier pose, there is no measurement (a coarse pose is not trusted as a measurement).

### Kalman filter

- **State (12):** `[x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]`,
  constant velocity for both position and angles, linear. Angle innovations are wrapped to [-180, 180).
- **Q:** discrete white-noise-acceleration model, per axis with state `[p, v]`:
  `Q = sigma_a^2 [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]`. The Phase 3 filter uses a uniform random
  acceleration in +/-`A_MAX`, whose std is `A_MAX / sqrt(3)`, so `sigma_a = A_MAX / sqrt(3)` for position
  and `W_MAX / sqrt(3)` deg/s^2 for the angles (we read `W_MAX` as how fast the rate can change
  per second: our interpretation, since in Phase 3 it is a rate, not an acceleration).
- **R (calibration):** `calibrate_pnp_noise` runs the PnP measurement on 1000 random poses (seed 424242,
  used nowhere else) at each noise level and takes the 6x6 error covariance. Position errors are divided by
  `range / 65 m` first, so one covariance describes the reference range; at run time the position
  variances are scaled by `(range / 65)^2` using the range of the measurement. Frames with no solution
  or a gross failure (position > 10 m or attitude > 20 deg, the Phase 4 definition) are left out of
  the calibration (11 of 1000 at 5 px, none otherwise), because the gate handles those at run
  time. A floor of 0.01 m / 0.01 deg std is added so that R is invertible at 0 px.
  Calibrated std at 65 m (x, y, z in m; roll, pitch, yaw in deg):

| Noise | x | y | z | roll | pitch | yaw |
|-------|------|------|------|------|------|------|
| 0 px (floor) | 0.010 | 0.010 | 0.010 | 0.010 | 0.010 | 0.010 |
| 1 px | 0.390 | 0.211 | 0.092 | 0.526 | 0.388 | 0.456 |
| 2 px | 0.781 | 0.423 | 0.183 | 1.052 | 0.777 | 0.913 |
| 5 px | 1.797 | 1.004 | 0.443 | 2.636 | 1.938 | 2.278 |

- **Initialisation:** from the first PnP measurement, velocities and rates zero, with std 5 m/s
  (`V_INIT_MAX`) and 20 deg/s (`W_MAX`) as the initial uncertainty.
- **Gating:** the update is skipped if the squared Mahalanobis distance of the innovation exceeds 22.46
  (chi-squared 99.9 %, 6 dof); the rejections are counted. With no measurement or a rejected one the
  filter only predicts.

### Improved particle filter

Same as the Phase 3 filter (dynamics, measured boresight, systematic resampling), with these changes only:
Gaussian likelihood `w_i ~ exp(-SSE_i / (2 sigma^2))`, `sigma = max(noise_px, 1)` px, SSE over the
visible keypoints in log space; initial and fresh particles are drawn from `N(PnP pose, R)` instead of the
classifier bin plus error samples; N = 1000, alpha = 0.9 every step. During an outage there is no PnP pose, so the
"fresh" 10 % are copied from the current cloud (plain resampling).

### Experiments

Approach trajectory, noise 0, 1, 2, 5 px, 5 seeds (each method sees the same observations for a given
seed). **A**: normal. **B**: all keypoints hidden for 4.0 <= t < 5.0 s (10 steps); single-frame PnP has no
output there. Cells are the mean over seeds of the per-run statistic (the standard deviations are in
`results/filter_metrics.json`). "Gated" = Kalman updates rejected per run (101 steps).

**Experiment A (normal run)**

| Noise | Method | Pos mean / med [m] | Att mean / med [deg] | Gated | ms/step |
|------|--------|------|------|------|------|
| 0 px | Phase 3 PF (Config B) | 4.91 / 4.68 | 5.19 / 4.64 | - | 8.5 |
|  | PnP alone | 0.00 / 0.00 | 0.00 / 0.00 | - | 0.3 |
|  | Improved PF | 0.58 / 0.58 | 0.50 / 0.46 | - | 2.1 |
|  | Kalman filter | 0.00 / 0.00 | 0.00 / 0.00 | 0.0 | 0.3 |
| 1 px | Phase 3 PF (Config B) | 6.77 / 6.47 | 5.90 / 5.18 | - | 8.5 |
|  | PnP alone | 0.43 / 0.31 | 0.74 / 0.72 | - | 0.3 |
|  | Improved PF | 0.75 / 0.70 | 1.10 / 1.06 | - | 2.1 |
|  | Kalman filter | 0.23 / 0.16 | 0.75 / 0.70 | 2.2 | 0.3 |
| 2 px | Phase 3 PF (Config B) | 6.35 / 6.07 | 6.20 / 5.64 | - | 8.5 |
|  | PnP alone | 0.87 / 0.63 | 1.49 / 1.44 | - | 0.3 |
|  | Improved PF | 1.18 / 1.05 | 1.56 / 1.47 | - | 2.1 |
|  | Kalman filter | 0.43 / 0.30 | 1.50 / 1.43 | 2.4 | 0.3 |
| 5 px | Phase 3 PF (Config B) | 7.91 / 7.70 | 7.45 / 7.12 | - | 8.5 |
|  | PnP alone | 2.17 / 1.60 | 3.72 / 3.56 | - | 0.3 |
|  | Improved PF | 2.13 / 1.76 | 3.08 / 2.88 | - | 2.1 |
|  | Kalman filter | 1.07 / 0.82 | 3.67 / 3.70 | 3.2 | 0.4 |

(Runtimes include the classifier and PnP calls for every method except the baseline.)

**Experiment B (outage 4.0-5.0 s; "after" = 5.0-6.0 s), mean errors**

| Noise | Method | Pos in outage [m] | Att in outage [deg] | Pos after [m] | Att after [deg] | Gated |
|------|--------|------|------|------|------|------|
| 0 px | Phase 3 PF (Config B) | 9.42 | 10.50 | 6.54 | 9.02 | - |
|  | PnP alone | missing (10 steps) | missing | 0.00 | 0.00 | - |
|  | Improved PF | 2.70 | 5.97 | 0.95 | 1.25 | - |
|  | Kalman filter | 0.00 | 4.10 | 0.00 | 0.00 | 0.0 |
| 1 px | Phase 3 PF (Config B) | 9.09 | 10.45 | 7.64 | 8.42 | - |
|  | PnP alone | missing | missing | 0.42 | 0.72 | - |
|  | Improved PF | 3.68 | 6.16 | 1.41 | 1.96 | - |
|  | Kalman filter | 0.48 | 8.67 | 0.75 | 28.34 | 26.6 |
| 2 px | Phase 3 PF (Config B) | 9.24 | 10.46 | 8.93 | 9.58 | - |
|  | PnP alone | missing | missing | 0.84 | 1.45 | - |
|  | Improved PF | 3.17 | 6.37 | 1.54 | 2.10 | - |
|  | Kalman filter | 0.82 | 11.51 | 1.24 | 35.08 | 28.2 |
| 5 px | Phase 3 PF (Config B) | 9.33 | 12.17 | 10.30 | 10.56 | - |
|  | PnP alone | missing | missing | 2.10 | 3.65 | - |
|  | Improved PF | 2.95 | 6.57 | 3.07 | 3.75 | - |
|  | Kalman filter | 2.20 | 16.24 | 3.03 | 45.59 | 21.0 |

![errors at 2 px](results/filter_errors_2px.png)
![noise sweep](results/filter_noise_sweep.png)
![outage](results/filter_outage.png)
![Kalman trajectory](results/filter_trajectory.png)

### Discussion

**Where filtering helps.**
- *Position under noise.* With 1-5 px noise the Kalman filter roughly halves the position error of
  single-frame PnP (0.43 vs 0.87 m at 2 px, 1.07 vs 2.17 m at 5 px). Position moves almost at constant
  velocity, which is exactly what the model assumes, so averaging over time pays off.
- *Outages.* The Kalman filter keeps a position estimate through the gap (0.5-2.2 m, similar to or better than
  PnP's own noise level) where single-frame PnP has nothing.

**Where it does not.**
- *Zero noise.* The Kalman filter equals PnP (0.00 m): nothing to average, and with R at its floor it simply follows
  the measurement without visible lag. The improved particle filter is *worse* than PnP at 0 px
  (0.58 m vs 0.00) and also at 1-2 px (1.18 vs 0.87 m at 2 px): it only approximates the posterior
  with 1000 samples, and its random-rate dynamics add scatter that a single PnP solve does not have.
- *Attitude.* The Kalman filter's attitude error is the same as PnP's (1.50 vs 1.49 deg at 2 px, 3.67 vs 3.72 at
  5 px). The roll swings +/-15 deg in 5 s, so the constant-rate model lags as much as smoothing gains.
- *After an outage the Kalman filter fails badly in attitude* (28-46 deg mean over the following second at
  noise > 0). The cause, checked step by step for seed 0 at 2 px: the roll rate estimated before the gap
  (-6.9 deg/s) is wrong for the next second (the true rate swings to +15), so the extrapolated roll
  is 24 deg off at t = 5 s, about 6 standard deviations of the predicted covariance. The gate then
  rejects 14 consecutive *good* measurements (a gate lock-out, which is why the gated counts are 21-28 per run), while
  the roll error drifts to -47 deg, until the covariance has grown enough at t = 6.4 s. Position is fine because it really is near constant velocity.
  This is a consequence of the specified design (white-noise-acceleration Q from `A_MAX`/`W_MAX`, hard gate), not
  tuned or fixed here. Typical remedies, which we did not apply: re-initialise after N consecutive rejections, or
  inflate the angular Q.
- *The gate rejects 2-3 % of normal measurements* (nominal 0.1 %). Measured on fresh random poses at 2 px,
  3 % of PnP errors exceed the gate under the calibrated R. The reason is that depth error grows faster
  than linearly with range: the position std rescaled to 65 m is 0.45 m for ranges of 25-50 m, 0.77 m (50-75),
  1.17 m (75-100) and 1.59 m (100-150), whereas the specified `(range / 65)^2` variance scaling assumes it is
  constant. At the far end of the trajectory R is therefore too small.
- *The single-frame PnP beats the filters* in these cases: attitude at 0-2 px (equal or marginally better than
  the Kalman filter, clearly better than the improved PF), position at 0-2 px against the improved PF, and everything after the outage
  against the Kalman filter's attitude.

**Improved vs Phase 3 baseline particle filter.** The improved filter is far better (1.2 vs 6.4 m and 1.6 vs 6.2 deg at
2 px) and approaches PnP quality, but it is not better than PnP itself except for attitude at 5 px
(3.08 vs 3.72 deg). Its two changes were made together, so to see which matters we ran an ablation
(N = 1000, alpha = 0.9 every step, 5 seeds, mean position [m] / attitude [deg]; reproduce with
`python run_phase5_ablation.py`, which needs `results/kf_noise_calibration.json` from `run_phase5.py`):

| Likelihood / particle source | 0 px | 2 px | 5 px |
|------|------|------|------|
| 1/(MSE+eps) + classifier (baseline design, N = 1000) | 5.82 / 6.36 | 7.80 / 7.15 | 9.28 / 8.43 |
| Gaussian + classifier | 4.81 / 5.56 | 6.53 / 4.91 | 6.35 / 5.90 |
| 1/(MSE+eps) + PnP | 1.18 / 1.70 | 2.15 / 3.60 | 3.51 / 5.54 |
| Gaussian + PnP (improved) | 0.58 / 0.50 | 1.18 / 1.56 | 2.13 / 3.08 |

The paper's choice of drawing the particles from the classifier and its error samples is what caused the
baseline's failure: those proposals are 5-7 m off, and no weighting can pull the estimate below the error of the
particles it is given. The `1/(MSE+eps)` likelihood is a smaller second problem (it weights particles only
polynomially, not sharply), worth about a factor of 1.5-2 once the particles are good.
Both are simulation results with an idealised detector (Gaussian pixel noise, our occlusion model).
