# formation-pose

A student reproduction of **"Vision-Based Precision Pose Estimation for Autonomous
Formation Flying"** (Punnoose, Stanford).

A follower aircraft carries a gimballed camera that always points at a leader aircraft.
The goal is to estimate the leader's relative pose (position + attitude) from the pixel
locations of 14 keypoints on the leader. Like the paper, we assume a perfect keypoint
detector and use purely synthetic data.

The project is built in 6 phases; this repository currently contains Phases 1 to 3.

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
