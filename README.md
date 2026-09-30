# formation-pose

A student reproduction of **"Vision-Based Precision Pose Estimation for Autonomous
Formation Flying"** (Punnoose, Stanford).

A follower aircraft carries a gimballed camera that always points at a leader aircraft.
The goal is to estimate the leader's relative pose (position + attitude) from the pixel
locations of 14 keypoints on the leader. Like the paper, we assume a perfect keypoint
detector and use purely synthetic data.

The project is built in 6 phases; this repository currently contains Phase 1.

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
```

This writes `data/train.npz` and `data/val.npz` (keys `X`, `y`, `poses`) and
`results/sample_keypoints.png`.

- **Pose**: `[x, y, z, roll, pitch, yaw]` - follower position relative to the leader
  (metres, leader body axes, x forward / y right / z down) and leader attitude relative
  to the follower (degrees, ZYX Euler). See the docstring in `config.py`.
- **Features**: 42 numbers, `(u, v, visible)` per keypoint, with `u, v` normalised
  about the image centre; hidden keypoints are zeroed.
- **Labels**: nearest point of the paper's Table I grid over x, y, z and roll
  (10 x 10 x 8 x 6 = 4800 classes).

![sample keypoints](results/sample_keypoints.png)

### Simplifications

- The 14 keypoints are an approximate F-16 layout, not measured from a real model.
- Occlusion uses a simple fuselage ellipsoid plus two flat wing triangles; the tail,
  canopy and stores do not occlude anything.
- Pitch and yaw are sampled (+/-10 deg) but are not part of the classification label.
