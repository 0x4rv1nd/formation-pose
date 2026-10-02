# Final results

Approach trajectory (101 steps), 5 seeds. Cells: mean ± std over seeds of the per-run statistic. Position error in m, attitude error (geodesic angle) in degrees. `n/a`: no estimate. Runtime includes the classifier / PnP calls of each method (for the paper's filter, the classifier is called inside the filter), measured on this machine.

## A. Approach trajectory

### 0 px keypoint noise

| Method | Pos mean [m] | Pos median [m] | Att mean [deg] | Att median [deg] | ms/step | Gate rejection |
|---|---|---|---|---|---|---|
| Classifier only | 7.05 ± 0.00 | 7.52 ± 0.00 | 5.03 ± 0.00 | 4.96 ± 0.00 | 0.05 | - |
| Paper's particle filter | 4.91 ± 1.14 | 4.68 ± 1.38 | 5.19 ± 0.55 | 4.64 ± 0.67 | 8.86 | - |
| PnP only (SQPnP) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.07 | - |
| Classifier + PnP | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.21 | - |
| Hybrid PnP | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.29 | - |
| Improved particle filter | 0.58 ± 0.13 | 0.58 ± 0.13 | 0.50 ± 0.22 | 0.46 ± 0.25 | 2.12 | - |
| Kalman filter (Phase 5) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.33 | 0.0 % |
| Range-calibrated Kalman filter | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.33 | 0.0 % |

### 1 px keypoint noise

| Method | Pos mean [m] | Pos median [m] | Att mean [deg] | Att median [deg] | ms/step | Gate rejection |
|---|---|---|---|---|---|---|
| Classifier only | 7.19 ± 0.05 | 7.64 ± 0.12 | 5.14 ± 0.07 | 4.96 ± 0.00 | 0.05 | - |
| Paper's particle filter | 6.77 ± 0.83 | 6.47 ± 0.97 | 5.90 ± 0.26 | 5.18 ± 0.52 | 8.69 | - |
| PnP only (SQPnP) | 0.44 ± 0.02 | 0.32 ± 0.01 | 0.75 ± 0.04 | 0.72 ± 0.05 | 0.08 | - |
| Classifier + PnP | 0.43 ± 0.02 | 0.31 ± 0.01 | 0.74 ± 0.04 | 0.72 ± 0.07 | 0.22 | - |
| Hybrid PnP | 0.43 ± 0.02 | 0.31 ± 0.01 | 0.74 ± 0.04 | 0.72 ± 0.07 | 0.30 | - |
| Improved particle filter | 0.75 ± 0.13 | 0.70 ± 0.13 | 1.10 ± 0.19 | 1.06 ± 0.19 | 2.12 | - |
| Kalman filter (Phase 5) | 0.23 ± 0.04 | 0.16 ± 0.04 | 0.75 ± 0.05 | 0.70 ± 0.05 | 0.34 | 2.2 % |
| Range-calibrated Kalman filter | 0.23 ± 0.03 | 0.16 ± 0.03 | 0.76 ± 0.03 | 0.72 ± 0.06 | 0.34 | 0.4 % |

### 2 px keypoint noise

| Method | Pos mean [m] | Pos median [m] | Att mean [deg] | Att median [deg] | ms/step | Gate rejection |
|---|---|---|---|---|---|---|
| Classifier only | 7.26 ± 0.11 | 7.49 ± 0.22 | 5.24 ± 0.12 | 4.98 ± 0.04 | 0.05 | - |
| Paper's particle filter | 6.35 ± 1.03 | 6.07 ± 1.29 | 6.20 ± 0.20 | 5.64 ± 0.33 | 8.53 | - |
| PnP only (SQPnP) | 0.89 ± 0.05 | 0.64 ± 0.05 | 1.50 ± 0.08 | 1.45 ± 0.10 | 0.09 | - |
| Classifier + PnP | 0.87 ± 0.04 | 0.63 ± 0.03 | 1.49 ± 0.08 | 1.44 ± 0.14 | 0.22 | - |
| Hybrid PnP | 0.87 ± 0.04 | 0.63 ± 0.03 | 1.49 ± 0.08 | 1.44 ± 0.14 | 0.31 | - |
| Improved particle filter | 1.18 ± 0.23 | 1.05 ± 0.23 | 1.56 ± 0.16 | 1.47 ± 0.17 | 2.13 | - |
| Kalman filter (Phase 5) | 0.43 ± 0.08 | 0.30 ± 0.05 | 1.50 ± 0.09 | 1.43 ± 0.11 | 0.35 | 2.4 % |
| Range-calibrated Kalman filter | 0.43 ± 0.05 | 0.30 ± 0.06 | 1.53 ± 0.06 | 1.50 ± 0.09 | 0.35 | 0.4 % |

### 5 px keypoint noise

| Method | Pos mean [m] | Pos median [m] | Att mean [deg] | Att median [deg] | ms/step | Gate rejection |
|---|---|---|---|---|---|---|
| Classifier only | 8.08 ± 0.23 | 7.86 ± 0.14 | 5.77 ± 0.19 | 5.19 ± 0.15 | 0.05 | - |
| Paper's particle filter | 7.91 ± 0.43 | 7.70 ± 0.57 | 7.45 ± 0.46 | 7.12 ± 0.66 | 8.55 | - |
| PnP only (SQPnP) | 2.76 ± 0.14 | 1.94 ± 0.24 | 4.35 ± 0.63 | 3.66 ± 0.25 | 0.10 | - |
| Classifier + PnP | 2.17 ± 0.10 | 1.60 ± 0.10 | 3.72 ± 0.20 | 3.56 ± 0.32 | 0.22 | - |
| Hybrid PnP | 2.17 ± 0.10 | 1.60 ± 0.10 | 3.72 ± 0.20 | 3.56 ± 0.32 | 0.33 | - |
| Improved particle filter | 2.13 ± 0.38 | 1.76 ± 0.39 | 3.08 ± 0.30 | 2.88 ± 0.38 | 2.14 | - |
| Kalman filter (Phase 5) | 1.07 ± 0.22 | 0.82 ± 0.20 | 3.67 ± 0.27 | 3.70 ± 0.20 | 0.35 | 3.2 % |
| Range-calibrated Kalman filter | 1.03 ± 0.14 | 0.73 ± 0.16 | 3.66 ± 0.19 | 3.72 ± 0.27 | 0.36 | 0.8 % |

## B. Outage: all keypoints hidden for 4.0 <= t < 5.0 s

Mean error during the outage and in the 1 s after it (5.0 <= t < 6.0 s). Single-frame methods have no estimate while there are no keypoints (`n/a`). Re-initialisations: range-calibrated Kalman filter, mean per run.

| Noise | Method | Pos in outage [m] | Att in outage [deg] | Pos after [m] | Att after [deg] |
|---|---|---|---|---|---|
| 0 px | Classifier only | n/a | n/a | 5.66 ± 0.00 | 4.85 ± 0.00 |
| 0 px | Paper's particle filter | 9.42 ± 0.75 | 10.50 ± 0.56 | 6.54 ± 0.92 | 9.02 ± 0.85 |
| 0 px | PnP only (SQPnP) | n/a | n/a | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 0 px | Classifier + PnP | n/a | n/a | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 0 px | Hybrid PnP | n/a | n/a | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 0 px | Improved particle filter | 2.70 ± 1.19 | 5.97 ± 0.06 | 0.95 ± 0.36 | 1.25 ± 0.41 |
| 0 px | Kalman filter (Phase 5) | 0.00 ± 0.00 | 4.10 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 0 px | Range-calibrated Kalman filter | 0.00 ± 0.00 | 4.10 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 1 px | Classifier only | n/a | n/a | 5.83 ± 0.15 | 4.85 ± 0.00 |
| 1 px | Paper's particle filter | 9.09 ± 0.64 | 10.45 ± 1.01 | 7.64 ± 1.44 | 8.42 ± 1.47 |
| 1 px | PnP only (SQPnP) | n/a | n/a | 0.40 ± 0.05 | 0.73 ± 0.08 |
| 1 px | Classifier + PnP | n/a | n/a | 0.42 ± 0.06 | 0.72 ± 0.08 |
| 1 px | Hybrid PnP | n/a | n/a | 0.42 ± 0.06 | 0.72 ± 0.08 |
| 1 px | Improved particle filter | 3.68 ± 0.26 | 6.16 ± 0.25 | 1.41 ± 0.11 | 1.96 ± 0.42 |
| 1 px | Kalman filter (Phase 5) | 0.48 ± 0.22 | 8.67 ± 0.69 | 0.75 ± 0.35 | 28.34 ± 2.78 |
| 1 px | Range-calibrated Kalman filter | 0.44 ± 0.19 | 9.13 ± 0.48 | 0.36 ± 0.07 | 4.78 ± 0.27 |
| 2 px | Classifier only | n/a | n/a | 5.83 ± 0.15 | 4.85 ± 0.00 |
| 2 px | Paper's particle filter | 9.24 ± 0.64 | 10.46 ± 0.73 | 8.93 ± 1.55 | 9.58 ± 1.16 |
| 2 px | PnP only (SQPnP) | n/a | n/a | 0.76 ± 0.07 | 1.45 ± 0.16 |
| 2 px | Classifier + PnP | n/a | n/a | 0.84 ± 0.12 | 1.45 ± 0.17 |
| 2 px | Hybrid PnP | n/a | n/a | 0.84 ± 0.12 | 1.45 ± 0.17 |
| 2 px | Improved particle filter | 3.17 ± 0.96 | 6.37 ± 0.30 | 1.54 ± 0.46 | 2.10 ± 0.58 |
| 2 px | Kalman filter (Phase 5) | 0.82 ± 0.35 | 11.51 ± 0.60 | 1.24 ± 0.53 | 35.08 ± 1.42 |
| 2 px | Range-calibrated Kalman filter | 0.73 ± 0.30 | 12.13 ± 0.49 | 0.61 ± 0.14 | 6.21 ± 0.30 |
| 5 px | Classifier only | n/a | n/a | 7.80 ± 0.71 | 5.20 ± 0.31 |
| 5 px | Paper's particle filter | 9.33 ± 0.84 | 12.17 ± 0.48 | 10.30 ± 2.25 | 10.56 ± 0.55 |
| 5 px | PnP only (SQPnP) | n/a | n/a | 1.84 ± 0.23 | 3.63 ± 0.37 |
| 5 px | Classifier + PnP | n/a | n/a | 2.10 ± 0.30 | 3.65 ± 0.41 |
| 5 px | Hybrid PnP | n/a | n/a | 2.10 ± 0.30 | 3.65 ± 0.41 |
| 5 px | Improved particle filter | 2.95 ± 1.13 | 6.57 ± 0.33 | 3.07 ± 1.20 | 3.75 ± 0.51 |
| 5 px | Kalman filter (Phase 5) | 2.20 ± 0.57 | 16.24 ± 1.16 | 3.03 ± 0.78 | 45.59 ± 2.62 |
| 5 px | Range-calibrated Kalman filter | 1.63 ± 0.68 | 16.44 ± 0.98 | 1.33 ± 0.37 | 8.89 ± 0.50 |

## C. Keypoint dropout (Phase 4b, 2000 random poses, 5 seeds)

Median errors and gross-failure rate (no solution, position > 10 m or attitude > 20 deg). `dropout clf` is the classifier retrained with hidden keypoints.

| Noise | Visible | Method | Pos mean / median [m] | Att mean / median [deg] | Fail % |
|---|---|---|---|---|---|
| 2 px | 4-6 | PnP only | 1.45 / 0.73 | 5.44 / 2.14 | 2.8 |
| 2 px | 4-6 | Classifier + PnP (orig clf) | 5.71 / 0.73 | 4.46 / 2.18 | 5.8 |
| 2 px | 4-6 | Classifier + PnP (dropout clf) | 1.35 / 0.70 | 2.64 / 2.11 | 0.8 |
| 2 px | 4-6 | Hybrid (dropout clf) | 1.37 / 0.70 | 3.99 / 2.12 | 1.7 |
| 2 px | 7-9 | PnP only | 0.89 / 0.48 | 1.66 / 1.45 | 0.0 |
| 2 px | 7-9 | Classifier + PnP (orig clf) | 5.58 / 0.49 | 2.94 / 1.47 | 4.3 |
| 2 px | 7-9 | Classifier + PnP (dropout clf) | 0.87 / 0.48 | 1.65 / 1.44 | 0.0 |
| 2 px | 7-9 | Hybrid (dropout clf) | 0.87 / 0.48 | 1.65 / 1.44 | 0.0 |
| 2 px | 10+ | PnP only | 0.80 / 0.44 | 1.37 / 1.21 | 0.0 |
| 2 px | 10+ | Classifier + PnP (orig clf) | 1.31 / 0.43 | 1.53 / 1.20 | 0.5 |
| 2 px | 10+ | Classifier + PnP (dropout clf) | 0.80 / 0.43 | 1.37 / 1.20 | 0.0 |
| 2 px | 10+ | Hybrid (dropout clf) | 0.77 / 0.43 | 1.37 / 1.20 | 0.0 |
| 5 px | 4-6 | PnP only | 3.67 / 1.94 | 13.98 / 5.55 | 14.4 |
| 5 px | 4-6 | Classifier + PnP (orig clf) | 7.53 / 1.89 | 8.43 / 5.55 | 13.1 |
| 5 px | 4-6 | Classifier + PnP (dropout clf) | 3.35 / 1.79 | 6.80 / 5.36 | 8.7 |
| 5 px | 4-6 | Hybrid (dropout clf) | 3.39 / 1.82 | 11.46 / 5.46 | 11.8 |
| 5 px | 7-9 | PnP only | 2.50 / 1.30 | 4.87 / 3.63 | 4.2 |
| 5 px | 7-9 | Classifier + PnP (orig clf) | 6.96 / 1.23 | 5.27 / 3.66 | 6.6 |
| 5 px | 7-9 | Classifier + PnP (dropout clf) | 2.17 / 1.18 | 4.11 / 3.58 | 2.3 |
| 5 px | 7-9 | Hybrid (dropout clf) | 2.14 / 1.18 | 4.37 / 3.58 | 2.5 |
| 5 px | 10+ | PnP only | 2.35 / 1.30 | 3.49 / 2.98 | 2.7 |
| 5 px | 10+ | Classifier + PnP (orig clf) | 2.59 / 1.10 | 3.58 / 2.96 | 1.6 |
| 5 px | 10+ | Classifier + PnP (dropout clf) | 1.90 / 1.10 | 3.37 / 2.95 | 1.0 |
| 5 px | 10+ | Hybrid (dropout clf) | 1.88 / 1.10 | 3.37 / 2.95 | 1.0 |
