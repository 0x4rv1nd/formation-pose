# Key results

All numbers are produced by `evaluate.py` (approach trajectory, 5 seeds, simulated keypoints with Gaussian pixel noise; mean over seeds of the time-averaged error).

- At 2 px noise, the range-calibrated Kalman filter reduces position error from 6.35 m (paper's particle filter) to 0.43 m (15x lower) and attitude error from 6.20 deg to 1.53 deg (4.0x lower).
- At 5 px noise, the range-calibrated Kalman filter reduces position error from 7.91 m (paper's particle filter) to 1.03 m (8x lower) and attitude error from 7.45 deg to 3.66 deg (2.0x lower).
- The classifier alone has 7.26 m / 5.24 deg error at 2 px; the paper's particle filter (Config B, 5000 particles) only reaches 6.35 m / 6.20 deg, i.e. it does not clearly improve on the classifier.
- Most of the improvement comes from geometry, not filtering: single-frame PnP alone gives 0.89 m / 1.50 deg at 2 px, and 2.76 m / 4.35 deg at 5 px.
- Classifier + PnP matches SQPnP alone at 2 px (0.87 vs 0.89 m) and helps at 5 px (2.17 vs 2.76 m position, 3.72 vs 4.35 deg attitude).
- Filtering PnP with the Kalman filter lowers position error again (2 px: 0.87 m for the hybrid PnP measurement alone vs 0.43 m filtered; 5 px: 2.17 vs 1.03 m), but attitude is unchanged (1.49 vs 1.53 deg at 2 px).
- The improved particle filter (1000 particles) reaches 1.18 m / 1.56 deg at 2 px but is not better than the Kalman filter (0.43 m position at 2 px).
- Calibrating the Kalman measurement noise by range cuts gate rejections from 2.4 % to 0.4 % of measurements at 2 px (nominal 0.1 %), with unchanged normal accuracy (0.43 vs 0.43 m).
- After a 1 s measurement outage at 2 px the Phase 5 Kalman filter's attitude error in the following second is 35.1 deg; the re-initialising, range-calibrated filter brings this to 6.2 deg (the paper's filter: 9.6 deg). It still has 1.5 deg in normal operation, so a few steps after the gap remain poor until the filter re-initialises.
- During the outage (2 px) the range-calibrated Kalman filter keeps the position error at 0.73 m, against 9.24 m for the paper's filter; its attitude error grows to 12.1 deg (paper's filter: 10.5 deg, improved particle filter: 6.4 deg) because the rate model extrapolates the roll swing. Single-frame PnP gives nothing in the gap.
- With only 4-6 visible keypoints at 2 px, classifier + PnP with the dropout-trained classifier has a gross-failure rate of 0.8 % against 2.8 % for PnP only; median errors are the same (0.70 vs 0.73 m).
- Runtime per step (this machine, includes classifier and PnP calls): paper's particle filter 8.5 ms, improved particle filter 2.1 ms, range-calibrated Kalman filter 0.35 ms.
