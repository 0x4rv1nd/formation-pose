| Noise | Filter | Trans mean [m] | Trans median [m] | Rot mean [deg] | Rot median [deg] | Gate rejection | Re-inits |
|---|---|---|---|---|---|---|---|
| 0 px | PnP only (reference) | 0.00 | 0.00 | 0.00 | 0.00 | - | - |
| | Kalman, attitude re-init (before) | 1.86 ± 0.00 | 0.00 | 0.12 ± 0.00 | 0.00 | 13.1 % | 98 |
| | Kalman, full re-init (after) | 0.01 ± 0.00 | 0.00 | 0.07 ± 0.00 | 0.00 | 5.5 % | 39 |
| 1 px | PnP only (reference) | 1.25 | 0.86 | 0.55 | 0.44 | - | - |
| | Kalman, attitude re-init (before) | 3.03 ± 0.44 | 0.60 | 0.49 ± 0.06 | 0.33 | 12.5 % | 81 |
| | Kalman, full re-init (after) | 0.78 ± 0.03 | 0.55 | 0.45 ± 0.06 | 0.32 | 6.2 % | 32 |
| 2 px | PnP only (reference) | 2.59 | 1.76 | 1.36 | 0.88 | - | - |
| | Kalman, attitude re-init (before) | 3.09 ± 0.34 | 1.15 | 0.79 ± 0.06 | 0.59 | 10.2 % | 61 |
| | Kalman, full re-init (after) | 1.52 ± 0.07 | 1.06 | 0.76 ± 0.06 | 0.57 | 5.4 % | 23 |
| 5 px | PnP only (reference) | 7.15 | 4.84 | 5.60 | 2.24 | - | - |
| | Kalman, attitude re-init (before) | 5.10 ± 0.59 | 3.52 | 2.57 ± 0.50 | 1.29 | 8.1 % | 31 |
| | Kalman, full re-init (after) | 4.73 ± 0.10 | 3.50 | 2.53 ± 0.46 | 1.28 | 6.8 % | 21 |

| Scene (2 px) | PnP trans / rot | Kalman before: trans / rot, gate, re-inits | Kalman after: trans / rot, gate, re-inits |
|---|---|---|---|
| 000096 | 1.07 m / 0.69° | 29.59 m / 0.99°, 82 %, 26.8 | 0.57 m / 0.51°, 3 %, 1.0 |
| 000009 | 2.77 m / 0.98° | 12.02 m / 0.92°, 42 %, 12.6 | 1.53 m / 0.64°, 5 %, 0.6 |

Kalman beats PnP in translation in 20 of 24 scenes before and 22 after (2 px).

| dt | Trans mean before / after [m] | Rot mean before / after [deg] | Gate rejection before / after | Re-inits before / after |
|---|---|---|---|---|
| 0.1 s | 3.09 / 1.52 | 0.79 / 0.76 | 10.2 % / 5.4 % | 61 / 23 |
| 0.05 s | 5.51 / 1.84 | 0.92 / 0.89 | 18.5 % / 8.5 % | 120 / 43 |
| 0.2 s | 1.27 / 1.31 | 0.80 / 0.80 | 3.5 % / 3.5 % | 11 / 10 |
