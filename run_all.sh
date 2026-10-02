#!/usr/bin/env bash
# Regenerates the data, trains both classifiers, runs every phase's experiments and the final evaluation.
# WARNING: overwrites data/, models/ and results/. Takes several minutes on a laptop CPU.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
start=$(date +%s)

$PY test_phase1.py
$PY dataset.py                  # data/train.npz, data/val.npz (Phase 1)
$PY classifier.py               # models/classifier.pt, models/error_samples.npy (Phase 2)
$PY run_phase3.py               # baseline particle filter
$PY run_phase4.py               # PnP
$PY train_dropout.py            # models/classifier_dropout.pt (Phase 4b)
$PY run_phase4b.py
$PY run_phase5.py               # Kalman / improved particle filter; writes kf_noise_calibration.json
$PY run_phase5_ablation.py
$PY run_phase5b.py              # range-calibrated Kalman filter (adds to kf_noise_calibration.json)
$PY evaluate.py                 # final tables and figures in results/final/
./run_tests.sh

echo "Total runtime: $(( $(date +%s) - start )) s"
