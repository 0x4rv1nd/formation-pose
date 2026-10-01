"""Sanity checks for Phase 2 (prints PASS/FAIL). Run after `python classifier.py`."""

import json
import os
import sys

import numpy as np

import classifier
import config
import dataset

results = []


def check(name, ok):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


d = np.load(os.path.join(config.DATA_DIR, "val.npz"))
X = d["X"][:500]
n_val = len(d["X"])

model = classifier.load_model()
probs = classifier.predict_probs(model, X)
check("predict_probs shape is (n, 4800)", probs.shape == (len(X), config.N_LABELS))
check("probabilities sum to 1", np.allclose(probs.sum(axis=1), 1.0, atol=1e-4))

labels = classifier.predict_label(model, X)
check("predict_label shape is (n,)", labels.shape == (len(X),))

poses = classifier.predict_pose(model, X)
valid = poses.shape == (len(X), 6)
for dim, grid in enumerate(config.GRID):
    valid &= bool(np.isin(np.round(poses[:, dim], 6), np.round(grid, 6)).all())
valid &= bool((poses[:, 4:] == 0).all())
check("predict_pose returns valid grid points", valid)
check("predict_pose agrees with label_to_pose",
      np.allclose(poses, dataset.label_to_pose(labels)))

model2 = classifier.load_model()
check("reloaded model gives identical predictions",
      np.array_equal(labels, classifier.predict_label(model2, X)))

err_path = classifier.ERROR_SAMPLES_PATH
ok = os.path.exists(err_path) and np.load(err_path).shape == (n_val, 6)
check(f"error_samples.npy exists with shape ({n_val}, 6)", ok)

with open(os.path.join(config.RESULTS_DIR, "classifier_metrics.json")) as f:
    top1 = json.load(f)["top1_accuracy"]
check(f"validation top-1 accuracy > 49% (got {top1 * 100:.1f}%)", top1 > 0.49)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
