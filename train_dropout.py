"""
Phase 4b: train a SEPARATE classifier that has seen missing keypoints.

Same architecture and settings as classifier.py (45 -> 100 -> 100 -> 4800, Adam 1e-3,
batch 256, <= 40 epochs, best checkpoint by validation accuracy, patience 5), but every
training sample has 0-6 extra visible keypoints hidden at random (u = v = visible = 0),
re-drawn every batch. models/classifier.pt is NOT touched; this writes
models/classifier_dropout.pt and results/classifier_dropout_metrics.json.

    python train_dropout.py

Model selection uses a validation set with the same dropout applied once (fixed seed).
Both models are also evaluated on the normal (clean) validation set.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import classifier
import config

DROPOUT_MODEL_PATH = os.path.join("models", "classifier_dropout.pt")
MAX_DROP = 6   # hide between 0 and this many extra visible keypoints per sample


def drop_features(X, gen, max_drop=MAX_DROP):
    """
    Hide a random number (0..max_drop) of the visible keypoints of every row of X
    (n, 45): their u, v and visible entries become 0. Returns a new tensor.
    """
    X = X.clone()
    n = len(X)
    vis = X[:, 2:3 * config.N_KEYPOINTS:3] > 0.5                       # (n, 14)
    n_drop = torch.randint(0, max_drop + 1, (n, 1), generator=gen).to(X.device)
    # Random order of the keypoints; the first n_drop VISIBLE ones in that order are hidden
    scores = torch.rand(n, config.N_KEYPOINTS, generator=gen).to(X.device)
    scores = torch.where(vis, scores, torch.full_like(scores, -1.0))   # hidden ones go last
    rank = scores.argsort(dim=1, descending=True).argsort(dim=1)       # 0 = first picked
    hide = vis & (rank < n_drop)
    for j in range(3):                                                 # u, v, visible columns
        X[:, j:3 * config.N_KEYPOINTS:3][hide] = 0.0
    return X


def main():
    torch.manual_seed(config.SEED)
    device = classifier.get_device()
    print(f"Device: {device}")

    Xtr, ytr, _ = classifier.load_split("train")
    Xva, yva, _ = classifier.load_split("val")
    Xtr, ytr, Xva, yva = (torch.as_tensor(a).to(device) for a in (Xtr, ytr, Xva, yva))
    gen = torch.Generator().manual_seed(config.SEED)
    Xva_drop = drop_features(Xva, torch.Generator().manual_seed(config.SEED + 1))  # fixed

    model = classifier.PoseClassifier().to(device)
    Xstat = drop_features(Xtr, torch.Generator().manual_seed(config.SEED + 2))
    model.mean.copy_(Xstat.mean(0))
    model.std.copy_(Xstat.std(0).clamp_min(1e-6))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, bad_epochs, patience = -1.0, 0, 5
    epochs_trained = 0
    for epoch in range(1, 41):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen).to(device)
        running = 0.0
        for i in range(0, len(Xtr), 256):
            idx = perm[i:i + 256]
            xb = drop_features(Xtr[idx], gen)          # fresh dropout every batch
            loss = loss_fn(model(xb), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        _, acc_drop = classifier.evaluate(model, Xva_drop, yva, loss_fn)
        epochs_trained = epoch
        print(f"epoch {epoch:2d} | train loss {running / len(Xtr):.4f} | "
              f"val acc (with dropout) {acc_drop:.4f} | {time.time() - t0:.1f} s")
        if acc_drop > best_acc:
            best_acc, bad_epochs = acc_drop, 0
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       DROPOUT_MODEL_PATH)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping: no improvement for {patience} epochs")
                break

    # Compare both models on clean and on dropped validation sets
    loss_fn = nn.CrossEntropyLoss()
    metrics = {"epochs_trained": epochs_trained}
    for name, path in [("original", classifier.MODEL_PATH), ("dropout", DROPOUT_MODEL_PATH)]:
        m = classifier.load_model(path).to(device)
        metrics[name] = {"val_top1_clean": classifier.evaluate(m, Xva, yva, loss_fn)[1],
                         "val_top1_with_dropout": classifier.evaluate(m, Xva_drop, yva, loss_fn)[1]}
    print(json.dumps(metrics, indent=2))
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "classifier_dropout_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
