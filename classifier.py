"""
Pose classifier (paper Section IV-D): a small MLP that maps the 45 keypoint
features to one of the 4800 grid cells (x, y, z, roll).

Usage:
    python classifier.py                  # train, evaluate, save figures/metrics
    python classifier.py --epochs 40

Later phases import: load_model, predict_probs, predict_label, predict_pose.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import config
import dataset

MODEL_PATH = os.path.join("models", "classifier.pt")
ERROR_SAMPLES_PATH = os.path.join("models", "error_samples.npy")
N_FEATURES = config.N_FEATURES   # 45


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class PoseClassifier(nn.Module):
    """
    45 -> 100 -> 100 -> 4800. Outputs raw logits (softmax is in the loss).

    The keypoint features are small numbers (std ~0.05-0.1), so the model
    standardises its input with a mean/std computed on the training set. They
    are stored as buffers, so they are saved and loaded with the weights and
    callers can keep passing raw features.
    """

    def __init__(self, n_in=N_FEATURES, n_hidden=100, n_out=config.N_LABELS):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Inference API (used by later phases)
# ---------------------------------------------------------------------------
def load_model(path=MODEL_PATH):
    """Load trained weights; the model is returned on the CPU in eval mode."""
    model = PoseClassifier()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict_probs(model, X):
    """(n, 45) features -> (n, 4800) class probabilities (numpy)."""
    X = torch.as_tensor(np.atleast_2d(X), dtype=torch.float32)
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(X.to(device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def predict_label(model, X):
    """(n, 45) features -> (n,) most likely grid labels."""
    return predict_probs(model, X).argmax(axis=1)


def predict_pose(model, X):
    """(n, 45) features -> (n, 6) coarse poses (grid point, pitch = yaw = 0)."""
    return dataset.label_to_pose(predict_label(model, X))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def load_split(name):
    d = np.load(os.path.join(config.DATA_DIR, f"{name}.npz"))
    return d["X"].astype(np.float32), d["y"].astype(np.int64), d["poses"]


def evaluate(model, X, y, loss_fn, batch=4096):
    """Returns (mean loss, accuracy) over a tensor dataset already on device."""
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            logits = model(X[i:i + batch])
            total_loss += loss_fn(logits, y[i:i + batch]).item() * len(logits)
            correct += (logits.argmax(1) == y[i:i + batch]).sum().item()
    return total_loss / len(X), correct / len(X)


def train(epochs=40, batch_size=256, lr=1e-3, patience=5):
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = get_device()
    print(f"Device: {device}")

    Xtr, ytr, _ = load_split("train")
    Xva, yva, _ = load_split("val")
    Xtr, ytr, Xva, yva = (torch.as_tensor(a).to(device) for a in (Xtr, ytr, Xva, yva))

    model = PoseClassifier().to(device)
    model.mean.copy_(Xtr.mean(0))
    model.std.copy_(Xtr.std(0).clamp_min(1e-6))   # avoid dividing by ~0
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(config.SEED)   # shuffling order

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, best_state, bad_epochs = -1.0, None, 0
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen).to(device)
        running = 0.0
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        train_loss = running / len(Xtr)
        val_loss, val_acc = evaluate(model, Xva, yva, loss_fn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch:2d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
              f"| val acc {val_acc:.4f} | {time.time() - t0:.1f} s")

        # Keep the best checkpoint; stop if no improvement for `patience` epochs
        if val_acc > best_acc:
            best_acc, bad_epochs = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, MODEL_PATH)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping: no improvement for {patience} epochs")
                break

    print(f"Best val accuracy {best_acc:.4f} -> saved {MODEL_PATH}")
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def compute_metrics(model):
    """Evaluate the saved model on the validation set. Returns (metrics, arrays)."""
    X, y, poses = load_split("val")
    probs = predict_probs(model, X)
    pred = probs.argmax(axis=1)

    top1 = float((pred == y).mean())
    top5_labels = np.argsort(-probs, axis=1)[:, :5]
    top5 = float((top5_labels == y[:, None]).any(axis=1).mean())

    # Neighbour accuracy: within one grid step in every dimension
    pred_idx = np.array(np.unravel_index(pred, config.GRID_SHAPE))
    true_idx = np.array(np.unravel_index(y, config.GRID_SHAPE))
    neighbour = float((np.abs(pred_idx - true_idx) <= 1).all(axis=0).mean())

    pred_pose = dataset.label_to_pose(pred)
    err_true = poses - pred_pose                       # true - predicted bin pose
    mae = np.abs(err_true).mean(axis=0)

    metrics = {
        "n_val": int(len(y)),
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "neighbour_accuracy": neighbour,
        "mae_x_m": float(mae[0]),
        "mae_y_m": float(mae[1]),
        "mae_z_m": float(mae[2]),
        "mae_roll_deg": float(mae[3]),
    }
    return metrics, {"pred": pred, "y": y, "poses": poses, "pred_pose": pred_pose,
                     "err_true": err_true}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
LABELS = ["x [m]", "y [m]", "z [m]", "roll [deg]"]


def plot_error_hists(err, title, path, log=False, bin_widths=None):
    """2x2 histograms of err[:, :4] (x, y, z, roll)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for d, ax in enumerate(axes.ravel()):
        ax.hist(err[:, d], bins=60, color="tab:blue", edgecolor="none", log=log)
        ax.set_xlabel(LABELS[d])
        ax.set_ylabel("count")
        ax.axvline(0, color="k", lw=0.5)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def plot_label_error_hists(err, path):
    """
    Predicted bin - correct bin, in metres/degrees. Bins are centred on the
    grid spacing so each grid step gets its own bar: a tall bar at 0 plus
    small bars at +/- one step for adjacent-bin mistakes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for d, ax in enumerate(axes.ravel()):
        step = config.GRID[d][1] - config.GRID[d][0]
        n_steps = len(config.GRID[d]) - 1
        edges = (np.arange(-n_steps, n_steps + 2) - 0.5) * step
        ax.hist(err[:, d], bins=edges, color="tab:blue", edgecolor="none")
        ax.set_xlabel(LABELS[d])
        ax.set_ylabel("count")
    fig.suptitle("Predicted bin pose - correct bin pose (validation set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def plot_training_curve(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], "o-", label="train")
    ax1.plot(epochs, history["val_loss"], "o-", label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")
    ax1.legend()
    ax2.plot(epochs, np.array(history["val_acc"]) * 100, "o-", color="tab:green")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation top-1 accuracy [%]")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    history = train(epochs=args.epochs)

    model = load_model()
    metrics, a = compute_metrics(model)

    # Error distribution for the particle filter's initialisation (Phase 3)
    np.save(ERROR_SAMPLES_PATH, a["err_true"])
    print(f"Saved {ERROR_SAMPLES_PATH} {a['err_true'].shape}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    r = config.RESULTS_DIR
    plot_error_hists(-a["err_true"],
                     "Predicted bin pose - true pose (validation set)",
                     os.path.join(r, "classifier_error_vs_true.png"))
    correct_pose = dataset.label_to_pose(a["y"])
    plot_label_error_hists(a["pred_pose"] - correct_pose,
                           os.path.join(r, "classifier_error_vs_label.png"))
    plot_training_curve(history, os.path.join(r, "training_curve.png"))

    metrics["epochs_trained"] = len(history["val_acc"])
    with open(os.path.join(r, "classifier_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nValidation metrics")
    print(f"  top-1 accuracy      {metrics['top1_accuracy'] * 100:6.2f} %   (paper: 54 %)")
    print(f"  top-5 accuracy      {metrics['top5_accuracy'] * 100:6.2f} %")
    print(f"  neighbour accuracy  {metrics['neighbour_accuracy'] * 100:6.2f} %")
    print(f"  MAE x, y, z         {metrics['mae_x_m']:.2f}, {metrics['mae_y_m']:.2f}, "
          f"{metrics['mae_z_m']:.2f} m")
    print(f"  MAE roll            {metrics['mae_roll_deg']:.2f} deg")


if __name__ == "__main__":
    main()
