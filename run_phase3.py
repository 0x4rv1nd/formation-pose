"""
Phase 3 experiments: baseline particle filter vs classifier only on the approach
trajectory (no keypoint noise), mirroring the paper's Figs. 9 and 10.

    Config A: N = 1000, alpha = 0.9 on every 10th step (1 otherwise)
    Config B: N = 5000, alpha = 0.9 on every step
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

import classifier
import config
import filters
import simulator

SEEDS = range(5)
CONFIGS = {
    "Config A (N=1000, alpha=0.9 every 10th step)":
        dict(n_particles=1000, alpha_schedule=filters.alpha_every_nth(0.9, 10)),
    "Config B (N=5000, alpha=0.9 every step)":
        dict(n_particles=5000, alpha_schedule=filters.alpha_every_step(0.9)),
}


def position_error(true, est):
    """Euclidean distance (m) per step."""
    return np.linalg.norm(true[:, :3] - est[:, :3], axis=1)


def attitude_error(true, est):
    """Geodesic angle (deg) between the true and estimated rotations per step."""
    r_true = Rotation.from_euler("ZYX", true[:, [5, 4, 3]], degrees=True)
    r_est = Rotation.from_euler("ZYX", est[:, [5, 4, 3]], degrees=True)
    return np.degrees((r_true.inv() * r_est).magnitude())


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    model = classifier.load_model()
    times, true = simulator.approach_trajectory()

    # Classifier only: predict_pose on each step's features (pitch = yaw = 0, grid bins)
    obs = [filters.make_observation(p) for p in true]
    cls_est = classifier.predict_pose(model, np.stack([o.features for o in obs]))
    runs = {"Classifier only": {"est": [cls_est],
                                "pos": [position_error(true, cls_est)],
                                "att": [attitude_error(true, cls_est)],
                                "ms": [0.0]}}

    for name, kwargs in CONFIGS.items():
        runs[name] = {"est": [], "pos": [], "att": [], "ms": []}
        for seed in SEEDS:
            pf = filters.ParticleFilter(model, seed=seed, **kwargs)
            est, runtime = filters.run_filter(times, true, pf,
                                              noise_px=config.KEYPOINT_NOISE_PX, seed=seed)
            runs[name]["est"].append(est)
            runs[name]["pos"].append(position_error(true, est))
            runs[name]["att"].append(attitude_error(true, est))
            runs[name]["ms"].append(1000 * runtime.mean())
            print(f"{name}, seed {seed}: pos {runs[name]['pos'][-1].mean():.2f} m, "
                  f"att {runs[name]['att'][-1].mean():.2f} deg, {runs[name]['ms'][-1]:.1f} ms/step")

    # Mean +/- std over seeds of the time-averaged errors
    metrics = {}
    print(f"\n{'Method':48s} {'pos err [m]':>14s} {'att err [deg]':>14s} {'ms/step':>9s}")
    for name, r in runs.items():
        pos = np.array([e.mean() for e in r["pos"]])
        att = np.array([e.mean() for e in r["att"]])
        metrics[name] = {"pos_err_m_mean": pos.mean(), "pos_err_m_std": pos.std(),
                         "att_err_deg_mean": att.mean(), "att_err_deg_std": att.std(),
                         "runtime_ms_per_step": float(np.mean(r["ms"])),
                         "n_seeds": len(pos)}
        print(f"{name[:48]:48s} {pos.mean():6.2f} +/-{pos.std():5.2f} "
              f"{att.mean():6.2f} +/-{att.std():5.2f} {np.mean(r['ms']):9.1f}")
    with open(os.path.join(config.RESULTS_DIR, "pf_baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Figure 1: errors vs time (seed 0)
    names = list(runs)
    colors = dict(zip(names, ["tab:gray", "tab:blue", "tab:red"]))
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, key, label in [(axes[0], "pos", "Position error [m]"),
                           (axes[1], "att", "Attitude error [deg]")]:
        for name in names:
            ax.plot(times, runs[name][key][0], color=colors[name], label=name, lw=1.3)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("Time [s]")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Baseline particle filter vs classifier only (seed 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "pf_baseline_errors.png"), dpi=120)
    plt.close(fig)

    # Figure 2: true vs estimated x, y, z for Config B (seed 0)
    est_b = runs[names[2]]["est"][0]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, i, label in zip(axes, range(3), ["x [m]", "y [m]", "z [m]"]):
        ax.plot(times, true[:, i], "k-", label="true")
        ax.plot(times, est_b[:, i], "r--", label="Config B estimate")
        ax.plot(times, cls_est[:, i], ":", color="tab:gray", label="classifier only")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[2].set_xlabel("Time [s]")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Position tracking, Config B (seed 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "pf_baseline_trajectory.png"), dpi=120)
    plt.close(fig)
    print("\nSaved results/pf_baseline_metrics.json, pf_baseline_errors.png, "
          "pf_baseline_trajectory.png")


if __name__ == "__main__":
    main()
