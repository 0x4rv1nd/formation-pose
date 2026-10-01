"""
Phase 4 experiments: classifier only vs PnP only vs classifier + PnP, single frame.

Experiment 1: the 10 s approach trajectory (+ Phase 3 Config B as a reference)
Experiment 2: 2000 random poses, broken down by the number of visible keypoints

For every keypoint noise level the classifier sees the same noisy keypoints as
the PnP methods. Nothing is retrained and nothing is tuned.
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import classifier
import config
import filters
import pnp
import simulator
from run_phase3 import attitude_error, position_error

NOISES = [0.0, 1.0, 2.0, 5.0]       # keypoint noise std [px]
SEEDS = range(5)
N_RANDOM = 2000
RANDOM_SEED = 12345                 # training uses seed 0, validation seed 1
GROSS_POS_M, GROSS_ATT_DEG = 10.0, 20.0   # beyond this an estimate counts as a failure
METHODS = ["Classifier only", "PnP only", "Classifier + PnP"]
PF_NAME = "Phase 3 Config B"
COLORS = {"Classifier only": "tab:gray", "PnP only": "tab:orange",
          "Classifier + PnP": "tab:green", PF_NAME: "tab:blue"}
GROUPS = {"4-6": (4, 6), "7-9": (7, 9), "10-14": (10, 14)}


# ---------------------------------------------------------------------------
# Running the three methods on a list of observations
# ---------------------------------------------------------------------------
def run_methods(model, observations):
    """
    Per frame: classifier, PnP only, classifier + PnP.
    Returns {method: dict(est (T,6) with NaN rows for 'no solution', valid (T,),
    fallback (T,), ms (T,))}. Runtime of 'Classifier + PnP' includes the classifier call.
    """
    T = len(observations)
    out = {m: dict(est=np.full((T, 6), np.nan), valid=np.ones(T, bool),
                   fallback=np.zeros(T, bool), ms=np.zeros(T)) for m in METHODS}
    for i, obs in enumerate(observations):
        t0 = time.perf_counter()
        coarse = classifier.predict_pose(model, obs.features[None])[0]
        t_cls = time.perf_counter() - t0

        t0 = time.perf_counter()
        pose, _ = pnp.pnp_only(obs)
        t_pnp = time.perf_counter() - t0

        t0 = time.perf_counter()
        refined, info = pnp.classifier_pnp(obs, coarse)
        t_ref = time.perf_counter() - t0

        out["Classifier only"]["est"][i] = coarse
        out["Classifier only"]["ms"][i] = 1000 * t_cls
        if pose is None:
            out["PnP only"]["valid"][i] = False
        else:
            out["PnP only"]["est"][i] = pose
        out["PnP only"]["ms"][i] = 1000 * t_pnp
        out["Classifier + PnP"]["est"][i] = refined
        out["Classifier + PnP"]["fallback"][i] = info["fallback"]
        out["Classifier + PnP"]["ms"][i] = 1000 * (t_cls + t_ref)
    return out


def errors(true, r):
    """Position (m) and attitude (deg) error per frame; NaN where there is no solution."""
    pos = np.full(len(true), np.nan)
    att = np.full(len(true), np.nan)
    ok = r["valid"]
    if ok.any():
        pos[ok] = position_error(true[ok], r["est"][ok])
        att[ok] = attitude_error(true[ok], r["est"][ok])
    return pos, att


def _nan_stat(fn, x):
    x = x[np.isfinite(x)]
    return float(fn(x)) if len(x) else float("nan")


def summarize(true, r, mask=None):
    """Mean / median errors over frames with a solution, failure + fallback rates, runtime."""
    pos, att = errors(true, r)
    fb, ms = r["fallback"], r["ms"]
    if mask is not None:
        pos, att, fb, ms = pos[mask], att[mask], fb[mask], ms[mask]
    no_solution = ~np.isfinite(pos)
    fail = no_solution | (pos > GROSS_POS_M) | (att > GROSS_ATT_DEG)
    return {"n": int(len(pos)),
            "pos_mean": _nan_stat(np.mean, pos), "pos_median": _nan_stat(np.median, pos),
            "att_mean": _nan_stat(np.mean, att), "att_median": _nan_stat(np.median, att),
            "fail_rate": float(fail.mean()) if len(pos) else float("nan"),
            "n_no_solution": int(no_solution.sum()),
            "fallback_rate": float(fb.mean()) if len(pos) else float("nan"),
            "ms": float(ms.mean()) if len(pos) else float("nan")}


def seed_stat(per_seed, key):
    v = np.array([s[key] for s in per_seed])
    return float(np.nanmean(v)), float(np.nanstd(v))


# ---------------------------------------------------------------------------
# Experiment 1: approach trajectory
# ---------------------------------------------------------------------------
def experiment_trajectory(model):
    times, true = simulator.approach_trajectory()
    per_seed = {n: {m: [] for m in METHODS + [PF_NAME]} for n in NOISES}
    curves = {}   # (noise, method) -> (pos, att) per step for seed 0
    for noise in NOISES:
        for seed in SEEDS:
            # Same observations the particle filter gets from run_filter(seed=seed)
            rng = np.random.default_rng(seed)
            observations = [filters.make_observation(p, noise, rng) for p in true]
            res = run_methods(model, observations)

            pf = filters.ParticleFilter(model, 5000, filters.alpha_every_step(0.9), seed=seed)
            est, runtime = filters.run_filter(times, true, pf, noise_px=noise, seed=seed)
            res[PF_NAME] = dict(est=est, valid=np.ones(len(times), bool),
                                fallback=np.zeros(len(times), bool), ms=1000 * runtime)

            for m, r in res.items():
                per_seed[noise][m].append(summarize(true, r))
                if seed == 0:
                    curves[(noise, m)] = errors(true, r)
        print(f"  trajectory: noise {noise} px done")
    return times, per_seed, curves


# ---------------------------------------------------------------------------
# Experiment 2: random-pose benchmark
# ---------------------------------------------------------------------------
def experiment_random(model):
    true = simulator.sample_random_poses(N_RANDOM, np.random.default_rng(RANDOM_SEED))
    out = {}
    for k, noise in enumerate(NOISES):
        rng = np.random.default_rng(RANDOM_SEED + 1 + k)
        observations = [filters.make_observation(p, noise, rng) for p in true]
        n_vis = np.array([o.visible.sum() for o in observations])
        res = run_methods(model, observations)
        groups = {"all": np.ones(len(true), bool), "<4": n_vis < 4}
        groups.update({name: (n_vis >= lo) & (n_vis <= hi) for name, (lo, hi) in GROUPS.items()})
        out[noise] = {g: {m: summarize(true, res[m], mask) for m in METHODS}
                      for g, mask in groups.items()}
        print(f"  random poses: noise {noise} px done")
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def print_trajectory_table(per_seed):
    print("\nEXPERIMENT 1: approach trajectory (101 steps x 5 seeds per noise level)")
    print("pos/att mean = mean +/- std over seeds of the time-averaged error; no-solution frames "
          "excluded from errors.")
    for noise in NOISES:
        print(f"\nnoise = {noise:g} px")
        print(f"{'Method':18s} {'pos mean [m]':>15s} {'pos med':>8s} {'att mean [deg]':>16s} "
              f"{'att med':>8s} {'fail %':>7s} {'fallb %':>8s} {'no-sol':>7s} {'ms':>7s}")
        for m in METHODS + [PF_NAME]:
            s = per_seed[noise][m]
            pm, ps = seed_stat(s, "pos_mean")
            am, a_s = seed_stat(s, "att_mean")
            print(f"{m:18s} {pm:7.2f} +/-{ps:5.2f} {seed_stat(s, 'pos_median')[0]:8.2f} "
                  f"{am:8.2f} +/-{a_s:5.2f} {seed_stat(s, 'att_median')[0]:8.2f} "
                  f"{100 * seed_stat(s, 'fail_rate')[0]:7.1f} "
                  f"{100 * seed_stat(s, 'fallback_rate')[0]:8.1f} "
                  f"{sum(x['n_no_solution'] for x in s):7d} {seed_stat(s, 'ms')[0]:7.2f}")


def print_random_table(rand):
    print(f"\nEXPERIMENT 2: {N_RANDOM} random poses per noise level, by number of visible keypoints")
    for noise in NOISES:
        print(f"\nnoise = {noise:g} px")
        print(f"{'Group':6s} {'n':>5s} {'Method':18s} {'pos mean':>9s} {'pos med':>8s} "
              f"{'att mean':>9s} {'att med':>8s} {'fail %':>7s} {'fallb %':>8s} {'no-sol':>7s} {'ms':>6s}")
        for g, by_m in rand[noise].items():
            for m in METHODS:
                s = by_m[m]
                print(f"{g:6s} {s['n']:5d} {m:18s} {s['pos_mean']:9.2f} {s['pos_median']:8.2f} "
                      f"{s['att_mean']:9.2f} {s['att_median']:8.2f} {100 * s['fail_rate']:7.1f} "
                      f"{100 * s['fallback_rate']:8.1f} {s['n_no_solution']:7d} {s['ms']:6.2f}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_trajectory(times, curves):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, j, label in [(axes[0], 0, "Position error [m]"), (axes[1], 1, "Attitude error [deg]")]:
        for m in METHODS + [PF_NAME]:
            ax.plot(times, curves[(2.0, m)][j], color=COLORS[m], label=m, lw=1.3)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("Time [s]")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Approach trajectory, 2 px keypoint noise (seed 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "pnp_trajectory_errors.png"), dpi=120)
    plt.close(fig)


def fig_noise_sweep(per_seed):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, label in [(axes[0], "pos_mean", "Mean position error [m]"),
                           (axes[1], "att_mean", "Mean attitude error [deg]")]:
        for m in METHODS + [PF_NAME]:
            stats = np.array([seed_stat(per_seed[n][m], key) for n in NOISES])
            ax.errorbar(NOISES, stats[:, 0], yerr=stats[:, 1], marker="o", capsize=3,
                        color=COLORS[m], label=m)
        ax.set_xlabel("Keypoint noise [px]")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Approach trajectory: error vs keypoint noise (mean, error bars = std over 5 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "pnp_noise_sweep.png"), dpi=120)
    plt.close(fig)


def fig_visibility(rand):
    names = list(GROUPS)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    panels = [("pos_median", "Median position error [m]", 1), ("att_median", "Median attitude error [deg]", 1),
              ("fail_rate", "Failure rate [%]", 100)]
    width = 0.27
    for ax, (key, label, scale) in zip(axes, panels):
        for k, m in enumerate(METHODS):
            vals = [scale * rand[2.0][g][m][key] for g in names]
            ax.bar(np.arange(len(names)) + (k - 1) * width, vals, width, color=COLORS[m], label=m)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f"{g}\n(n={rand[2.0][g]['Classifier only']['n']})" for g in names])
        ax.set_xlabel("Visible keypoints")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("Random poses, 2 px keypoint noise")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "pnp_by_visibility.png"), dpi=120)
    plt.close(fig)


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    model = classifier.load_model()

    print("Experiment 1 ...")
    times, per_seed, curves = experiment_trajectory(model)
    print("Experiment 2 ...")
    rand = experiment_random(model)

    print_trajectory_table(per_seed)
    print_random_table(rand)

    fig_trajectory(times, curves)
    fig_noise_sweep(per_seed)
    fig_visibility(rand)

    metrics = {"settings": {"noises_px": NOISES, "n_seeds": len(SEEDS), "n_random": N_RANDOM,
                            "random_seed": RANDOM_SEED, "gross_pos_m": GROSS_POS_M,
                            "gross_att_deg": GROSS_ATT_DEG},
               "trajectory": {str(n): per_seed[n] for n in NOISES},
               "random": {str(n): rand[n] for n in NOISES}}
    with open(os.path.join(config.RESULTS_DIR, "pnp_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved results/pnp_metrics.json, pnp_trajectory_errors.png, pnp_noise_sweep.png, "
          "pnp_by_visibility.png")


if __name__ == "__main__":
    main()
