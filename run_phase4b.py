"""
Phase 4b: keypoint-dropout sweep on the random-pose benchmark.

Extra visible keypoints are hidden at random so that 4-6, 7-9 or 10+ remain visible
(features rebuilt from the reduced set). Methods:
    PnP only                         SQPnP, no guess
    Classifier + PnP (dropout clf)   classifier_pnp, models/classifier_dropout.pt
    Hybrid (dropout clf)             hybrid_pnp, models/classifier_dropout.pt
plus, for reference, classifier only (dropout clf) and classifier + PnP with the ORIGINAL
classifier. Nothing is tuned.

    python train_dropout.py     # first, creates models/classifier_dropout.pt
    python run_phase4b.py
"""

import dataclasses
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import classifier
import config
import dataset
import filters
import pnp
import simulator
from run_phase4 import RANDOM_SEED, errors, seed_stat, summarize
from train_dropout import DROPOUT_MODEL_PATH

NOISES = [2.0, 5.0]
SEEDS = range(5)
N_RANDOM = 2000
TARGETS = {"4-6": (4, 6), "7-9": (7, 9), "10+": (10, 14)}   # visible keypoints that remain
METHODS = ["PnP only", "Classifier + PnP (orig clf)", "Classifier + PnP (dropout clf)",
           "Hybrid (dropout clf)", "Classifier only (dropout clf)"]
PLOT_METHODS = METHODS[:4]
COLORS = dict(zip(METHODS, ["tab:orange", "tab:olive", "tab:green", "tab:purple", "tab:gray"]))


def reduce_visibility(obs, lo, hi, rng):
    """Copy of obs with extra keypoints hidden so about k in [lo, hi] stay visible (never adds any)."""
    k = rng.integers(lo, hi + 1)
    vis = obs.visible.copy()
    idx = np.flatnonzero(vis)
    if len(idx) > k:
        vis[rng.choice(idx, len(idx) - k, replace=False)] = False
    return dataclasses.replace(obs, visible=vis,
                               features=dataset.make_features(obs.uv, vis, obs.boresight))


def run_methods(models, observations):
    """-> {method: dict(est, valid, fallback, ms)} in the format run_phase4.errors/summarize expect."""
    T = len(observations)
    out = {m: dict(est=np.full((T, 6), np.nan), valid=np.ones(T, bool),
                   fallback=np.zeros(T, bool), ms=np.zeros(T)) for m in METHODS}
    for i, obs in enumerate(observations):
        coarse_o = classifier.predict_pose(models["orig"], obs.features[None])[0]
        coarse_d = classifier.predict_pose(models["dropout"], obs.features[None])[0]

        pose, _ = pnp.pnp_only(obs)
        if pose is None:
            out["PnP only"]["valid"][i] = False
        else:
            out["PnP only"]["est"][i] = pose

        for name, fn, coarse in [("Classifier + PnP (orig clf)", pnp.classifier_pnp, coarse_o),
                                 ("Classifier + PnP (dropout clf)", pnp.classifier_pnp, coarse_d),
                                 ("Hybrid (dropout clf)", pnp.hybrid_pnp, coarse_d)]:
            est, info = fn(obs, coarse)
            out[name]["est"][i] = est
            out[name]["fallback"][i] = info["fallback"]
        out["Classifier only (dropout clf)"]["est"][i] = coarse_d
    return out


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    models = {"orig": classifier.load_model(), "dropout": classifier.load_model(DROPOUT_MODEL_PATH)}
    true = simulator.sample_random_poses(N_RANDOM, np.random.default_rng(RANDOM_SEED))

    # per_seed[noise][group][method] -> list of summaries (one per seed)
    per_seed = {n: {g: {m: [] for m in METHODS} for g in TARGETS} for n in NOISES}
    group_n = {g: [] for g in TARGETS}
    for noise in NOISES:
        for seed in SEEDS:
            rng = np.random.default_rng(10_000 + 100 * seed + int(noise))
            base = [filters.make_observation(p, noise, rng) for p in true]
            for g, (lo, hi) in TARGETS.items():
                if g == "10+":      # only poses that really have >= 10 visible keypoints
                    keep = np.array([o.visible.sum() >= 10 for o in base])
                else:
                    keep = np.ones(len(base), bool)
                obs = [reduce_visibility(o, lo, hi, rng) for o, k in zip(base, keep) if k]
                res = run_methods(models, obs)
                for m in METHODS:
                    per_seed[noise][g][m].append(summarize(true[keep], res[m]))
                if noise == NOISES[0] and seed == 0:
                    group_n[g] = int(keep.sum())
            print(f"  noise {noise:g} px, seed {seed} done")

    # ---- table --------------------------------------------------------------
    print("\nKEYPOINT DROPOUT SWEEP (2000 random poses, mean over 5 seeds; fail = no solution or "
          f"pos > 10 m or att > 20 deg)")
    summary = {}
    for noise in NOISES:
        print(f"\nnoise = {noise:g} px")
        print(f"{'Visible':8s} {'Method':32s} {'pos mean':>9s} {'pos med':>8s} {'att mean':>9s} "
              f"{'att med':>8s} {'fail %':>7s} {'fallb %':>8s} {'no-sol':>7s}")
        summary[str(noise)] = {}
        for g in TARGETS:
            summary[str(noise)][g] = {"n_poses": group_n[g]}
            for m in METHODS:
                s = per_seed[noise][g][m]
                row = {k: seed_stat(s, k) for k in
                       ["pos_mean", "pos_median", "att_mean", "att_median", "fail_rate", "fallback_rate"]}
                row["n_no_solution_total"] = int(sum(x["n_no_solution"] for x in s))
                summary[str(noise)][g][m] = row
                print(f"{g:8s} {m:32s} {row['pos_mean'][0]:9.2f} {row['pos_median'][0]:8.2f} "
                      f"{row['att_mean'][0]:9.2f} {row['att_median'][0]:8.2f} "
                      f"{100 * row['fail_rate'][0]:7.1f} {100 * row['fallback_rate'][0]:8.1f} "
                      f"{row['n_no_solution_total']:7d}")

    # ---- figure ---------------------------------------------------------------
    panels = [("pos_median", "Median position error [m]", 1), ("att_median", "Median attitude error [deg]", 1),
              ("fail_rate", "Gross-failure rate [%]", 100), ("fallback_rate", "Fallback rate [%]", 100)]
    fig, axes = plt.subplots(len(NOISES), 4, figsize=(17, 7.5))
    width = 0.2
    for r, noise in enumerate(NOISES):
        for ax, (key, label, scale) in zip(axes[r], panels):
            for k, m in enumerate(PLOT_METHODS):
                stats = [summary[str(noise)][g][m][key] for g in TARGETS]
                ax.bar(np.arange(3) + (k - 1.5) * width, [scale * s[0] for s in stats], width,
                       yerr=[scale * s[1] for s in stats], capsize=2, color=COLORS[m], label=m)
            ax.set_xticks(range(3))
            ax.set_xticklabels(list(TARGETS))
            ax.set_xlabel("Visible keypoints")
            ax.set_ylabel(label)
            ax.set_title(f"{noise:g} px noise", fontsize=9)
            ax.grid(alpha=0.3, axis="y")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Keypoint dropout sweep (random poses, mean over 5 seeds, error bars = std)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "pnp_dropout.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(config.RESULTS_DIR, "pnp_dropout_metrics.json"), "w") as f:
        json.dump({"settings": {"noises_px": NOISES, "n_seeds": len(SEEDS), "n_random": N_RANDOM,
                                "random_seed": RANDOM_SEED, "targets": TARGETS},
                   "results": summary}, f, indent=2)
    print("\nSaved results/pnp_dropout_metrics.json, pnp_dropout.png")


if __name__ == "__main__":
    main()
