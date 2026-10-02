"""
Phase 5 experiments: filtering on top of PnP, on the approach trajectory.

Methods:
    Phase 3 PF (Config B)   baseline particle filter, N = 5000, alpha = 0.9 every step
    PnP alone               single-frame hybrid PnP, no filter (missing when there is no solution)
    Improved PF             Gaussian likelihood + particles drawn around the PnP pose, N = 1000
    Kalman filter           linear KF on the PnP pose, chi-squared gating

Experiment A: normal run.
Experiment B: measurement outage, all keypoints hidden for 4.0 <= t < 5.0 s.
Noise levels 0, 1, 2, 5 px, 5 seeds each. R is calibrated first on a separate set of random poses.
Nothing is tuned.

    python run_phase5.py
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

NOISES = [0.0, 1.0, 2.0, 5.0]
SEEDS = range(5)
OUTAGE = (4.0, 5.0)                 # s
AFTER = (5.0, 6.0)                  # the 1 s after the outage
PF_BASELINE = "Phase 3 PF (Config B)"
PNP = "PnP alone"
PF_IMPROVED = "Improved PF"
KF = "Kalman filter"
METHODS = [PF_BASELINE, PNP, PF_IMPROVED, KF]
COLORS = {PF_BASELINE: "tab:blue", PNP: "tab:orange", PF_IMPROVED: "tab:purple", KF: "tab:green"}


# ---------------------------------------------------------------------------
# Errors (NaN where the method has no estimate)
# ---------------------------------------------------------------------------
def errors(true, est):
    """Position error (m) and attitude geodesic error (deg) per step; NaN rows stay NaN."""
    pos = np.full(len(true), np.nan)
    att = np.full(len(true), np.nan)
    ok = np.all(np.isfinite(est), axis=1)
    if ok.any():
        pos[ok] = np.linalg.norm(true[ok, :3] - est[ok, :3], axis=1)
        r_true = Rotation.from_euler("ZYX", true[ok][:, [5, 4, 3]], degrees=True)
        r_est = Rotation.from_euler("ZYX", est[ok][:, [5, 4, 3]], degrees=True)
        att[ok] = np.degrees((r_true.inv() * r_est).magnitude())
    return pos, att


def window(times, lo, hi):
    return (times >= lo - 1e-9) & (times < hi - 1e-9)


def stat(fn, x):
    x = x[np.isfinite(x)]
    return float(fn(x)) if len(x) else float("nan")


def make_filter(name, model, noise, R_ref, seed):
    if name == PF_BASELINE:
        return filters.ParticleFilter(model, 5000, filters.alpha_every_step(0.9), seed=seed)
    if name == PNP:
        return filters.PnPOnly(model)
    if name == PF_IMPROVED:
        return filters.ImprovedParticleFilter(model, 1000, filters.alpha_every_step(0.9),
                                              noise, R_ref, seed=seed)
    return filters.KalmanFilter(model, R_ref)


def run_one(name, model, times, true, noise, R_ref, seed, outage):
    """One filter, one seed: all methods with the same seed see identical observations."""
    filt = make_filter(name, model, noise, R_ref, seed)
    est, runtime = filters.run_filter(times, true, filt, noise_px=noise, seed=seed, outage=outage)
    pos, att = errors(true, est)
    return {"est": est, "pos": pos, "att": att, "ms": 1000 * runtime.mean(),
            "gated": getattr(filt, "n_gated", None),
            "n_missing_steps": int((~np.isfinite(pos)).sum())}


def mean_std(values):
    v = np.array(values, dtype=float)
    return (float(np.nanmean(v)), float(np.nanstd(v))) if np.isfinite(v).any() else (float("nan"),) * 2


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    model = classifier.load_model()
    times, true = simulator.approach_trajectory()

    # ---- calibrate R on a separate set of random poses --------------------------
    print("Calibrating PnP measurement noise (1000 random poses per noise level)...")
    calib = {}
    for noise in NOISES:
        calib[noise] = filters.calibrate_pnp_noise(model, noise)
        c = calib[noise]
        print(f"  {noise:g} px: std at {c['ref_range_m']:g} m = "
              + ", ".join(f"{s:.3f}" for s in c["std"])
              + f"  (used {c['n_used']}, no solution {c['n_no_solution']}, gross {c['n_gross_failures']})")
    with open(os.path.join(config.RESULTS_DIR, "kf_noise_calibration.json"), "w") as f:
        json.dump({"settings": {"calibration_seed": filters.CALIB_SEED, "n_poses": filters.CALIB_N,
                                "ref_range_m": filters.REF_RANGE_M,
                                "std_floor_m_deg": [filters.R_FLOOR_POS_M, filters.R_FLOOR_ATT_DEG],
                                "state_order": "x, y, z (m), roll, pitch, yaw (deg)"},
                   "by_noise_px": {str(n): c for n, c in calib.items()}}, f, indent=2)

    # ---- run everything -----------------------------------------------------------
    runs = {}   # (experiment, noise, method) -> list of run dicts, one per seed
    for exp, outage in [("A", None), ("B", OUTAGE)]:
        for noise in NOISES:
            R_ref = np.array(calib[noise]["cov"])
            for m in METHODS:
                runs[(exp, noise, m)] = [run_one(m, model, times, true, noise, R_ref, s, outage)
                                         for s in SEEDS]
            print(f"  experiment {exp}, noise {noise:g} px done")

    # ---- metrics ------------------------------------------------------------------
    out_win, after_win = window(times, *OUTAGE), window(times, *AFTER)
    metrics = {"A": {}, "B": {}}
    for noise in NOISES:
        for m in METHODS:
            ra = runs[("A", noise, m)]
            rb = runs[("B", noise, m)]
            a = {"pos_mean": mean_std([stat(np.mean, r["pos"]) for r in ra]),
                 "pos_median": mean_std([stat(np.median, r["pos"]) for r in ra]),
                 "att_mean": mean_std([stat(np.mean, r["att"]) for r in ra]),
                 "att_median": mean_std([stat(np.median, r["att"]) for r in ra]),
                 "gate_rejections": mean_std([r["gated"] for r in ra]) if ra[0]["gated"] is not None else None,
                 "missing_steps": mean_std([r["n_missing_steps"] for r in ra]),
                 "ms_per_step": mean_std([r["ms"] for r in ra])}
            b = {"pos_mean_all": mean_std([stat(np.mean, r["pos"]) for r in rb]),
                 "att_mean_all": mean_std([stat(np.mean, r["att"]) for r in rb]),
                 "gate_rejections": mean_std([r["gated"] for r in rb]) if rb[0]["gated"] is not None else None,
                 "missing_steps": mean_std([r["n_missing_steps"] for r in rb])}
            for tag, w in [("outage", out_win), ("after", after_win)]:
                b[f"pos_mean_{tag}"] = mean_std([stat(np.mean, r["pos"][w]) for r in rb])
                b[f"pos_median_{tag}"] = mean_std([stat(np.median, r["pos"][w]) for r in rb])
                b[f"att_mean_{tag}"] = mean_std([stat(np.mean, r["att"][w]) for r in rb])
                b[f"att_median_{tag}"] = mean_std([stat(np.median, r["att"][w]) for r in rb])
                b[f"missing_{tag}"] = mean_std([int((~np.isfinite(r["pos"][w])).sum()) for r in rb])
            metrics["A"][f"{noise:g}"] = metrics["A"].get(f"{noise:g}", {})
            metrics["A"][f"{noise:g}"][m] = a
            metrics["B"][f"{noise:g}"] = metrics["B"].get(f"{noise:g}", {})
            metrics["B"][f"{noise:g}"][m] = b

    # ---- tables -----------------------------------------------------------------
    def fmt(v, nd=2):
        return "   -  " if v is None or not np.isfinite(v[0]) else f"{v[0]:6.{nd}f}"

    print("\nEXPERIMENT A: normal run, 5 seeds, mean over seeds of per-run statistics")
    print(f"{'noise':>5s} {'Method':24s} {'pos mean':>8s} {'pos med':>8s} {'att mean':>8s} "
          f"{'att med':>8s} {'gated':>6s} {'missing':>8s} {'ms/step':>8s}")
    for noise in NOISES:
        for m in METHODS:
            a = metrics["A"][f"{noise:g}"][m]
            print(f"{noise:5g} {m:24s} {fmt(a['pos_mean']):>8s} {fmt(a['pos_median']):>8s} "
                  f"{fmt(a['att_mean']):>8s} {fmt(a['att_median']):>8s} "
                  f"{fmt(a['gate_rejections'], 1):>6s} {fmt(a['missing_steps'], 1):>8s} "
                  f"{fmt(a['ms_per_step'], 1):>8s}")

    print("\nEXPERIMENT B: outage 4.0-5.0 s (10 steps), 'after' = 5.0-6.0 s; mean position / attitude error")
    print(f"{'noise':>5s} {'Method':24s} {'pos in':>8s} {'att in':>8s} {'miss in':>8s} "
          f"{'pos after':>9s} {'att after':>9s} {'miss aft':>8s} {'gated':>6s}")
    for noise in NOISES:
        for m in METHODS:
            b = metrics["B"][f"{noise:g}"][m]
            print(f"{noise:5g} {m:24s} {fmt(b['pos_mean_outage']):>8s} {fmt(b['att_mean_outage']):>8s} "
                  f"{fmt(b['missing_outage'], 1):>8s} {fmt(b['pos_mean_after']):>9s} "
                  f"{fmt(b['att_mean_after']):>9s} {fmt(b['missing_after'], 1):>8s} "
                  f"{fmt(b['gate_rejections'], 1):>6s}")

    with open(os.path.join(config.RESULTS_DIR, "filter_metrics.json"), "w") as f:
        json.dump({"settings": {"noises_px": NOISES, "n_seeds": len(SEEDS), "outage_s": OUTAGE,
                                "after_window_s": AFTER,
                                "note": "each stat is [mean over seeds, std over seeds] of a per-run statistic"},
                   "experiment_A": metrics["A"], "experiment_B": metrics["B"]}, f, indent=2)

    # ---- figures ------------------------------------------------------------------
    # 1. errors vs time at 2 px (seed 0)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, key, label in [(axes[0], "pos", "Position error [m]"), (axes[1], "att", "Attitude error [deg]")]:
        for m in METHODS:
            ax.plot(times, runs[("A", 2.0, m)][0][key], color=COLORS[m], label=m, lw=1.3)
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[1].set_xlabel("Time [s]")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Errors on the approach trajectory, 2 px noise (seed 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "filter_errors_2px.png"), dpi=120)
    plt.close(fig)

    # 2. noise sweep
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, label in [(axes[0], "pos_mean", "Mean position error [m]"),
                           (axes[1], "att_mean", "Mean attitude error [deg]")]:
        for k, m in enumerate(METHODS):
            vals = [metrics["A"][f"{n:g}"][m][key] for n in NOISES]
            x = np.arange(len(NOISES)) + (k - 1.5) * 0.06
            ax.errorbar(x, [v[0] for v in vals], yerr=[v[1] for v in vals], marker="o", capsize=3,
                        color=COLORS[m], label=m)
        ax.set_xticks(range(len(NOISES)))
        ax.set_xticklabels([f"{n:g}" for n in NOISES])
        ax.set_xlabel("Keypoint noise [px]")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    fig.suptitle("Noise sweep (mean over 5 seeds, error bars = std over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "filter_noise_sweep.png"), dpi=120)
    plt.close(fig)

    # 3. outage (2 px, seed 0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in METHODS:
        ax.plot(times, runs[("B", 2.0, m)][0]["pos"], color=COLORS[m], label=m, lw=1.3)
    ax.axvspan(*OUTAGE, color="gray", alpha=0.25, label="outage (no keypoints)")
    ax.set_yscale("log")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position error [m]")
    ax.set_title("Measurement outage, 2 px noise (seed 0); single-frame PnP has no output in the gap")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "filter_outage.png"), dpi=120)
    plt.close(fig)

    # 4. Kalman trajectory (2 px, seed 0)
    kf_est = runs[("A", 2.0, KF)][0]["est"]
    pnp_est = runs[("A", 2.0, PNP)][0]["est"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    labels = ["x [m]", "y [m]", "z [m]", "roll [deg]", "pitch [deg]", "yaw [deg]"]
    for i, ax in enumerate(axes.T.ravel()):        # left column positions, right column angles
        ax.plot(times, true[:, i], "k-", label="true")
        ax.plot(times, pnp_est[:, i], ".", color=COLORS[PNP], ms=3, label="PnP measurement")
        ax.plot(times, kf_est[:, i], "-", color=COLORS[KF], lw=1.5, label="Kalman estimate")
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.3)
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 1].set_xlabel("Time [s]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("True vs Kalman-estimated pose, 2 px noise (seed 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "filter_trajectory.png"), dpi=120)
    plt.close(fig)

    print("\nSaved results/kf_noise_calibration.json, filter_metrics.json, filter_errors_2px.png, "
          "filter_noise_sweep.png, filter_outage.png, filter_trajectory.png")


if __name__ == "__main__":
    main()
