"""
Phase 5b: range-calibrated measurement noise and re-initialisation for the Kalman filter.

Both changes were fixed before looking at any test result:
  1. R from the PnP error covariance measured per range bin on the CALIBRATION poses
     (25-50, 50-75, 75-100, 100+ m; the approach never goes beyond ~113 m, so there is no 150+ bin),
     interpolated linearly between bin centres at the estimated range.
  2. After 3 consecutive gated measurements the attitude and rates are re-initialised
     from the current PnP measurement (3 is not tuned).

Runs the Phase 5 Kalman filter and the fixed one on the same observations (experiments A and B of
run_phase5.py; same seeds, noise levels and outage). Nothing else is changed.

    python run_phase5b.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import classifier
import config
import filters
import run_phase5 as p5
import simulator

KF5 = "Kalman filter (Phase 5)"
KF5B = "Kalman filter (fixed)"
CALIB_PATH = os.path.join(config.RESULTS_DIR, "kf_noise_calibration.json")


def run_one(name, model, times, true, noise, R_ref, range_bins, seed, outage):
    filt = filters.KalmanFilter(model, R_ref) if name == KF5 else filters.RangeKalmanFilter(model, range_bins, reinit_mode="attitude")
    est, _ = filters.run_filter(times, true, filt, noise_px=noise, seed=seed, outage=outage)
    pos, att = p5.errors(true, est)
    n_meas = len(times) - 1 - filt.n_missing          # updates attempted after the initial one
    return {"pos": pos, "att": att, "gated": filt.n_gated, "n_meas": n_meas,
            "reinit": getattr(filt, "n_reinit", 0)}


def main():
    model = classifier.load_model()
    times, true = simulator.approach_trajectory()

    # ---- range-binned calibration (calibration poses only) -------------------------------
    print("Calibrating PnP error per range bin...")
    range_cal = {}
    for noise in p5.NOISES:
        range_cal[noise] = filters.calibrate_pnp_noise_by_range(model, noise)
        print(f"  {noise:g} px: " + "; ".join(
            f"{b['range_lo_m']:g}-{'inf' if b['range_hi_m'] is None else format(b['range_hi_m'], 'g')} m "
            f"(n={b['n']}) pos std {np.sqrt(np.mean(np.square(b['std'][:3]))):.2f} m"
            for b in range_cal[noise]["bins"]))

    with open(CALIB_PATH) as f:                      # keep the Phase 5 content, add the range table
        calib_file = json.load(f)
    calib_file["phase5b_range_bins"] = {
        "settings": {"calibration_seed": filters.CALIB_SEED, "n_poses": filters.RANGE_CALIB_N,
                     "bin_edges_m": [25, 50, 75, 100, None],
                     "note": "raw covariance per bin (no range rescaling); R is linearly interpolated "
                             "between bin centres (centre_m = mean range of the bin's samples)"},
        "by_noise_px": {str(n): c for n, c in range_cal.items()}}
    with open(CALIB_PATH, "w") as f:
        json.dump(calib_file, f, indent=2)

    # ---- plot error vs range ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for noise in p5.NOISES:
        bins = range_cal[noise]["bins"]
        c = [b["centre_m"] for b in bins]
        axes[0].plot(c, [np.sqrt(np.mean(np.square(b["std"][:3]))) for b in bins], "o-", label=f"{noise:g} px")
        axes[1].plot(c, [np.sqrt(np.mean(np.square(b["std"][3:]))) for b in bins], "o-", label=f"{noise:g} px")
    axes[0].set_ylabel("PnP position error, RMS std per axis [m]")
    axes[1].set_ylabel("PnP attitude error, RMS std per axis [deg]")
    for ax in axes:
        ax.set_xlabel("Range (bin centre) [m]")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    fig.suptitle("PnP error vs range on the calibration poses (includes the 0.01 floor)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "kf_noise_vs_range.png"), dpi=120)
    plt.close(fig)

    # ---- experiments --------------------------------------------------------------------
    out_win, after_win = p5.window(times, *p5.OUTAGE), p5.window(times, *p5.AFTER)
    res = {"A": {}, "B": {}}
    for exp, outage in [("A", None), ("B", p5.OUTAGE)]:
        for noise in p5.NOISES:
            R_ref = np.array(json.load(open(CALIB_PATH))["by_noise_px"][str(noise)]["cov"])
            for name in (KF5, KF5B):
                runs = [run_one(name, model, times, true, noise, R_ref, range_cal[noise]["bins"], s, outage)
                        for s in p5.SEEDS]
                d = {"gate_rate_pct": p5.mean_std([100 * r["gated"] / r["n_meas"] for r in runs]),
                     "gated": p5.mean_std([r["gated"] for r in runs]),
                     "reinit": p5.mean_std([r["reinit"] for r in runs])}
                if exp == "A":
                    for k in ("pos", "att"):
                        d[f"{k}_mean"] = p5.mean_std([p5.stat(np.mean, r[k]) for r in runs])
                        d[f"{k}_median"] = p5.mean_std([p5.stat(np.median, r[k]) for r in runs])
                    d["att_mean_after_window"] = p5.mean_std([p5.stat(np.mean, r["att"][after_win]) for r in runs])
                else:
                    for tag, w in [("outage", out_win), ("after", after_win)]:
                        for k in ("pos", "att"):
                            d[f"{k}_mean_{tag}"] = p5.mean_std([p5.stat(np.mean, r[k][w]) for r in runs])
                res[exp].setdefault(f"{noise:g}", {})[name] = d
            print(f"  experiment {exp}, {noise:g} px done")

    f = lambda v, nd=2: f"{v[0]:.{nd}f}"
    print("\nEXPERIMENT A (mean over 5 seeds of per-run mean / median)")
    print(f"{'noise':>5s} {'filter':24s} {'pos mean':>8s} {'pos med':>8s} {'att mean':>8s} {'att med':>8s} "
          f"{'gate %':>7s} {'gated':>6s} {'reinit':>6s}")
    for noise in p5.NOISES:
        for name in (KF5, KF5B):
            a = res["A"][f"{noise:g}"][name]
            print(f"{noise:5g} {name:24s} {f(a['pos_mean']):>8s} {f(a['pos_median']):>8s} {f(a['att_mean']):>8s} "
                  f"{f(a['att_median']):>8s} {f(a['gate_rate_pct']):>7s} {f(a['gated'], 1):>6s} {f(a['reinit'], 1):>6s}")
    print("\nEXPERIMENT B: outage 4-5 s, after = 5-6 s")
    print(f"{'noise':>5s} {'filter':24s} {'pos in':>7s} {'att in':>7s} {'pos aft':>8s} {'att aft':>8s} "
          f"{'att aft (no outage)':>19s} {'gate %':>7s} {'reinit':>6s}")
    for noise in p5.NOISES:
        for name in (KF5, KF5B):
            b = res["B"][f"{noise:g}"][name]
            ref = res["A"][f"{noise:g}"][name]["att_mean_after_window"]
            print(f"{noise:5g} {name:24s} {f(b['pos_mean_outage']):>7s} {f(b['att_mean_outage']):>7s} "
                  f"{f(b['pos_mean_after']):>8s} {f(b['att_mean_after']):>8s} {f(ref):>19s} "
                  f"{f(b['gate_rate_pct']):>7s} {f(b['reinit'], 1):>6s}")

    with open(os.path.join(config.RESULTS_DIR, "kf_phase5b_metrics.json"), "w") as fh:
        json.dump({"settings": {"noises_px": p5.NOISES, "n_seeds": len(p5.SEEDS), "outage_s": p5.OUTAGE,
                                "after_window_s": p5.AFTER, "reinit_after": filters.REINIT_AFTER,
                                "note": "each stat is [mean over seeds, std over seeds]"},
                   "experiment_A": res["A"], "experiment_B": res["B"]}, fh, indent=2)
    print("\nSaved results/kf_noise_calibration.json (range table added), kf_noise_vs_range.png, kf_phase5b_metrics.json")


if __name__ == "__main__":
    main()
