"""
Phase 7a-fix: does the full-state re-initialisation make the synthetic Phase 5 / 5b results worse?

Runs experiments A (approach trajectory) and B (1 s outage, 4-5 s) of run_phase5b.py with three filters on identical
observations: the Phase 5 Kalman filter, the Phase 5b filter (attitude-only re-initialisation) and the same filter
with full-state re-initialisation. Same seeds, noise levels, calibration and constants; nothing is tuned.

    python run_phase7a_fix_synthetic.py
"""

import json
import os

import numpy as np

import classifier
import config
import filters
import run_phase5 as p5
import simulator

OUT_DIR = os.path.join("results", "phase7a_fix")
KF5 = "Kalman filter (Phase 5)"
KF5B_ATT = "Phase 5b (attitude re-init)"
KF5B_FULL = "Phase 5b + full-state re-init"
FILTERS = [KF5, KF5B_ATT, KF5B_FULL]


def make(name, model, R_ref, bins):
    if name == KF5:
        return filters.KalmanFilter(model, R_ref)
    return filters.RangeKalmanFilter(model, bins, reinit_mode="attitude" if name == KF5B_ATT else "full")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = classifier.load_model()
    times, true = simulator.approach_trajectory()
    out_win, after_win = p5.window(times, *p5.OUTAGE), p5.window(times, *p5.AFTER)
    res = {"A": {}, "B": {}}
    for noise in p5.NOISES:
        R_ref = np.array(filters.calibrate_pnp_noise(model, noise)["cov"])
        bins = filters.calibrate_pnp_noise_by_range(model, noise)["bins"]
        for name in FILTERS:
            runs = {"A": [], "B": []}
            for exp, outage in (("A", None), ("B", p5.OUTAGE)):
                for seed in p5.SEEDS:
                    filt = make(name, model, R_ref, bins)
                    est, _ = filters.run_filter(times, true, filt, noise_px=noise, seed=seed, outage=outage)
                    pos, att = p5.errors(true, est)
                    runs[exp].append({"pos": pos, "att": att, "gated": filt.n_gated, "reinit": getattr(filt, "n_reinit", 0),
                                      "n_meas": len(times) - 1 - filt.n_missing})
            a = {"gate_rate_pct": p5.mean_std([100 * r["gated"] / r["n_meas"] for r in runs["A"]]),
                 "reinit": p5.mean_std([r["reinit"] for r in runs["A"]])}
            for k in ("pos", "att"):
                a[f"{k}_mean"] = p5.mean_std([p5.stat(np.mean, r[k]) for r in runs["A"]])
                a[f"{k}_median"] = p5.mean_std([p5.stat(np.median, r[k]) for r in runs["A"]])
            a["att_mean_after_window"] = p5.mean_std([p5.stat(np.mean, r["att"][after_win]) for r in runs["A"]])
            b = {"gate_rate_pct": p5.mean_std([100 * r["gated"] / r["n_meas"] for r in runs["B"]]),
                 "reinit": p5.mean_std([r["reinit"] for r in runs["B"]])}
            for tag, w in (("outage", out_win), ("after", after_win)):
                for k in ("pos", "att"):
                    b[f"{k}_mean_{tag}"] = p5.mean_std([p5.stat(np.mean, r[k][w]) for r in runs["B"]])
            res["A"].setdefault(f"{noise:g}", {})[name] = a
            res["B"].setdefault(f"{noise:g}", {})[name] = b
        print(f"  noise {noise:g} px done")

    f = lambda v, nd=2: f"{v[0]:.{nd}f}"
    print("\nEXPERIMENT A (normal run)")
    print(f"{'noise':>5s} {'filter':32s} {'pos mean':>8s} {'pos med':>8s} {'att mean':>8s} {'att med':>8s} {'gate %':>7s} {'reinit':>6s}")
    for noise in p5.NOISES:
        for name in FILTERS:
            a = res["A"][f"{noise:g}"][name]
            print(f"{noise:5g} {name:32s} {f(a['pos_mean']):>8s} {f(a['pos_median']):>8s} {f(a['att_mean']):>8s} "
                  f"{f(a['att_median']):>8s} {f(a['gate_rate_pct']):>7s} {f(a['reinit'], 1):>6s}")
    print("\nEXPERIMENT B (outage 4-5 s; after = 5-6 s)")
    print(f"{'noise':>5s} {'filter':32s} {'pos in':>7s} {'att in':>7s} {'pos aft':>8s} {'att aft':>8s} {'reinit':>6s}")
    for noise in p5.NOISES:
        for name in FILTERS:
            b = res["B"][f"{noise:g}"][name]
            print(f"{noise:5g} {name:32s} {f(b['pos_mean_outage']):>7s} {f(b['att_mean_outage']):>7s} "
                  f"{f(b['pos_mean_after']):>8s} {f(b['att_mean_after']):>8s} {f(b['reinit'], 1):>6s}")
    with open(os.path.join(OUT_DIR, "synthetic_comparison.json"), "w") as fh:
        json.dump({"settings": {"noises_px": p5.NOISES, "n_seeds": len(p5.SEEDS), "outage_s": p5.OUTAGE,
                                "after_window_s": p5.AFTER, "note": "[mean over seeds, std over seeds]"},
                   "experiment_A": res["A"], "experiment_B": res["B"]}, fh, indent=1)
    print(f"\nSaved {OUT_DIR}/synthetic_comparison.json")


if __name__ == "__main__":
    main()
