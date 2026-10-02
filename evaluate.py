"""
Phase 6: final evaluation. Re-runs every method from scratch in one consistent setting by importing
the existing modules (no method or parameter is changed), checks the numbers against the
earlier phases' stored results, and writes tables, figures and KEY_RESULTS.md to results/final/.

Methods: classifier only, the paper's particle filter (Phase 3 Config B), PnP only (SQPnP),
classifier + PnP, hybrid PnP, improved particle filter, Kalman filter (Phase 5) and the
range-calibrated Kalman filter (Phase 5b).

Experiments (5 seeds, every method sees identical observations for a given noise level and seed):
    A. approach trajectory at 0 / 1 / 2 / 5 px
    B. 1 s outage (all keypoints hidden, 4.0 <= t < 5.0 s); error during it and in the 1 s after it
    C. keypoint-dropout summary of Phase 4b (4-6 / 7-9 / 10+ visible keypoints)

    python evaluate.py            # stops without writing outputs if the sanity check fails
    python evaluate.py --force    # write the outputs anyway
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import classifier
import config
import filters
import pnp
import run_phase4b
import run_phase5 as p5
import simulator
from train_dropout import DROPOUT_MODEL_PATH

OUT_DIR = os.path.join(config.RESULTS_DIR, "final")
NOISES = p5.NOISES
SEEDS = p5.SEEDS
OUTAGE, AFTER = p5.OUTAGE, p5.AFTER
DPI = 200
PLOT_FLOOR = 1e-3        # errors below this (zero-noise PnP is exact to ~1e-12) are drawn at the floor

CLS = "Classifier only"
PAPER_PF = "Paper's particle filter"
SQPNP = "PnP only (SQPnP)"
CLS_PNP = "Classifier + PnP"
HYBRID = "Hybrid PnP"
IMP_PF = "Improved particle filter"
KF5 = "Kalman filter (Phase 5)"
KF5B = "Range-calibrated Kalman filter"
METHODS = [CLS, PAPER_PF, SQPNP, CLS_PNP, HYBRID, IMP_PF, KF5, KF5B]
COLORS = {CLS: "#7f7f7f", PAPER_PF: "#1f77b4", SQPNP: "#ff7f0e", CLS_PNP: "#bcbd22",
          HYBRID: "#8c564b", IMP_PF: "#9467bd", KF5: "#17becf", KF5B: "#2ca02c"}
MARKERS = {CLS: "s", PAPER_PF: "o", SQPNP: "^", CLS_PNP: "v", HYBRID: "D", IMP_PF: "P", KF5: "X", KF5B: "*"}


# ---------------------------------------------------------------------------
# Single-frame methods wrapped so that filters.run_filter can run all methods the same way
# (estimate NaN where the method has no estimate; the same rule as run_phase5.PnPOnly)
# ---------------------------------------------------------------------------
class _SingleFrame:
    def __init__(self, model):
        self.model = model

    def initialise(self, obs):
        return self.step(obs)

    def step(self, obs):
        if obs.visible.sum() == 0:                       # no keypoints at all: nothing to estimate from
            return np.full(6, np.nan)
        return self.estimate(obs)


class ClassifierOnly(_SingleFrame):
    def estimate(self, obs):
        return classifier.predict_pose(self.model, obs.features[None])[0]


class SQPnPOnly(_SingleFrame):
    def estimate(self, obs):
        pose, _ = pnp.pnp_only(obs)
        return np.full(6, np.nan) if pose is None else pose


class ClassifierPnP(_SingleFrame):
    def estimate(self, obs):
        return pnp.classifier_pnp(obs, classifier.predict_pose(self.model, obs.features[None])[0])[0]


def make_filter(name, model, noise, R_ref, range_bins, seed):
    if name == CLS:
        return ClassifierOnly(model)
    if name == PAPER_PF:
        return filters.ParticleFilter(model, 5000, filters.alpha_every_step(0.9), seed=seed)
    if name == SQPNP:
        return SQPnPOnly(model)
    if name == CLS_PNP:
        return ClassifierPnP(model)
    if name == HYBRID:
        return filters.PnPOnly(model)
    if name == IMP_PF:
        return filters.ImprovedParticleFilter(model, 1000, filters.alpha_every_step(0.9), noise, R_ref, seed=seed)
    if name == KF5:
        return filters.KalmanFilter(model, R_ref)
    return filters.RangeKalmanFilter(model, range_bins)


def ms(values):
    """[mean over seeds, std over seeds]"""
    return p5.mean_std(values)


# ---------------------------------------------------------------------------
# Experiments A and B
# ---------------------------------------------------------------------------
def run_trajectory_experiments(model):
    times, true = simulator.approach_trajectory()
    out_win, after_win = p5.window(times, *OUTAGE), p5.window(times, *AFTER)
    curves = {}                                    # (exp, noise, method) -> (pos, att) of seed 0
    A, B = {}, {}
    for noise in NOISES:
        R_ref = np.array(filters.calibrate_pnp_noise(model, noise)["cov"])
        range_bins = filters.calibrate_pnp_noise_by_range(model, noise)["bins"]
        for m in METHODS:
            runs = {"A": [], "B": []}
            for exp, outage in [("A", None), ("B", OUTAGE)]:
                for seed in SEEDS:
                    filt = make_filter(m, model, noise, R_ref, range_bins, seed)
                    est, runtime = filters.run_filter(times, true, filt, noise_px=noise, seed=seed, outage=outage)
                    pos, att = p5.errors(true, est)
                    runs[exp].append({"pos": pos, "att": att, "ms": 1000 * runtime.mean(),
                                      "gated": getattr(filt, "n_gated", None),
                                      "reinit": getattr(filt, "n_reinit", None),
                                      "n_meas": len(times) - 1 - getattr(filt, "n_missing", 0)})
                    if seed == 0:
                        curves[(exp, noise, m)] = (pos, att)
            ra, rb = runs["A"], runs["B"]
            a = {"pos_mean": ms([p5.stat(np.mean, r["pos"]) for r in ra]),
                 "pos_median": ms([p5.stat(np.median, r["pos"]) for r in ra]),
                 "att_mean": ms([p5.stat(np.mean, r["att"]) for r in ra]),
                 "att_median": ms([p5.stat(np.median, r["att"]) for r in ra]),
                 "ms_per_step": ms([r["ms"] for r in ra]),
                 "missing_steps": ms([int((~np.isfinite(r["pos"])).sum()) for r in ra])}
            if ra[0]["gated"] is not None:
                a["gate_rejections"] = ms([r["gated"] for r in ra])
                a["gate_rate_pct"] = ms([100 * r["gated"] / r["n_meas"] for r in ra])
            b = {}
            for tag, w in [("outage", out_win), ("after", after_win)]:
                for k in ("pos", "att"):
                    b[f"{k}_mean_{tag}"] = ms([p5.stat(np.mean, r[k][w]) for r in rb])
                b[f"missing_{tag}"] = ms([int((~np.isfinite(r["pos"][w])).sum()) for r in rb])
            if rb[0]["reinit"] is not None:
                b["reinit"] = ms([r["reinit"] for r in rb])
            A.setdefault(f"{noise:g}", {})[m] = a
            B.setdefault(f"{noise:g}", {})[m] = b
        print(f"  noise {noise:g} px done")
    return times, true, A, B, curves


# ---------------------------------------------------------------------------
# Sanity check against the earlier phases' stored results
# ---------------------------------------------------------------------------
def _load(name):
    with open(os.path.join(config.RESULTS_DIR, name)) as f:
        return json.load(f)


def sanity_check(A, B, dropout):
    """Compare with results/*.json written by Phases 3-5b. Returns (rows, n_bad)."""
    rows = []

    def cmp(source, label, new, old, atol=1e-6):
        both_nan = not np.isfinite(new) and (old is None or not np.isfinite(old))
        ok = both_nan or (old is not None and np.isfinite(new) and np.isfinite(old)
                          and abs(new - old) <= atol + 1e-4 * abs(old))
        rows.append({"source": source, "quantity": label, "new": new, "stored": old, "ok": bool(ok)})

    # Phase 4 (pnp_metrics.json): single-frame methods and Phase 3 Config B, mean over seeds of per-seed stats
    p4 = _load("pnp_metrics.json")["trajectory"]
    names4 = {CLS: "Classifier only", SQPNP: "PnP only", CLS_PNP: "Classifier + PnP", PAPER_PF: "Phase 3 Config B"}
    for noise in NOISES:
        for m, m4 in names4.items():
            for key in ("pos_mean", "pos_median", "att_mean", "att_median"):
                old = np.mean([s[key] for s in p4[str(noise)][m4]])
                cmp("Phase 4", f"{noise:g} px, {m}, {key}", A[f"{noise:g}"][m][key][0], old)
    # Phase 3 (pf_baseline_metrics.json): Config B at 0 px
    p3 = _load("pf_baseline_metrics.json")["Config B (N=5000, alpha=0.9 every step)"]
    cmp("Phase 3", "0 px, Config B, pos_mean", A["0"][PAPER_PF]["pos_mean"][0], p3["pos_err_m_mean"])
    cmp("Phase 3", "0 px, Config B, att_mean", A["0"][PAPER_PF]["att_mean"][0], p3["att_err_deg_mean"])
    # Phase 5 (filter_metrics.json)
    p5m = _load("filter_metrics.json")
    names5 = {PAPER_PF: "Phase 3 PF (Config B)", HYBRID: "PnP alone", IMP_PF: "Improved PF", KF5: "Kalman filter"}
    for noise in NOISES:
        for m, m5 in names5.items():
            old = p5m["experiment_A"][f"{noise:g}"][m5]
            for key in ("pos_mean", "pos_median", "att_mean", "att_median"):
                cmp("Phase 5", f"A, {noise:g} px, {m}, {key}", A[f"{noise:g}"][m][key][0], old[key][0])
            if m == KF5:
                cmp("Phase 5", f"A, {noise:g} px, {m}, gated", A[f"{noise:g}"][m]["gate_rejections"][0],
                    old["gate_rejections"][0])
            oldb = p5m["experiment_B"][f"{noise:g}"][m5]
            for key in ("pos_mean_outage", "att_mean_outage", "pos_mean_after", "att_mean_after"):
                cmp("Phase 5", f"B, {noise:g} px, {m}, {key}", B[f"{noise:g}"][m][key][0], oldb[key][0])
    # Phase 5b (kf_phase5b_metrics.json)
    p5b = _load("kf_phase5b_metrics.json")
    names5b = {KF5: "Kalman filter (Phase 5)", KF5B: "Kalman filter (fixed)"}
    for noise in NOISES:
        for m, mb in names5b.items():
            olda, oldb = p5b["experiment_A"][f"{noise:g}"][mb], p5b["experiment_B"][f"{noise:g}"][mb]
            for key in ("pos_mean", "pos_median", "att_mean", "att_median"):
                cmp("Phase 5b", f"A, {noise:g} px, {m}, {key}", A[f"{noise:g}"][m][key][0], olda[key][0])
            cmp("Phase 5b", f"A, {noise:g} px, {m}, gated", A[f"{noise:g}"][m]["gate_rejections"][0], olda["gated"][0])
            for key in ("pos_mean_outage", "att_mean_outage", "pos_mean_after", "att_mean_after"):
                cmp("Phase 5b", f"B, {noise:g} px, {m}, {key}", B[f"{noise:g}"][m][key][0], oldb[key][0])
            if m == KF5B:
                cmp("Phase 5b", f"B, {noise:g} px, {m}, reinit", B[f"{noise:g}"][m]["reinit"][0], oldb["reinit"][0])
    # Phase 4b (pnp_dropout_metrics.json)
    old4b = _load("pnp_dropout_metrics.json")["results"]
    for noise, groups in dropout.items():
        for g, methods in groups.items():
            for m, row in methods.items():
                if m == "n_poses":
                    continue
                for key in ("pos_mean", "pos_median", "att_mean", "att_median", "fail_rate"):
                    cmp("Phase 4b", f"{noise} px, {g}, {m}, {key}", row[key][0], old4b[noise][g][m][key][0])
    return rows, sum(not r["ok"] for r in rows)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def pm(v, nd=2):
    return "n/a" if v is None or not np.isfinite(v[0]) else f"{v[0]:.{nd}f} ± {v[1]:.{nd}f}"


def write_tables(A, B, dropout):
    with open(os.path.join(OUT_DIR, "final_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["noise_px", "method", "pos_mean_m", "pos_mean_std", "pos_median_m", "pos_median_std",
                    "att_mean_deg", "att_mean_std", "att_median_deg", "att_median_std",
                    "runtime_ms_per_step", "gate_rate_pct"])
        for noise in NOISES:
            for m in METHODS:
                a = A[f"{noise:g}"][m]
                w.writerow([f"{noise:g}", m]
                           + [f"{x:.6g}" for k in ("pos_mean", "pos_median", "att_mean", "att_median") for x in a[k]]
                           + [f"{a['ms_per_step'][0]:.3f}",
                              f"{a['gate_rate_pct'][0]:.2f}" if "gate_rate_pct" in a else ""])
    with open(os.path.join(OUT_DIR, "final_outage_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["noise_px", "method", "pos_in_outage_m", "pos_in_outage_std", "att_in_outage_deg",
                    "att_in_outage_std", "pos_after_m", "pos_after_std", "att_after_deg", "att_after_std"])
        for noise in NOISES:
            for m in METHODS:
                b = B[f"{noise:g}"][m]
                w.writerow([f"{noise:g}", m] + ["" if not np.isfinite(x) else f"{x:.6g}" for k in
                           ("pos_mean_outage", "att_mean_outage", "pos_mean_after", "att_mean_after") for x in b[k]])

    md = ["# Final results", "",
          "Approach trajectory (101 steps), 5 seeds. Cells: mean ± std over seeds of the per-run statistic. "
          "Position error in m, attitude error (geodesic angle) in degrees. `n/a`: no estimate. "
          "Runtime includes the classifier / PnP calls of each method (for the paper's filter, the classifier is called "
          "inside the filter), measured on this machine.", "",
          "## A. Approach trajectory", ""]
    for noise in NOISES:
        md += [f"### {noise:g} px keypoint noise", "",
               "| Method | Pos mean [m] | Pos median [m] | Att mean [deg] | Att median [deg] | ms/step | Gate rejection |",
               "|---|---|---|---|---|---|---|"]
        for m in METHODS:
            a = A[f"{noise:g}"][m]
            gate = f"{a['gate_rate_pct'][0]:.1f} %" if "gate_rate_pct" in a else "-"
            md.append(f"| {m} | {pm(a['pos_mean'])} | {pm(a['pos_median'])} | {pm(a['att_mean'])} | "
                      f"{pm(a['att_median'])} | {a['ms_per_step'][0]:.2f} | {gate} |")
        md.append("")
    md += ["## B. Outage: all keypoints hidden for 4.0 <= t < 5.0 s",
           "", "Mean error during the outage and in the 1 s after it (5.0 <= t < 6.0 s). Single-frame methods have no "
           "estimate while there are no keypoints (`n/a`). Re-initialisations: range-calibrated Kalman filter, mean per run.", "",
           "| Noise | Method | Pos in outage [m] | Att in outage [deg] | Pos after [m] | Att after [deg] |",
           "|---|---|---|---|---|---|"]
    for noise in NOISES:
        for m in METHODS:
            b = B[f"{noise:g}"][m]
            md.append(f"| {noise:g} px | {m} | {pm(b['pos_mean_outage'])} | {pm(b['att_mean_outage'])} | "
                      f"{pm(b['pos_mean_after'])} | {pm(b['att_mean_after'])} |")
    md += ["", "## C. Keypoint dropout (Phase 4b, 2000 random poses, 5 seeds)", "",
           "Median errors and gross-failure rate (no solution, position > 10 m or attitude > 20 deg). "
           "`dropout clf` is the classifier retrained with hidden keypoints.", "",
           "| Noise | Visible | Method | Pos mean / median [m] | Att mean / median [deg] | Fail % |",
           "|---|---|---|---|---|---|"]
    for noise, groups in dropout.items():
        for g, methods in groups.items():
            for m, r in methods.items():
                if m in ("n_poses", "Classifier only (dropout clf)"):
                    continue
                md.append(f"| {float(noise):g} px | {g} | {m} | {r['pos_mean'][0]:.2f} / {r['pos_median'][0]:.2f} | "
                          f"{r['att_mean'][0]:.2f} / {r['att_median'][0]:.2f} | {100 * r['fail_rate'][0]:.1f} |")
    with open(os.path.join(OUT_DIR, "final_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def style():
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
                         "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})


def floored(v):
    return np.maximum(np.asarray(v, dtype=float), PLOT_FLOOR)


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_noise_sweep(A):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x0 = np.arange(len(NOISES))
    for ax, key, label in [(axes[0], "pos_mean", "Mean position error [m]"),
                           (axes[1], "att_mean", "Mean attitude error [deg]")]:
        for k, m in enumerate(METHODS):
            vals = np.array([A[f"{n:g}"][m][key] for n in NOISES])
            x = x0 + (k - (len(METHODS) - 1) / 2) * 0.05
            mean = floored(vals[:, 0])
            lower = np.minimum(vals[:, 1], mean - PLOT_FLOOR * 0.999)    # keep the bar on the log axis
            ax.errorbar(x, mean, yerr=[np.maximum(lower, 0), vals[:, 1]], marker=MARKERS[m], ms=7, capsize=3,
                        lw=1.6, color=COLORS[m], label=m)
        ax.set_yscale("log")
        ax.set_xticks(x0)
        ax.set_xticklabels([f"{n:g}" for n in NOISES])
        ax.set_xlabel("Keypoint noise [px]")
        ax.set_ylabel(label)
        ax.axhline(PLOT_FLOOR, color="k", ls=":", lw=0.8)
        ax.set_ylim(PLOT_FLOOR * 0.6, 30)
        ax.grid(which="both", alpha=0.25)
    axes[0].text(0.02, PLOT_FLOOR * 1.15, f"values below {PLOT_FLOOR:g} drawn at this line",
                 transform=axes[0].get_yaxis_transform(), fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Error vs keypoint noise on the approach trajectory (mean over 5 seeds, error bars = std over seeds)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig_noise_sweep.png")


def fig_trajectory(times, curves):
    story = [PAPER_PF, HYBRID, KF5B]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    for ax, i, label in [(axes[0], 0, "Position error [m]"), (axes[1], 1, "Attitude error [deg]")]:
        for m in story:
            ax.plot(times, curves[("A", 2.0, m)][i], color=COLORS[m], lw=1.8, label=m)
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.grid(which="both", alpha=0.25)
    axes[1].set_xlabel("Time [s]")
    axes[0].set_title("Errors on the approach trajectory, 2 px noise (seed 0)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_trajectory_2px.png")


def fig_outage(times, curves):
    shown = [PAPER_PF, HYBRID, IMP_PF, KF5, KF5B]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    for ax, i, label in [(axes[0], 0, "Position error [m]"), (axes[1], 1, "Attitude error [deg]")]:
        for m in shown:
            ax.plot(times, curves[("B", 2.0, m)][i], color=COLORS[m], lw=1.6, marker=MARKERS[m], ms=3, label=m)
        ax.axvspan(*OUTAGE, color="gray", alpha=0.25, label="outage (no keypoints)")
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.grid(which="both", alpha=0.25)
    axes[1].set_xlabel("Time [s]")
    axes[0].legend(loc="lower left", ncol=2, fontsize=8)
    axes[0].set_title("Measurement outage, 2 px noise (seed 0); single-frame PnP has no output in the gap")
    fig.tight_layout()
    save(fig, "fig_outage.png")


TICK_LABELS = ["Classifier\nonly", "Paper's\nparticle\nfilter", "PnP only\n(SQPnP)", "Classifier\n+ PnP", "Hybrid\nPnP",
               "Improved\nparticle\nfilter", "Kalman\nfilter\n(Phase 5)", "Range-\ncalibrated\nKalman\nfilter"]


def fig_summary(A):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, key, label in [(axes[0], "pos_mean", "Mean position error [m]"),
                           (axes[1], "att_mean", "Mean attitude error [deg]")]:
        vals = np.array([A["2"][m][key] for m in METHODS])
        bars = ax.bar(range(len(METHODS)), vals[:, 0], yerr=vals[:, 1], capsize=3,
                      color=[COLORS[m] for m in METHODS])
        for b, (v, sd) in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, (v + sd) * 1.1, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_yscale("log")
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(TICK_LABELS, fontsize=9)
        ax.set_ylabel(label)
        ax.set_ylim(0.3 * min(vals[:, 0]), 3 * max(vals[:, 0] + vals[:, 1]))
        ax.grid(axis="y", which="both", alpha=0.25)
        ax.grid(axis="x", visible=False)
    fig.suptitle("From the paper's method to the final system, 2 px noise (mean over 5 seeds, error bars = std)")
    fig.tight_layout()
    save(fig, "fig_summary_bars.png")


# ---------------------------------------------------------------------------
# KEY_RESULTS.md (every number comes from this run)
# ---------------------------------------------------------------------------
def write_key_results(A, B, dropout):
    g = lambda noise, m, k: A[f"{noise:g}"][m][k][0]
    b = lambda noise, m, k: B[f"{noise:g}"][m][k][0]
    L = ["# Key results", "",
         "All numbers are produced by `evaluate.py` (approach trajectory, 5 seeds, simulated keypoints with Gaussian pixel noise; "
         "mean over seeds of the time-averaged error).", ""]
    for noise in (2, 5):
        L.append(f"- At {noise} px noise, the range-calibrated Kalman filter reduces position error from "
                 f"{g(noise, PAPER_PF, 'pos_mean'):.2f} m (paper's particle filter) to {g(noise, KF5B, 'pos_mean'):.2f} m "
                 f"({g(noise, PAPER_PF, 'pos_mean') / g(noise, KF5B, 'pos_mean'):.0f}x lower) and attitude error from "
                 f"{g(noise, PAPER_PF, 'att_mean'):.2f} deg to {g(noise, KF5B, 'att_mean'):.2f} deg "
                 f"({g(noise, PAPER_PF, 'att_mean') / g(noise, KF5B, 'att_mean'):.1f}x lower).")
    L.append(f"- The classifier alone has {g(2, CLS, 'pos_mean'):.2f} m / {g(2, CLS, 'att_mean'):.2f} deg error at 2 px; "
             f"the paper's particle filter (Config B, 5000 particles) only reaches {g(2, PAPER_PF, 'pos_mean'):.2f} m / "
             f"{g(2, PAPER_PF, 'att_mean'):.2f} deg, i.e. it does not clearly improve on the classifier.")
    L.append(f"- Most of the improvement comes from geometry, not filtering: single-frame PnP alone gives "
             f"{g(2, SQPNP, 'pos_mean'):.2f} m / {g(2, SQPNP, 'att_mean'):.2f} deg at 2 px, and "
             f"{g(5, SQPNP, 'pos_mean'):.2f} m / {g(5, SQPNP, 'att_mean'):.2f} deg at 5 px.")
    L.append(f"- Classifier + PnP matches SQPnP alone at 2 px ({g(2, CLS_PNP, 'pos_mean'):.2f} vs {g(2, SQPNP, 'pos_mean'):.2f} m) "
             f"and helps at 5 px ({g(5, CLS_PNP, 'pos_mean'):.2f} vs {g(5, SQPNP, 'pos_mean'):.2f} m position, "
             f"{g(5, CLS_PNP, 'att_mean'):.2f} vs {g(5, SQPNP, 'att_mean'):.2f} deg attitude).")
    L.append(f"- Filtering PnP with the Kalman filter lowers position error again (2 px: {g(2, HYBRID, 'pos_mean'):.2f} m for the hybrid "
             f"PnP measurement alone vs {g(2, KF5B, 'pos_mean'):.2f} m filtered; 5 px: {g(5, HYBRID, 'pos_mean'):.2f} vs "
             f"{g(5, KF5B, 'pos_mean'):.2f} m), but attitude is unchanged ({g(2, HYBRID, 'att_mean'):.2f} vs "
             f"{g(2, KF5B, 'att_mean'):.2f} deg at 2 px).")
    L.append(f"- The improved particle filter (1000 particles) reaches {g(2, IMP_PF, 'pos_mean'):.2f} m / {g(2, IMP_PF, 'att_mean'):.2f} deg at 2 px "
             f"but is not better than the Kalman filter ({g(2, KF5B, 'pos_mean'):.2f} m position at 2 px).")
    L.append(f"- Calibrating the Kalman measurement noise by range cuts gate rejections from {g(2, KF5, 'gate_rate_pct'):.1f} % to "
             f"{g(2, KF5B, 'gate_rate_pct'):.1f} % of measurements at 2 px (nominal 0.1 %), with unchanged normal accuracy "
             f"({g(2, KF5, 'pos_mean'):.2f} vs {g(2, KF5B, 'pos_mean'):.2f} m).")
    L.append(f"- After a 1 s measurement outage at 2 px the Phase 5 Kalman filter's attitude error in the following second is "
             f"{b(2, KF5, 'att_mean_after'):.1f} deg; the re-initialising, range-calibrated filter brings this to "
             f"{b(2, KF5B, 'att_mean_after'):.1f} deg (the paper's filter: {b(2, PAPER_PF, 'att_mean_after'):.1f} deg). It still "
             f"has {g(2, KF5B, 'att_mean'):.1f} deg in normal operation, so a few steps after the gap remain poor until the filter re-initialises.")
    L.append(f"- During the outage (2 px) the range-calibrated Kalman filter keeps the position error at {b(2, KF5B, 'pos_mean_outage'):.2f} m, "
             f"against {b(2, PAPER_PF, 'pos_mean_outage'):.2f} m for the paper's filter; its attitude error grows to "
             f"{b(2, KF5B, 'att_mean_outage'):.1f} deg (paper's filter: {b(2, PAPER_PF, 'att_mean_outage'):.1f} deg, improved particle filter: "
             f"{b(2, IMP_PF, 'att_mean_outage'):.1f} deg) because the rate model extrapolates the roll swing. Single-frame PnP gives nothing in the gap.")
    r = dropout["2.0"]["4-6"]
    L.append(f"- With only 4-6 visible keypoints at 2 px, classifier + PnP with the dropout-trained classifier has a gross-failure rate of "
             f"{100 * r['Classifier + PnP (dropout clf)']['fail_rate'][0]:.1f} % against {100 * r['PnP only']['fail_rate'][0]:.1f} % "
             f"for PnP only; median errors are the same ({r['Classifier + PnP (dropout clf)']['pos_median'][0]:.2f} vs "
             f"{r['PnP only']['pos_median'][0]:.2f} m).")
    L.append(f"- Runtime per step (this machine, includes classifier and PnP calls): "
             f"paper's particle filter {g(2, PAPER_PF, 'ms_per_step'):.1f} ms, improved particle filter {g(2, IMP_PF, 'ms_per_step'):.1f} ms, "
             f"range-calibrated Kalman filter {g(2, KF5B, 'ms_per_step'):.2f} ms.")
    with open(os.path.join(OUT_DIR, "KEY_RESULTS.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return L


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="write outputs even if the sanity check fails")
    force = parser.parse_args().force

    os.makedirs(OUT_DIR, exist_ok=True)
    model = classifier.load_model()
    print("Experiments A and B (approach trajectory, outage)...")
    times, true, A, B, curves = run_trajectory_experiments(model)
    print("Experiment C (keypoint dropout)...")
    dropout, _ = run_phase4b.sweep({"orig": model, "dropout": classifier.load_model(DROPOUT_MODEL_PATH)}, verbose=False)

    rows, n_bad = sanity_check(A, B, dropout)
    with open(os.path.join(OUT_DIR, "sanity_check.json"), "w") as f:
        json.dump({"n_compared": len(rows), "n_mismatch": n_bad, "rows": rows}, f, indent=1)
    print(f"\nSANITY CHECK against stored Phase 3-5b results: {len(rows) - n_bad}/{len(rows)} values match "
          f"(abs 1e-6 + rel 1e-4)")
    for r in rows:
        if not r["ok"]:
            print(f"  MISMATCH {r['source']}: {r['quantity']}: new {r['new']} vs stored {r['stored']}")
    if n_bad and not force:
        print("Stopping: results differ from the earlier phases. Nothing else written (use --force to override).")
        sys.exit(1)

    write_tables(A, B, dropout)
    style()
    fig_noise_sweep(A)
    fig_trajectory(times, curves)
    fig_outage(times, curves)
    fig_summary(A)
    lines = write_key_results(A, B, dropout)
    with open(os.path.join(OUT_DIR, "final_metrics.json"), "w") as f:
        json.dump({"experiment_A": A, "experiment_B": B, "dropout": dropout}, f, indent=1)
    print(f"\nSaved final_table.csv/.md, final_outage_table.csv, fig_*.png, KEY_RESULTS.md, final_metrics.json to {OUT_DIR}/")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
