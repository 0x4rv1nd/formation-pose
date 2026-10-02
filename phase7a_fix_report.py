"""
Phase 7a-fix: before / after report for the Kalman filter on FW-UAV6DPose.

Before = Phase 7a filter (attitude-only re-initialisation, results/phase7a/uav_metrics.json),
after  = full-state re-initialisation (results/phase7a_fix/uav_metrics.json). Writes
results/phase7a_fix/before_after.md and uav_before_after.png.

    python phase7a_evaluate.py --reinit attitude   # before
    python phase7a_evaluate.py                     # after
    python phase7a_fix_report.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join("results", "phase7a_fix")
NOISES = ["0", "1", "2", "5"]
HIGHLIGHT = ["000096", "000009"]


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    before = load(os.path.join("results", "phase7a", "uav_metrics.json"))
    after = load(os.path.join(OUT_DIR, "uav_metrics.json"))
    L = ["| Noise | Filter | Trans mean [m] | Trans median [m] | Rot mean [deg] | Rot median [deg] | Gate rejection | Re-inits |",
         "|---|---|---|---|---|---|---|---|"]
    for n in NOISES:
        p = after["results"][n]["all"]
        L.append(f"| {n} px | PnP only (reference) | {p['pnp_pos_mean'][0]:.2f} | {p['pnp_pos_median'][0]:.2f} | "
                 f"{p['pnp_rot_mean'][0]:.2f} | {p['pnp_rot_median'][0]:.2f} | - | - |")
        for tag, M in (("Kalman, attitude re-init (before)", before), ("Kalman, full re-init (after)", after)):
            a = M["results"][n]["all"]
            L.append(f"| | {tag} | {a['kf_pos_mean'][0]:.2f} ± {a['kf_pos_mean'][1]:.2f} | {a['kf_pos_median'][0]:.2f} | "
                     f"{a['kf_rot_mean'][0]:.2f} ± {a['kf_rot_mean'][1]:.2f} | {a['kf_rot_median'][0]:.2f} | "
                     f"{a['gate_rate_pct'][0]:.1f} % | {a['reinit_total'][0]:.0f} |")
    table = "\n".join(L)
    S = ["| Scene (2 px) | PnP trans / rot | Kalman before: trans / rot, gate, re-inits | Kalman after: trans / rot, gate, re-inits |",
         "|---|---|---|---|"]
    for s in HIGHLIGHT:
        b, a = before["per_scene_2px"][s], after["per_scene_2px"][s]
        S.append(f"| {s} | {b['pnp_pos_mean']:.2f} m / {b['pnp_rot_mean']:.2f}° | {b['kf_pos_mean']:.2f} m / {b['kf_rot_mean']:.2f}°, "
                 f"{b['gate_rate_pct']:.0f} %, {b['reinit']:.1f} | {a['kf_pos_mean']:.2f} m / {a['kf_rot_mean']:.2f}°, "
                 f"{a['gate_rate_pct']:.0f} %, {a['reinit']:.1f} |")
    n_before = sum(d["kf_pos_mean"] < d["pnp_pos_mean"] for d in before["per_scene_2px"].values())
    n_after = sum(d["kf_pos_mean"] < d["pnp_pos_mean"] for d in after["per_scene_2px"].values())
    S.append("")
    S.append(f"Kalman beats PnP in translation in {n_before} of 24 scenes before and {n_after} after (2 px).")
    D = ["| dt | Trans mean before / after [m] | Rot mean before / after [deg] | Gate rejection before / after | Re-inits before / after |",
         "|---|---|---|---|---|"]
    for k in before["dt_sensitivity_2px"]:
        b, a = before["dt_sensitivity_2px"][k], after["dt_sensitivity_2px"][k]
        D.append(f"| {k} s | {b['kf_pos_mean'][0]:.2f} / {a['kf_pos_mean'][0]:.2f} | {b['kf_rot_mean'][0]:.2f} / {a['kf_rot_mean'][0]:.2f} | "
                 f"{b['gate_rate_pct'][0]:.1f} % / {a['gate_rate_pct'][0]:.1f} % | {b['reinit_total'][0]:.0f} / {a['reinit_total'][0]:.0f} |")
    with open(os.path.join(OUT_DIR, "before_after.md"), "w") as f:
        f.write(table + "\n\n" + "\n".join(S) + "\n\n" + "\n".join(D) + "\n")
    print(table, "\n", "\n".join(S), "\n", "\n".join(D), sep="\n")

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(NOISES))
    for ax, e, label in [(axes[0], "pos", "Mean translation error [m]"), (axes[1], "rot", "Mean rotation error [deg]")]:
        for k, (name, M, key, col) in enumerate([("PnP only (SQPnP)", after, "pnp", "tab:orange"),
                                                 ("Kalman, attitude re-init (before)", before, "kf", "tab:red"),
                                                 ("Kalman, full re-init (after)", after, "kf", "tab:green")]):
            v = np.array([M["results"][n]["all"][f"{key}_{e}_mean"] for n in NOISES])
            ax.errorbar(x + (k - 1) * 0.05, np.maximum(v[:, 0], 1e-3), yerr=v[:, 1], marker="o", capsize=3, color=col, label=name)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(NOISES)
        ax.set_xlabel("Keypoint noise [px]")
        ax.set_ylabel(label)
        ax.grid(which="both", alpha=0.25)
    axes[0].legend(fontsize=9)
    fig.suptitle("FW-UAV6DPose: Kalman filter before / after full-state re-initialisation (5 seeds; 0 px PnP drawn at 1e-3)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "uav_before_after.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
