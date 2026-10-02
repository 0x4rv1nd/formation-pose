"""
Phase 7a: evaluation on the FW-UAV6DPose validation split with REAL ground-truth poses (no images are used).

Keypoints on the UAV's 3D model are projected with each frame's ground-truth pose and intrinsics, visibility comes
from the visible-object mask (dilated by 2 px), Gaussian pixel noise is added (0, 1, 2, 5 px, 5 seeds) and two
methods run in the camera frame:
    PnP only (SQPnP)                    single frame, no classifier
    Range-calibrated Kalman filter      Phase 5b filter on the PnP measurements of each sequence, R calibrated by
                                        range bin on SYNTHETIC poses made with this UAV model and camera
Scene 000125 contains a data glitch between frames 76 and 77 (translation jump of 217 m, rotation jump of 131 deg): the
filter is restarted at frame 77 (sequences 0-76 and 77-99), PnP keeps all frames. dt = 0.1 s is assumed (no timestamps); dt = 0.05 s and 0.2 s are run as a sensitivity check at 2 px.

    python phase7a_evaluate.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fw_uav_config as cfg
import fw_uav_io as io
import fw_uav_methods as meth

OUT_DIR = os.path.join("results", "phase7a")
PNP, KF = "PnP only (SQPnP)", "Range-calibrated Kalman filter"
COLORS = {PNP: "tab:orange", KF: "tab:green"}
BIN_LABELS = ["150-250 m", "250-350 m", "350-450 m", "450+ m"]
EDGES = np.array(cfg.RANGE_BINS_M)


def sequences(scene, n_frames):
    """Index ranges of the filter sequences of a scene (cut at the glitch frames)."""
    cuts = [0] + cfg.GLITCH_SPLITS.get(scene, []) + [n_frames]
    return list(zip(cuts[:-1], cuts[1:]))


def run_scene(c, scene, noise, seed, range_bins, dt=cfg.DT, run_pnp=True):
    """One scene, one noise level and seed. Returns per-frame arrays for PnP and the Kalman filter."""
    T = len(c["frames"])
    rng = np.random.default_rng([int(scene), int(noise * 10), seed])      # same noisy pixels for every method / dt
    uv = np.stack([c["uv"][i] + (rng.normal(0.0, noise, c["uv"][i].shape) if noise > 0 else 0.0) for i in range(T)])
    z = [meth.pnp_measurement(uv[i], c["vis_dilated"][i], c["K"]) for i in range(T)]
    out = {"range": np.linalg.norm(c["t"], axis=1)}
    pnp_pos, pnp_rot = np.full(T, np.nan), np.full(T, np.nan)
    for i in range(T):
        if z[i] is not None:
            pnp_pos[i], pnp_rot[i] = meth.pose_errors(z[i], c["R"][i], c["t"][i])
    out["pnp_pos"], out["pnp_rot"] = pnp_pos, pnp_rot
    out["pnp_fail"] = (~np.isfinite(pnp_pos)) | (pnp_pos > meth.GROSS_REL_POS * out["range"]) | (pnp_rot > meth.GROSS_ROT_DEG)
    kf_pos, kf_rot = np.full(T, np.nan), np.full(T, np.nan)
    est = np.full((T, 6), np.nan)
    gated = offered = reinit = 0
    for a, b in sequences(scene, T):
        kf = meth.UAVKalman(range_bins, dt)
        for i in range(a, b):
            est[i] = kf.initialise(z[i]) if i == a else kf.step(z[i])
            if np.isfinite(est[i]).all():
                kf_pos[i], kf_rot[i] = meth.pose_errors(est[i], c["R"][i], c["t"][i])
        gated += kf.n_gated
        offered += (b - a) - 1 - kf.n_missing
        reinit += kf.n_reinit
    out.update(kf_pos=kf_pos, kf_rot=kf_rot, gated=gated, offered=offered, reinit=reinit, kf_est=est,
               pnp_est=np.array([np.full(6, np.nan) if v is None else v for v in z]))
    return out


def pooled(per_scene, key, mask_fn=None):
    """Concatenate a per-frame array over scenes (optionally keeping frames where mask_fn(run) is True)."""
    parts = []
    for r in per_scene.values():
        v = r[key]
        parts.append(v[mask_fn(r)] if mask_fn else v)
    return np.concatenate(parts)


def summarise(per_scene):
    """Mean / median errors, failure and gate rates, re-initialisations for one (noise, seed), overall and per range bin."""
    res = {}
    rng_all = pooled(per_scene, "range")
    b_idx = np.digitize(rng_all, EDGES[1:-1])                 # 0..3
    for tag, sel in [("all", np.ones(len(rng_all), bool))] + [(BIN_LABELS[k], b_idx == k) for k in range(4)]:
        d = {"n_frames": int(sel.sum())}
        for m in ("pnp", "kf"):
            for e in ("pos", "rot"):
                v = pooled(per_scene, f"{m}_{e}")[sel]
                v = v[np.isfinite(v)]
                d[f"{m}_{e}_mean"] = float(v.mean()) if len(v) else float("nan")
                d[f"{m}_{e}_median"] = float(np.median(v)) if len(v) else float("nan")
            pos = pooled(per_scene, f"{m}_pos")[sel]
            d[f"{m}_pos_rel_mean_pct"] = float(np.nanmean(100 * pos / rng_all[sel])) if np.isfinite(pos).any() else float("nan")
        d["pnp_fail_pct"] = float(100 * pooled(per_scene, "pnp_fail")[sel].mean())
        d["pnp_no_solution_pct"] = float(100 * (~np.isfinite(pooled(per_scene, "pnp_pos")[sel])).mean())
        res[tag] = d
    res["all"]["gate_rate_pct"] = float(100 * sum(r["gated"] for r in per_scene.values())
                                        / max(sum(r["offered"] for r in per_scene.values()), 1))
    res["all"]["reinit_total"] = int(sum(r["reinit"] for r in per_scene.values()))
    return res


def aggregate(per_seed):
    """list over seeds of summarise() dicts -> {tag: {key: [mean, std]}}."""
    out = {}
    for tag in per_seed[0]:
        out[tag] = {k: [float(np.nanmean([s[tag][k] for s in per_seed])), float(np.nanstd([s[tag][k] for s in per_seed]))]
                    for k in per_seed[0][tag]}
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = io.build_keypoint_cache(cfg.KEYPOINTS)
    K = cache[io.scenes()[0]]["K"]

    # ---- visibility rates ------------------------------------------------------------------
    vis = {}
    for k in ("vis_strict", "vis_dilated"):
        a = np.concatenate([c[k] for c in cache.values()])
        vis[k] = {"mean_visible_fraction": float(a.mean()), "per_keypoint": a.mean(0).tolist(),
                  "frames_with_fewer_than_6": int((a.sum(1) < 6).sum()), "frames_with_fewer_than_4": int((a.sum(1) < 4).sum()),
                  "min_visible": int(a.sum(1).min()), "n_frames": int(len(a))}
    print(f"visible keypoints: strict mask {100 * vis['vis_strict']['mean_visible_fraction']:.1f} %, "
          f"dilated by {io.MASK_DILATE_PX} px {100 * vis['vis_dilated']['mean_visible_fraction']:.1f} %")

    # ---- synthetic calibration of R ------------------------------------------------------------
    print("Calibrating R on synthetic poses...")
    calib = {n: meth.calibrate_range_noise(n, K) for n in cfg.NOISES}
    with open(os.path.join(OUT_DIR, "uav_noise_calibration.json"), "w") as f:
        json.dump({"settings": {"seed": meth.CALIB_SEED, "n_poses": meth.CALIB_N, "range_m": cfg.CALIB_RANGE_M,
                                "orientation": "uniform on SO(3), |pitch| < 80 deg", "position": "uniform pixel in the image, "
                                "range uniform", "gross": "position > 10 % of range or rotation > 20 deg"},
                   "by_noise_px": {str(n): c for n, c in calib.items()}}, f, indent=2)
    for n, c in calib.items():
        print(f"  {n:g} px: z std per bin " + ", ".join(f"{b['std'][2]:.2f}" for b in c["bins"]) + " m")

    # ---- main experiment ------------------------------------------------------------------------
    results, curves, per_seed_all, per_scene_2px = {}, {}, {}, {s: [] for s in cache}
    for noise in cfg.NOISES:
        per_seed = []
        for seed in cfg.SEEDS:
            runs = {s: run_scene(cache[s], s, noise, seed, calib[noise]["bins"]) for s in cache}
            per_seed.append(summarise(runs))
            if noise == 2.0:
                for s, r in runs.items():
                    per_scene_2px[s].append({"range_min_m": float(r["range"].min()), "range_max_m": float(r["range"].max()),
                                             "pnp_pos_mean": float(np.nanmean(r["pnp_pos"])), "kf_pos_mean": float(np.nanmean(r["kf_pos"])),
                                             "pnp_rot_mean": float(np.nanmean(r["pnp_rot"])), "kf_rot_mean": float(np.nanmean(r["kf_rot"])),
                                             "gate_rate_pct": 100 * r["gated"] / max(r["offered"], 1), "reinit": r["reinit"]})
            if seed == 0:
                curves[noise] = runs
        results[f"{noise:g}"] = aggregate(per_seed)
        per_seed_all[f"{noise:g}"] = per_seed
        print(f"  noise {noise:g} px done")

    # ---- dt sensitivity (2 px) ----------------------------------------------------------------------
    sens = {}
    for dt in [cfg.DT] + cfg.DT_SENSITIVITY:
        ps = [summarise({s: run_scene(cache[s], s, 2.0, seed, calib[2.0]["bins"], dt=dt) for s in cache})
              for seed in cfg.SEEDS]
        sens[f"{dt:g}"] = aggregate(ps)["all"]
    nan_check = bool(all(np.isfinite(r["kf_pos"]).all() and np.isfinite(r["kf_rot"]).all()
                         for runs in curves.values() for r in runs.values()))

    # ---- tables ---------------------------------------------------------------------------------------
    def f(v, nd=2):
        return f"{v[0]:.{nd}f} ± {v[1]:.{nd}f}"

    print("\nPOSE ERRORS (all 2400 frames, mean over 5 seeds ± std over seeds)")
    print(f"{'noise':>5s} {'method':32s} {'trans mean [m]':>16s} {'trans med':>10s} {'rot mean [deg]':>16s} {'rot med':>9s} "
          f"{'fail %':>8s} {'gate %':>7s} {'reinit':>7s}")
    for noise in cfg.NOISES:
        a = results[f"{noise:g}"]["all"]
        print(f"{noise:5g} {PNP:32s} {f(a['pnp_pos_mean']):>16s} {a['pnp_pos_median'][0]:10.2f} {f(a['pnp_rot_mean']):>16s} "
              f"{a['pnp_rot_median'][0]:9.2f} {a['pnp_fail_pct'][0]:8.2f}")
        print(f"{'':5s} {KF:32s} {f(a['kf_pos_mean']):>16s} {a['kf_pos_median'][0]:10.2f} {f(a['kf_rot_mean']):>16s} "
              f"{a['kf_rot_median'][0]:9.2f} {'':8s} {a['gate_rate_pct'][0]:7.2f} {a['reinit_total'][0]:7.1f}")
    print("\nBY DISTANCE (mean translation error [m] / rotation error [deg])")
    for noise in cfg.NOISES:
        for m, name in (("pnp", PNP), ("kf", KF)):
            cells = [f"{results[f'{noise:g}'][b][f'{m}_pos_mean'][0]:7.2f} / {results[f'{noise:g}'][b][f'{m}_rot_mean'][0]:5.2f}"
                     for b in BIN_LABELS]
            print(f"{noise:5g} {name:32s} " + "  ".join(cells))
    print("frames per bin:", [results["2"][b]["n_frames"][0] for b in BIN_LABELS])
    print("\nDT SENSITIVITY (2 px, Kalman filter)")
    for dt, a in sens.items():
        print(f"  dt = {dt} s: trans {f(a['kf_pos_mean'])} m, rot {f(a['kf_rot_mean'])} deg, gate {a['gate_rate_pct'][0]:.2f} %, "
              f"re-inits {a['reinit_total'][0]:.1f}")
    per_scene = {s: {k: float(np.mean([x[k] for x in v])) for k in v[0]} for s, v in per_scene_2px.items()}
    print("\nPER SCENE at 2 px (mean over seeds): range, PnP / Kalman translation [m], rotation [deg], gate %, re-inits")
    for s, d in per_scene.items():
        print(f"  {s} {d['range_min_m']:3.0f}-{d['range_max_m']:3.0f} m  trans {d['pnp_pos_mean']:5.2f} / {d['kf_pos_mean']:5.2f}   "
              f"rot {d['pnp_rot_mean']:5.2f} / {d['kf_rot_mean']:5.2f}   gate {d['gate_rate_pct']:5.1f}   re-inits {d['reinit']:4.1f}")
    print("Kalman results free of NaN:", nan_check)

    with open(os.path.join(OUT_DIR, "uav_metrics.json"), "w") as fh:
        json.dump({"settings": {"noises_px": cfg.NOISES, "n_seeds": len(cfg.SEEDS), "dt_s": cfg.DT,
                                "range_bins_m": [150, 250, 350, 450, None], "n_frames": int(sum(len(c["frames"]) for c in cache.values())),
                                "glitch_splits": cfg.GLITCH_SPLITS, "stats": "[mean over seeds, std over seeds]"},
                   "visibility": vis, "results": results, "per_scene_2px": per_scene, "dt_sensitivity_2px": sens, "kalman_no_nan": nan_check}, fh, indent=1)

    # ---- figures ---------------------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    x = np.arange(len(cfg.NOISES))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, e, label in [(axes[0], "pos", "Mean translation error [m]"), (axes[1], "rot", "Mean rotation error [deg]")]:
        for k, (m, name) in enumerate((("pnp", PNP), ("kf", KF))):
            vals = np.array([results[f"{n:g}"]["all"][f"{m}_{e}_mean"] for n in cfg.NOISES])
            ax.errorbar(x + (k - 0.5) * 0.05, np.maximum(vals[:, 0], 1e-3), yerr=vals[:, 1], marker="o", capsize=3,
                        color=COLORS[name], label=name)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n:g}" for n in cfg.NOISES])
        ax.set_xlabel("Keypoint noise [px]")
        ax.set_ylabel(label)
        ax.grid(which="both", alpha=0.25)
    axes[0].legend()
    fig.suptitle("FW-UAV6DPose (2400 frames, 164-518 m): error vs noise (mean over 5 seeds, bars = std)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "uav_error_vs_noise.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, key, label in [(axes[0], "pos_mean", "Mean translation error [m]"),
                           (axes[1], "pos_rel_mean_pct", "Mean translation error [% of range]"),
                           (axes[2], "rot_mean", "Mean rotation error [deg]")]:
        for ni, noise in enumerate(cfg.NOISES[1:]):
            col = plt.cm.viridis(ni / 2.5)
            for m, ls, name in (("pnp", "--", PNP), ("kf", "-", KF)):
                vals = [results[f"{noise:g}"][b][f"{m}_{key}"][0] for b in BIN_LABELS]
                ax.plot(range(4), vals, ls, marker="o", color=col, label=f"{noise:g} px, {name}")
        ax.set_xticks(range(4))
        ax.set_xticklabels(BIN_LABELS)
        ax.set_xlabel("True distance")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.grid(which="both", alpha=0.25)
    axes[2].legend(fontsize=7, ncol=1)
    fig.suptitle("Error vs distance (dashed: PnP only, solid: Kalman filter; 1, 2 and 5 px)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "uav_error_vs_distance.png"), dpi=200)
    plt.close(fig)

    scene = "000012"
    r = curves[2.0][scene]
    c = cache[scene]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(c["t"][:, 2], "k-", label="true")
    axes[0].plot(r["pnp_est"][:, 2], ".", color=COLORS[PNP], ms=4, label=PNP)
    axes[0].plot(r["kf_est"][:, 2], "-", color=COLORS[KF], label=KF)
    axes[0].set_ylabel("Depth z in camera frame [m]")
    axes[1].plot(r["pnp_pos"], ".", color=COLORS[PNP], ms=4, label=PNP)
    axes[1].plot(r["kf_pos"], "-", color=COLORS[KF], label=KF)
    axes[1].set_ylabel("Translation error [m]")
    axes[2].plot(r["pnp_rot"], ".", color=COLORS[PNP], ms=4, label=PNP)
    axes[2].plot(r["kf_rot"], "-", color=COLORS[KF], label=KF)
    axes[2].set_ylabel("Rotation error [deg]")
    axes[2].set_xlabel("Frame")
    axes[0].legend()
    fig.suptitle(f"Scene {scene}, 2 px noise (seed 0), dt = 0.1 s")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "uav_sequence_000012.png"), dpi=200)
    plt.close(fig)
    print("\nSaved uav_metrics.json, uav_noise_calibration.json, uav_error_vs_noise.png, uav_error_vs_distance.png, "
          "uav_sequence_000012.png in results/phase7a/")


if __name__ == "__main__":
    main()
