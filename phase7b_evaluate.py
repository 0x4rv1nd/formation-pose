"""
Phase 7b, part B, step 2: evaluate the trained YOLO-pose detector and the image -> pose pipeline on the TEST scenes.

    python phase7b_detect.py --split val ; python phase7b_tune_val.py     # choices on val (results/phase7b/val_choices.json)
    python phase7b_detect.py --split test
    python phase7b_evaluate.py

Reads the choices from results/phase7b/val_choices.json and changes nothing about them. Writes results/phase7b/test_results.json,
val_results.json (the same numbers on the val scenes, for the README) and the figures. Needs data/fw_uav/pred_{val,test}.npz.
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
import phase7a_evaluate as p7a
import phase7b_detect as detect
import phase7b_eval as ev
import phase7b_prepare as prep

OUT_DIR = os.path.join("results", "phase7b")
CHOICES_PATH = os.path.join(OUT_DIR, "val_choices.json")
SYNTH_NOISES = [0.0, 1.0, 2.0, 5.0, 10.0]            # Phase 7a comparison (with Kalman filter)
EFFECTIVE_GRID = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0]       # PnP only, to locate the detector's effective noise
SEEDS = range(5)
SEQUENCE_SCENE = "000029"          # fixed in advance (the middle-range test scene)
LABEL = {"a_sqpnp": "(a) SQPnP", "b_ransac": "(b) RANSAC PnP", "c_sqpnp_sym": "(c) symmetry-aware SQPnP",
         "c_ransac_sym": "(c) symmetry-aware RANSAC", "kalman": "(d) Kalman filter"}
COLOR = {"a_sqpnp": "tab:orange", "b_ransac": "tab:blue", "c_sqpnp_sym": "tab:purple", "c_ransac_sym": "tab:brown",
         "kalman": "tab:green"}
BIN_LABELS = ["150-250 m", "250-350 m", "350-450 m", "450+ m"]


def evaluate(split, choices):
    scenes = prep.load_split()[split]
    sp = prep.load_split()
    g = ev.load_ground_truth(scenes)
    pred = detect.load_predictions(split)
    assert sorted(pred) == sorted(scenes), "cached predictions do not match the split"
    K = g[scenes[0]]["K"]
    det = {s: ev.select_detection(pred[s], choices["box_threshold"]) for s in scenes}
    res = {"scenes": scenes, "diagnostics": ev.keypoint_diagnostics(det, g)}

    # per scene context
    near = {s: ev.nearest_train_rotation_deg(g[s]["R"], sp["train"]) for s in scenes}
    res["per_scene_context"] = {}
    for s in scenes:
        e = np.linalg.norm(det[s]["kpts"] - g[s]["uv"], axis=-1)
        o = ev.oks(np.where(det[s]["has"][:, None, None], det[s]["kpts"], 0.0), g[s]["uv"], g[s]["vis_dilated"], g[s]["box"])
        res["per_scene_context"][s] = {
            "range_m": [float(g[s]["range"].min()), float(g[s]["range"].max())], "detection_rate": float(det[s]["has"].mean()),
            "keypoint_error_px_median": float(np.median(e[g[s]["vis_dilated"] & det[s]["has"][:, None]])),
            "keypoint_error_px_p90": float(np.percentile(e[g[s]["vis_dilated"] & det[s]["has"][:, None]], 90)),
            "oks_gt_0.5": float((o > 0.5).mean()),
            "oks_gt_0.5_with_whole_aircraft_mirror_oracle": float((np.maximum(o, ev.oks(np.where(det[s]["has"][:, None, None], det[s]["kpts"], 0.0)[:, ev.FLIP], g[s]["uv"], g[s]["vis_dilated"], g[s]["box"])) > 0.5).mean()),
            "lr_swap_rate": float(np.nanmean(ev.partner_swap(det[s]["kpts"], g[s]["uv"], g[s]["vis_dilated"], ev.FLIP))),
            "nearest_train_rotation_deg_median": float(np.median(near[s])),
            "nearest_train_rotation_deg_max": float(near[s].max())}

    # single-frame methods (settings from val_choices.json)
    kp, rp = choices["keypoint_conf_threshold"], choices["ransac_reproj_px"]
    cfgs = {"a_sqpnp": ("sqpnp", False), "b_ransac": ("ransac", False), "c_sqpnp_sym": ("sqpnp", True), "c_ransac_sym": ("ransac", True)}
    runs = {}
    for name, (method, sym) in cfgs.items():
        runs[name] = {}
        for s in scenes:
            est = ev.pose_sequence(det[s], K, kp, method, sym, rp)
            pos, rot = ev.errors(est, g[s])
            runs[name][s] = {"range": g[s]["range"], "est": est, "m_pos": pos, "m_rot": rot}
    # Kalman filter on the best single-frame method chosen on val
    best = choices["best_single_frame_method"]
    bins = meth.calibrate_range_noise(choices["kalman_calibration_noise_px"], K)["bins"]
    runs["kalman"] = {}
    for s in scenes:
        est, info = ev.kalman_sequence(runs[best][s]["est"], bins)
        pos, rot = ev.errors(est, g[s])
        runs["kalman"][s] = {"range": g[s]["range"], "est": est, "m_pos": pos, "m_rot": rot, **info}
    res["best_single_frame_method"] = best
    res["methods"] = {n: ev.summarise(r, "m") for n, r in runs.items()}
    res["methods_per_scene"] = {n: {s: ev.summarise({s: r[s]}, "m")["all"] for s in scenes} for n, r in runs.items()}
    kf = runs["kalman"]
    res["kalman_gate_rate_pct"] = float(100 * sum(r["gated"] for r in kf.values()) / max(sum(r["offered"] for r in kf.values()), 1))
    res["kalman_reinit_total"] = int(sum(r["reinit"] for r in kf.values()))

    # Phase 7a comparison on the SAME scenes: projected keypoints + Gaussian noise, PnP and Kalman filter
    cache = io.build_keypoint_cache(cfg.KEYPOINTS)
    synth = {}
    for n in SYNTH_NOISES:
        cal = meth.calibrate_range_noise(max(n, 1.0), K)["bins"]
        rows_p, rows_k, gates = [], [], []
        for seed in SEEDS:
            rr = {s: p7a.run_scene(cache[s], s, n, seed, cal) for s in scenes}
            rows_p.append(ev.summarise({s: {"range": r["range"], "p_pos": r["pnp_pos"], "p_rot": r["pnp_rot"]} for s, r in rr.items()}, "p")["all"])
            rows_k.append(ev.summarise({s: {"range": r["range"], "k_pos": r["kf_pos"], "k_rot": r["kf_rot"]} for s, r in rr.items()}, "k")["all"])
            gates.append(100 * sum(r["gated"] for r in rr.values()) / max(sum(r["offered"] for r in rr.values()), 1))
        mean = lambda rows: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        synth[f"{n:g}"] = {"pnp": mean(rows_p), "kalman": mean(rows_k), "gate_rate_pct": float(np.mean(gates))}
    res["synthetic_phase7a"] = synth
    # synthetic PnP by distance bin (for the figure) and the effective noise
    eff_rows = {}
    for n in EFFECTIVE_GRID:
        rows = []
        for seed in range(3):
            m = {}
            for s in scenes:
                p, r = ev.synthetic_pnp(cache[s], n, seed)
                m[s] = {"range": g[s]["range"], "p_pos": p, "p_rot": r}
            rows.append(ev.summarise(m, "p")["all"])
        eff_rows[n] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    res["synthetic_pnp_grid"] = {str(k): v for k, v in eff_rows.items()}
    d = res["methods"][best]["all"]
    xs = np.array(EFFECTIVE_GRID)
    res["effective_noise_px"] = {}
    for key in ("pos_median", "rot_median", "fail_pct"):
        ys = np.array([eff_rows[n][key] for n in EFFECTIVE_GRID])
        ys = np.maximum.accumulate(ys)
        v = d[key]
        res["effective_noise_px"][key] = (float(np.interp(v, ys, xs)) if ys[0] <= v <= ys[-1] else
                                          (f"> {xs[-1]:g}" if v > ys[-1] else f"< {xs[0]:g}"))
    res["effective_noise_px"]["from_keypoint_median_error"] = res["diagnostics"]["error_px_all"]["median"] / 1.1774
    return res, dict(g=g, det=det, runs=runs, near=near)


def fmt_table(res):
    lines = ["| Method | Failure % | Trans. mean / median [m] | Rot. mean / median [deg] | No solution % |", "|---|---|---|---|---|"]
    for n in LABEL:
        s = res["methods"][n]["all"]
        lines.append(f"| {LABEL[n]} | {s['fail_pct']:.1f} | {s['pos_mean']:.1f} / {s['pos_median']:.1f} | {s['rot_mean']:.1f} / {s['rot_median']:.1f} | {s['no_solution_pct']:.1f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_overlays(ctx, scenes):
    import cv2, zipfile
    fig, axes = plt.subplots(2, 2, figsize=(17, 9))
    zf = zipfile.ZipFile(io.ZIP_PATH)
    for ax, s in zip(axes.ravel(), scenes):
        g, det = ctx["g"][s], ctx["det"][s]
        i = len(g["frames"]) // 2
        f = int(g["frames"][i])
        img = cv2.imdecode(np.frombuffer(zf.read(f"val/{s}/rgb/{f:06d}.png"), np.uint8), 1)[..., ::-1]
        b = g["box"][i]
        pad = 50
        ax.imshow(img)
        ax.set_xlim(b[0] - pad, b[2] + pad)
        ax.set_ylim(b[3] + pad, b[1] - pad)
        gt, pr, vis = g["uv"][i], det["kpts"][i], g["vis_dilated"][i]
        own = np.linalg.norm(pr - gt, axis=1)
        other = np.linalg.norm(pr - gt[ev.FLIP], axis=1)
        swapped = (other < own) & (np.array(ev.FLIP) != np.arange(13)) & vis
        ax.scatter(*gt[vis].T, c="lime", s=28, label="true (projected)")
        for k in range(13):
            if vis[k]:
                ax.plot([gt[k, 0], pr[k, 0]], [gt[k, 1], pr[k, 1]], c="orange" if swapped[k] else "yellow", lw=0.8)
        ax.scatter(*pr[~swapped].T, c="red", marker="x", s=28, label="detected")
        ax.scatter(*pr[swapped].T, c="magenta", marker="D", s=30, label="detected, closer to its mirror partner's truth (swap)")
        ax.set_title(f"{s}  frame {f}  range {g['range'][i]:.0f} m  median keypoint error {np.median(own[vis]):.0f} px")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Test scenes: detector keypoints against the projected ground truth (middle frame of each scene)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "test_overlays.png"), dpi=110)
    plt.close(fig)


def fig_hist(res_val, res_test, ctx_val, ctx_test):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, ctx, c in (("val", ctx_val, "tab:blue"), ("test", ctx_test, "tab:red")):
        e = np.concatenate([np.linalg.norm(ctx["det"][s]["kpts"] - ctx["g"][s]["uv"], axis=-1)[ctx["g"][s]["vis_dilated"] & ctx["det"][s]["has"][:, None]] for s in ctx["g"]])
        ax.hist(e, bins=np.arange(0, 260, 5), alpha=0.55, color=c, density=True, label=f"{name}: median {np.median(e):.0f} px, p90 {np.percentile(e, 90):.0f} px")
    ax.axvline(21.5, c="k", ls="--", lw=1)
    ax.text(23, ax.get_ylim()[1] * 0.9, "OKS 0.5 on the\nmedian box (21 px)", fontsize=8)
    ax.set_xlabel("keypoint pixel error")
    ax.set_ylabel("density")
    ax.set_title("Detector keypoint error (visible keypoints)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "test_keypoint_error_hist.png"), dpi=130)
    plt.close(fig)


def fig_swaps(res_val, res_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, key, title in ((axes[0], "lr_swap_rate_by_keypoint", "left <-> right (closer to the mirror partner's truth)"),
                           (axes[1], "edge_swap_rate_by_keypoint", "leading <-> trailing edge")):
        for off, (lab, r, c) in zip((-0.2, 0.2), (("val", res_val, "tab:blue"), ("test", res_test, "tab:red"))):
            d = r["diagnostics"][key]
            ax.bar(np.arange(len(d)) + off, [100 * (d.get(n) or 0) for n in d], 0.4, color=c, label=lab)
            ax.set_xticks(np.arange(len(d)))
            ax.set_xticklabels(list(d), rotation=60, ha="right", fontsize=8)
        ax.axhline(50, c="k", ls=":", lw=1)
        ax.set_title(title)
        ax.set_ylabel("swap rate [%] (50 % = chance)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "test_swap_rate_by_keypoint.png"), dpi=130)
    plt.close(fig)


def fig_vs_distance(res):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    xs = np.arange(4)
    for ax, key, ylabel in ((axes[0], "pos_mean", "mean translation error [m]"), (axes[1], "rot_mean", "mean rotation error [deg]"),
                            (axes[2], "fail_pct", "failure rate [%]")):
        for n in LABEL:
            ys = [res["methods"][n].get(b, {}).get(key, np.nan) or np.nan for b in BIN_LABELS]
            ax.plot(xs, ys, "o-", c=COLOR[n], label=LABEL[n])
        ax.set_xticks(xs)
        ax.set_xticklabels(BIN_LABELS)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    n_fr = [res["methods"]["kalman"].get(b, {}).get("n_frames", 0) for b in BIN_LABELS]
    fig.suptitle(f"Test scenes: pose error against distance (frames per bin {n_fr}; empty bins are not drawn)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "test_error_vs_distance.png"), dpi=130)
    plt.close(fig)


def fig_sequence(ctx, res, scene):
    g, runs = ctx["g"][scene], ctx["runs"]
    best = res["best_single_frame_method"]
    t = np.arange(len(g["frames"])) * cfg.DT
    truth = np.array([np.concatenate([g["t"][i], meth.rotation_to_pose_angles(g["R"][i])]) for i in range(len(t))])
    fig, axes = plt.subplots(2, 3, figsize=(18, 7), sharex=True)
    labels = ["x [m]", "y [m]", "z [m]", "roll [deg]", "pitch [deg]", "yaw [deg]"]
    for k, ax in enumerate(axes.ravel()):
        ax.plot(t, truth[:, k], "k-", lw=2, label="truth")
        ax.plot(t, runs[best][scene]["est"][:, k], ".", c=COLOR[best], ms=4, label=f"best single frame: {LABEL[best]}")
        ax.plot(t, runs["kalman"][scene]["est"][:, k], "-", c=COLOR["kalman"], lw=1.5, label="Kalman filter")
        ax.set_ylabel(labels[k])
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("time [s] (dt = 0.1 s assumed)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Test scene {scene}: truth, best single-frame method and Kalman filter")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"test_sequence_{scene}.png"), dpi=120)
    plt.close(fig)


def main():
    assert os.path.exists(CHOICES_PATH), "run phase7b_tune_val.py first: all choices must come from the validation scenes"
    choices = json.load(open(CHOICES_PATH))
    assert choices["scenes_used"] == prep.load_split()["val"]
    res_val, ctx_val = evaluate("val", choices)
    res_test, ctx_test = evaluate("test", choices)
    for name, r in (("val", res_val), ("test", res_test)):
        json.dump(r, open(os.path.join(OUT_DIR, f"{name}_results.json"), "w"), indent=1)
    fig_overlays(ctx_test, res_test["scenes"])
    fig_hist(res_val, res_test, ctx_val, ctx_test)
    fig_swaps(res_val, res_test)
    fig_vs_distance(res_test)
    fig_sequence(ctx_test, res_test, SEQUENCE_SCENE)

    for name, r in (("VAL", res_val), ("TEST", res_test)):
        d = r["diagnostics"]
        print(f"\n=========== {name} ===========")
        print(f"detection rate {100 * d['detection_rate']:.1f} %, median box IoU {d['box_iou_median']:.2f}")
        print(f"keypoint error px: median {d['error_px_all']['median']:.1f}, p90 {d['error_px_all']['p90']:.1f}; by distance bin:",
              {k: (round(v['median'], 1), round(v['p90'], 1)) for k, v in d['error_px_by_bin'].items() if v})
        print(f"L/R swap rate {100 * d['lr_swap_rate']:.1f} %, leading/trailing {100 * d['edge_swap_rate']:.1f} %, frames with majority L/R swap {100 * d['frames_with_majority_lr_swap']:.1f} %")
        print("L/R swap by distance bin:", {k: (None if v is None else round(100 * v, 1)) for k, v in d['lr_swap_rate_by_bin'].items()})
        print("error by keypoint confidence (median px):", {k: (None if v is None else (v['n'], round(v['median'], 1))) for k, v in d['error_by_confidence'].items()})
        print("OKS:", {k: {m: round(x, 3) for m, x in v.items()} for k, v in d['oks'].items()})
        print(f"px error for OKS 0.5 / 0.75 on the median box ({d['gt_box_width_px_median']:.0f} x {d['gt_box_height_px_median']:.0f}): "
              f"{d['px_error_for_oks_0.5_median_box']:.1f} / {d['px_error_for_oks_0.75_median_box']:.1f}")
        print("per scene:")
        for s, c in r["per_scene_context"].items():
            m = r["methods_per_scene"][r["best_single_frame_method"]][s]
            k = r["methods_per_scene"]["kalman"][s]
            print(f"  {s} range {c['range_m'][0]:.0f}-{c['range_m'][1]:.0f} m | det {100 * c['detection_rate']:.0f}% | kp err med/p90 {c['keypoint_error_px_median']:.0f}/{c['keypoint_error_px_p90']:.0f} px "
                  f"| OKS>0.5 {100 * c['oks_gt_0.5']:.0f}% (mirror oracle {100 * c['oks_gt_0.5_with_whole_aircraft_mirror_oracle']:.0f}%, L/R swap {100 * c['lr_swap_rate']:.0f}%) | nearest-train rot {c['nearest_train_rotation_deg_median']:.0f} deg (max {c['nearest_train_rotation_deg_max']:.0f}) "
                  f"| best single: fail {m['fail_pct']:.0f}%, pos med {m['pos_median']:.1f} m, rot med {m['rot_median']:.0f} deg "
                  f"| KF: fail {k['fail_pct']:.0f}%, pos med {k['pos_median']:.1f} m")
        print(fmt_table(r))
        print(f"Kalman gate rate {r['kalman_gate_rate_pct']:.1f} %, re-initialisations {r['kalman_reinit_total']}")
        print("by distance (mean pos m / rot deg / fail %):")
        for n in LABEL:
            print(f"  {LABEL[n]:28s}", "  ".join(f"{b}: {r['methods'][n][b]['pos_mean']:.1f}/{r['methods'][n][b]['rot_mean']:.0f}/{r['methods'][n][b]['fail_pct']:.0f}" for b in BIN_LABELS if b in r['methods'][n] and r['methods'][n][b]['pos_mean'] is not None))
        print("Phase 7a synthetic (same scenes): PnP / Kalman  fail %, trans mean/median m, rot mean/median deg")
        for n, v in r["synthetic_phase7a"].items():
            p, k = v["pnp"], v["kalman"]
            print(f"  {n:>3s} px  PnP {p['fail_pct']:.1f}%, {p['pos_mean']:.2f}/{p['pos_median']:.2f} m, {p['rot_mean']:.1f}/{p['rot_median']:.1f} deg | "
                  f"KF {k['fail_pct']:.1f}%, {k['pos_mean']:.2f}/{k['pos_median']:.2f} m, {k['rot_mean']:.1f}/{k['rot_median']:.1f} deg, gate {v['gate_rate_pct']:.1f}%")
        print("synthetic PnP grid:", {k: (round(v['fail_pct'], 1), round(v['pos_median'], 1), round(v['rot_median'], 1)) for k, v in r['synthetic_pnp_grid'].items()})
        print("effective noise [px]:", r["effective_noise_px"])


if __name__ == "__main__":
    main()
