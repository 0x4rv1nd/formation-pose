"""
Phase 7b, part B, step 1: everything that is CHOSEN is chosen here, on the VALIDATION scenes only (000042-000045).

    python phase7b_tune_val.py

Writes results/phase7b/val_diagnostics.json and results/phase7b/val_choices.json (box threshold, keypoint-confidence
threshold, RANSAC reprojection threshold, best single-frame method, Kalman noise level). phase7b_evaluate.py reads the
choices and runs the test scenes once. Selection rule for every threshold / method: lowest failure rate (no solution, or
position error > 10 % of the range, or rotation error > 20 deg), ties (within 0.5 percentage points) by lower mean
translation error.
"""

import json
import os

import numpy as np

import fw_uav_config as cfg
import fw_uav_io as io
import fw_uav_methods as meth
import phase7a_evaluate as p7a
import phase7b_detect as detect
import phase7b_eval as ev
import phase7b_prepare as prep

OUT_DIR = os.path.join("results", "phase7b")
BOX_GRID = [0.1, 0.25, 0.5, 0.7]
KP_GRID = [0.0, 0.95, 0.98, 0.99, 0.995]
RANSAC_GRID = [4.0, 8.0, 16.0, 32.0, 64.0]
KF_NOISE_GRID = [5.0, 10.0, 20.0, 40.0]
SYNTH_NOISES = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
SEEDS = range(3)
FAIL_TIE_PCT = 0.5


def score(per_scene, key):
    s = ev.summarise(per_scene, key)["all"]
    return s["fail_pct"], (s["pos_mean"] if s["pos_mean"] is not None else np.inf)


def pick(options):
    """options: {name: (fail_pct, mean_pos)} -> name with the lowest failure rate (ties within FAIL_TIE_PCT by mean position error)."""
    best_fail = min(v[0] for v in options.values())
    cands = [k for k, v in options.items() if v[0] <= best_fail + FAIL_TIE_PCT]
    return min(cands, key=lambda k: options[k][1])


def run_single(det, g, K, kp_thr, method, sym, ransac_px=8.0):
    out = {}
    for s in g:
        est = ev.pose_sequence(det[s], K, kp_thr, method, sym, ransac_px)
        pos, rot = ev.errors(est, g[s])
        out[s] = {"range": g[s]["range"], "est": est, "m_pos": pos, "m_rot": rot}
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scenes = prep.load_split()["val"]
    g = ev.load_ground_truth(scenes)
    pred = detect.load_predictions("val")
    K = g[scenes[0]]["K"]
    log = {}

    # ---- box threshold -------------------------------------------------------------------------
    box_rows = {}
    for thr in BOX_GRID:
        det = {s: ev.select_detection(pred[s], thr) for s in scenes}
        has = np.concatenate([det[s]["has"] for s in scenes])
        iou = np.concatenate([ev.box_iou(det[s]["box"], g[s]["box"]) for s in scenes])
        box_rows[thr] = {"detection_rate": float(has.mean()), "correct_box_rate": float((has & (iou >= 0.5)).mean()),
                         "false_box_rate": float((has & (iou < 0.5)).mean())}
    best_correct = max(r["correct_box_rate"] for r in box_rows.values())
    box_thr = max(t for t, r in box_rows.items() if r["correct_box_rate"] >= best_correct - 0.005)   # highest that loses < 0.5 %
    log["box_threshold_table"] = {str(k): v for k, v in box_rows.items()}
    print("box threshold -> detection / correct (IoU>=0.5) / false:", {k: tuple(round(x, 3) for x in v.values()) for k, v in box_rows.items()})
    print("chosen box threshold:", box_thr)
    det = {s: ev.select_detection(pred[s], box_thr) for s in scenes}

    # ---- keypoint diagnostics ---------------------------------------------------------------------
    diag = ev.keypoint_diagnostics(det, g)
    json.dump(diag, open(os.path.join(OUT_DIR, "val_diagnostics.json"), "w"), indent=1)

    # ---- (a) SQPnP: keypoint-confidence threshold ---------------------------------------------------
    a = {t: score(run_single(det, g, K, t, "sqpnp", False), "m") for t in KP_GRID}
    kp_thr = pick(a)
    print("(a) SQPnP, kp-conf threshold -> (fail %, mean pos m):", {k: tuple(round(x, 2) for x in v) for k, v in a.items()}, "chosen", kp_thr)

    # ---- (b) RANSAC reprojection threshold -------------------------------------------------------------
    b = {r: score(run_single(det, g, K, kp_thr, "ransac", False, r), "m") for r in RANSAC_GRID}
    ransac_px = pick(b)
    print("(b) RANSAC, reprojection threshold [px] -> (fail %, mean pos m):", {k: tuple(round(x, 2) for x in v) for k, v in b.items()}, "chosen", ransac_px)

    # ---- (c) symmetry-aware ---------------------------------------------------------------------------
    singles = {"a_sqpnp": run_single(det, g, K, kp_thr, "sqpnp", False),
               "b_ransac": run_single(det, g, K, kp_thr, "ransac", False, ransac_px),
               "c_sqpnp_sym": run_single(det, g, K, kp_thr, "sqpnp", True),
               "c_ransac_sym": run_single(det, g, K, kp_thr, "ransac", True, ransac_px)}
    sc = {k: score(v, "m") for k, v in singles.items()}
    best = pick(sc)
    print("single-frame methods -> (fail %, mean pos m):", {k: tuple(round(x, 2) for x in v) for k, v in sc.items()}, "chosen", best)
    val_summ = {k: ev.summarise(v, "m") for k, v in singles.items()}

    # ---- (d) Kalman filter: noise level of the synthetic R calibration ----------------------------------------
    kf_rows, kf_runs = {}, {}
    for n in KF_NOISE_GRID:
        try:
            bins = meth.calibrate_range_noise(n, K)["bins"]
        except RuntimeError as e:
            print(f"  calibration at {n} px failed: {e}")
            continue
        per = {}
        for s in scenes:
            est, info = ev.kalman_sequence(singles[best][s]["est"], bins)
            pos, rot = ev.errors(est, g[s])
            per[s] = {"range": g[s]["range"], "k_pos": pos, "k_rot": rot, **info}
        kf_runs[n] = per
        kf_rows[n] = score(per, "k")
    kf_noise = pick(kf_rows)
    print("(d) Kalman R calibrated at noise [px] -> (fail %, mean pos m):", {k: tuple(round(x, 2) for x in v) for k, v in kf_rows.items()}, "chosen", kf_noise)
    val_kf = ev.summarise(kf_runs[kf_noise], "k")

    # ---- effective noise (Phase 7a protocol on the same scenes) ---------------------------------------------------
    cache = {s: io.build_keypoint_cache(cfg.KEYPOINTS)[s] for s in scenes}
    synth = {}
    for n in SYNTH_NOISES:
        try:
            bins = meth.calibrate_range_noise(n, K)["bins"] if n > 0 else meth.calibrate_range_noise(1.0, K)["bins"]
        except RuntimeError:
            continue
        rows = []
        for seed in SEEDS:
            runs = {s: p7a.run_scene(cache[s], s, n, seed, bins) for s in scenes}
            m = {s: {"range": r["range"], "p_pos": r["pnp_pos"], "p_rot": r["pnp_rot"]} for s, r in runs.items()}
            rows.append(ev.summarise(m, "p")["all"])
        synth[n] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    det_all = val_summ[best]["all"]
    eff = {}
    for key in ("pos_mean", "pos_median", "rot_mean", "rot_median"):
        xs = np.array([n for n in synth if n > 0]); ys = np.array([synth[n][key] for n in xs])
        eff[key] = float(np.exp(np.interp(np.log(det_all[key]), np.log(ys), np.log(xs)))) if ys.min() <= det_all[key] <= ys.max() else None
    err_all = diag["error_px_all"]
    eff["from_keypoint_median_error_px"] = err_all["median"] / 1.1774          # median radial error of 2-D Gaussian noise = 1.1774 sigma
    print("effective noise [px] (Phase 7a PnP matched on val):", eff)
    print("synthetic PnP on val scenes:", {n: (round(v["pos_mean"], 2), round(v["rot_mean"], 1)) for n, v in synth.items()})

    choices = {"scenes_used": scenes, "box_threshold": box_thr, "keypoint_conf_threshold": kp_thr, "ransac_reproj_px": ransac_px,
               "best_single_frame_method": best, "kalman_calibration_noise_px": kf_noise,
               "selection_rule": f"lowest failure rate, ties within {FAIL_TIE_PCT} points by mean translation error; box threshold: highest losing < 0.5 % correct boxes",
               "grids": {"box": BOX_GRID, "keypoint_conf": KP_GRID, "ransac_px": RANSAC_GRID, "kalman_noise_px": KF_NOISE_GRID}}
    json.dump(choices, open(os.path.join(OUT_DIR, "val_choices.json"), "w"), indent=1)
    json.dump({"box": log, "a_sqpnp_by_kp_thr": {str(k): v for k, v in a.items()}, "b_ransac_by_px": {str(k): v for k, v in b.items()},
               "single_frame": val_summ, "kalman_by_noise": {str(k): v for k, v in kf_rows.items()}, "kalman": val_kf,
               "synthetic_on_val": {str(k): v for k, v in synth.items()}, "effective_noise_px": eff},
              open(os.path.join(OUT_DIR, "val_tuning.json"), "w"), indent=1)
    print("\nVAL results of chosen methods:")
    for k, v in val_summ.items():
        s = v["all"]
        print(f"  {k:14s} fail {s['fail_pct']:5.1f}%  pos mean/med {s['pos_mean']:.1f}/{s['pos_median']:.1f} m  rot mean/med {s['rot_mean']:.1f}/{s['rot_median']:.1f} deg  no-solution {s['no_solution_pct']:.1f}%")
    s = val_kf["all"]
    print(f"  {'kalman':14s} fail {s['fail_pct']:5.1f}%  pos mean/med {s['pos_mean']:.1f}/{s['pos_median']:.1f} m  rot mean/med {s['rot_mean']:.1f}/{s['rot_median']:.1f} deg  gate {100 * sum(r['gated'] for r in kf_runs[kf_noise].values()) / max(sum(r['offered'] for r in kf_runs[kf_noise].values()), 1):.1f}%")


if __name__ == "__main__":
    main()
