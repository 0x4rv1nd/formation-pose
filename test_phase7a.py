"""Sanity checks for Phase 7a (prints PASS/FAIL). Needs data/fw_uav/ (val.zip, val_meta/, models/) and `python phase7a_keypoints.py`."""

import sys

import numpy as np

import fw_uav_config as cfg
import fw_uav_io as io
import fw_uav_methods as meth
import phase7a_evaluate as ev
import phase7a_model_check as mc

results = []


def check(name, ok):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


cache = io.build_keypoint_cache(cfg.KEYPOINTS)
K = cache[io.scenes()[0]]["K"]

# --- keypoints and model correction ----------------------------------------------------------
check(f"{len(cfg.KEYPOINTS)} keypoints defined, finite, within the UAV's extent (20 m x 32 m x 5 m)",
      len(cfg.KEYPOINTS) == 13 and np.isfinite(cfg.KEYPOINTS).all() and np.abs(cfg.KEYPOINTS).max(axis=0)[1] < 17
      and np.abs(cfg.KEYPOINTS).max(axis=0)[0] < 20)
V, F = io.load_obj_vertices()
Vc = V * cfg.MODEL_UNIT_TO_M @ cfg.MODEL_ROTATION.T
d = np.min(np.linalg.norm(Vc[None, :, :] - cfg.KEYPOINTS[:, None, :], axis=2), axis=1)
check(f"every keypoint is a vertex of the corrected model (max distance {d.max():.4f} m)", d.max() < 0.01)
ids, R, t, _ = io.load_scene("000009")
mask = io.read_mask("000009", 0)
iou_fixed = mc.iou(mc.silhouette(Vc, F, R[0], t[0], K), mask)
iou_raw = mc.iou(mc.silhouette(V * cfg.MODEL_UNIT_TO_M, F, R[0], t[0], K), mask)
check(f"model silhouette matches the mask after the corrections (IoU {iou_fixed:.2f} > 0.6) and not before ({iou_raw:.2f} < 0.4)",
      iou_fixed > 0.6 and iou_raw < 0.4)

# --- units ----------------------------------------------------------------------------------------
ranges = np.concatenate([np.linalg.norm(c["t"], axis=1) for c in cache.values()])
check(f"ground-truth translations in metres: ranges {ranges.min():.0f}-{ranges.max():.0f} m "
      f"(expected 100-500 m; the dataset reaches 518 m, bound 550 m)", ranges.min() >= 100 and ranges.max() <= 550)
check("camera intrinsics: fx = fy about 2059 px, principal point (960, 540)",
      abs(K[0, 0] - 2058.73) < 0.01 and K[0, 2] == 960 and K[1, 2] == 540)

# --- visibility ------------------------------------------------------------------------------------
strict = np.concatenate([c["vis_strict"] for c in cache.values()])
dil = np.concatenate([c["vis_dilated"] for c in cache.values()])
check(f"dilated visibility includes the strict one ({100 * strict.mean():.1f} % -> {100 * dil.mean():.1f} % of keypoints)",
      (dil | ~strict).all() and dil.mean() > strict.mean())

# --- noise-free PnP recovers the ground-truth pose ------------------------------------------------------
worst_t = worst_r = 0.0
n_used = n_fail = 0
for c in cache.values():
    for i in range(len(c["frames"])):
        if c["vis_dilated"][i].sum() < 6:
            continue
        z = meth.pnp_measurement(c["uv"][i], c["vis_dilated"][i], c["K"])
        if z is None:
            n_fail += 1
            continue
        dt, dr = meth.pose_errors(z, c["R"][i], c["t"][i])
        worst_t, worst_r = max(worst_t, dt), max(worst_r, dr)
        n_used += 1
check(f"0 px PnP recovers the pose on {n_used} frames with >= 6 visible keypoints: worst translation error "
      f"{1000 * worst_t:.4f} mm (< 1 mm), worst rotation error {worst_r:.6f} deg (< 0.01), {n_fail} failures",
      n_fail == 0 and worst_t < 1e-3 and worst_r < 0.01)

# --- calibration and Kalman filter -----------------------------------------------------------------------
cal = meth.calibrate_range_noise(2.0, K, n=1500)
covs = [np.array(b["cov"]) for b in cal["bins"]]
check("synthetic range-binned R is symmetric positive definite and its depth std grows with range",
      all(np.allclose(c, c.T) and np.linalg.eigvalsh(c).min() > 0 for c in covs)
      and all(covs[i][2, 2] < covs[i + 1][2, 2] for i in range(len(covs) - 1)))
check("filter sequences: scene 000125 is cut at frame 77, others are not",
      ev.sequences("000125", 100) == [(0, 77), (77, 100)] and ev.sequences("000009", 100) == [(0, 100)])
j = int(np.argmax(np.linalg.norm(np.diff(cache["000125"]["t"], axis=0), axis=1)))
check(f"the glitch of scene 000125 is between frames {j} and {j + 1} (cut at 77)", j + 1 == cfg.GLITCH_SPLITS["000125"][0])
for noise in (0.0, 2.0):
    calib = meth.calibrate_range_noise(noise, K, n=1500)["bins"]
    ok = True
    for s, c in cache.items():
        r = ev.run_scene(c, s, noise, 0, calib)
        ok &= bool(np.isfinite(r["kf_pos"]).all() and np.isfinite(r["kf_rot"]).all() and np.isfinite(r["kf_est"]).all())
    check(f"no NaN in the Kalman results of any frame ({noise:g} px, all 24 scenes)", ok)

# --- Phase 7a-fix: the full-state re-initialisation removes the gate lock-out of scene 000096 --------------
calib = meth.calibrate_range_noise(2.0, K, n=1500)["bins"]
rates = {}
for mode in ("attitude", "full"):
    r = ev.run_scene(cache["000096"], "000096", 2.0, 0, calib, reinit_mode=mode)
    rates[mode] = (100 * r["gated"] / r["offered"], np.nanmean(r["kf_pos"]))
check(f"scene 000096 at 2 px: attitude-only re-init locks out ({rates['attitude'][0]:.0f} % gated, {rates['attitude'][1]:.1f} m), "
      f"full re-init does not ({rates['full'][0]:.0f} % gated, {rates['full'][1]:.1f} m)",
      rates["attitude"][0] > 50 and rates["full"][0] < 20 and rates["full"][1] < rates["attitude"][1])

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
