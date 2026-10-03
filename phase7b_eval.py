"""
Phase 7b, part B: library for the image-based evaluation (no scene is chosen here; see phase7b_tune_val.py and
phase7b_evaluate.py).

    ground truth    keypoints projected with the true pose (fw_uav_io cache), visibility = mask dilated by 2 px,
                    GT box = visible-mask bounding box + 4 px (the label box used for training)
    diagnostics     per-keypoint pixel error, left/right and leading/trailing swap rates, OKS (Ultralytics definition)
    pose methods    SQPnP / RANSAC PnP (each optionally symmetry-aware) on the detected keypoints, Kalman filter on top
"""

import cv2
import numpy as np

import fw_uav_config as cfg
import fw_uav_io as io
import fw_uav_methods as meth
import phase7b_prepare as prep

FLIP = prep.flip_idx_from_names(cfg.KEYPOINT_NAMES)
# (leading, trailing) pairs on the same wing: wingtips (5,6), (9,10) and wing roots (7,8), (11,12)
EDGE_PAIRS = [(5, 6), (7, 8), (9, 10), (11, 12)]
LR_PAIRS = [(i, FLIP[i]) for i in range(13) if FLIP[i] != i]
MIN_SEP_PX = 6.0           # a swap is only counted as such if the two ground-truth points are this far apart
OKS_SIGMA = 1.0 / 13       # Ultralytics default for a custom kpt_shape without OKS_SIGMA
RANSAC_ITERS = 300


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def gt_boxes(scene, frames):
    """Label boxes (x1, y1, x2, y2) = visible-mask bounding box grown by BOX_MARGIN_PX."""
    out = np.zeros((len(frames), 4))
    for i, f in enumerate(frames):
        ys, xs = np.nonzero(io.read_mask(scene, int(f)))
        out[i] = [xs.min() - prep.BOX_MARGIN_PX, ys.min() - prep.BOX_MARGIN_PX,
                  xs.max() + 1 + prep.BOX_MARGIN_PX, ys.max() + 1 + prep.BOX_MARGIN_PX]
    return out


def load_ground_truth(scenes):
    cache = io.build_keypoint_cache(cfg.KEYPOINTS)
    gt = {}
    for s in scenes:
        c = dict(cache[s])
        c["box"] = gt_boxes(s, c["frames"])
        c["range"] = np.linalg.norm(c["t"], axis=1)
        gt[s] = c
    return gt


def box_iou(a, b):
    lt, rb = np.maximum(a[..., :2], b[..., :2]), np.minimum(a[..., 2:], b[..., 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: np.clip(x[..., 2:] - x[..., :2], 0, None).prod(-1)
    return inter / (area(a) + area(b) - inter + 1e-9)


def select_detection(pred, box_thr):
    """Top-confidence detection of every frame: kpts (T,13,2), kconf (T,13), box (T,4), conf (T,), has (T,) bool."""
    conf = pred["box_conf"][:, 0]
    return dict(kpts=pred["kpts"][:, 0], kconf=pred["kpt_conf"][:, 0], box=pred["boxes"][:, 0], conf=conf, has=conf >= box_thr)


# ---------------------------------------------------------------------------
# Keypoint diagnostics
# ---------------------------------------------------------------------------
def oks(pred, gt_uv, vis, box):
    """Ultralytics kpt_iou for one object per frame: pred, gt_uv (T,13,2), vis (T,13) bool, box (T,4) -> (T,)."""
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    d2 = ((pred - gt_uv) ** 2).sum(-1)
    e = d2 / ((2 * OKS_SIGMA) ** 2 * (area[:, None] + 1e-7) * 2)
    return (np.exp(-e) * vis).sum(1) / np.maximum(vis.sum(1), 1)


def partner_swap(pred, gt_uv, vis, partner):
    """
    For every keypoint i with a partner j != i: swapped[t, i] = |pred_i - gt_j| < |pred_i - gt_i|, evaluated only where
    both keypoints are visible and their ground-truth points are MIN_SEP_PX apart (else NaN).
    """
    own = np.linalg.norm(pred - gt_uv, axis=-1)
    other = np.linalg.norm(pred - gt_uv[:, partner], axis=-1)
    sep = np.linalg.norm(gt_uv - gt_uv[:, partner], axis=-1)
    ok = vis & vis[:, partner] & (sep >= MIN_SEP_PX) & (np.array(partner) != np.arange(13))[None]
    return np.where(ok, (other < own).astype(float), np.nan)


def edge_partner():
    p = list(range(13))
    for a, b in EDGE_PAIRS:
        p[a], p[b] = b, a
    return p


def keypoint_diagnostics(det, g, bins_edges=cfg.RANGE_BINS_M):
    """All keypoint statistics of one split (pooled over its scenes). det / g: dicts scene -> selected detection / GT."""
    cat = lambda f: np.concatenate([f(s) for s in g])
    has = cat(lambda s: det[s]["has"])
    pred = cat(lambda s: det[s]["kpts"])
    kconf = cat(lambda s: det[s]["kconf"])
    uv = cat(lambda s: g[s]["uv"])
    vis = cat(lambda s: g[s]["vis_dilated"])
    box = cat(lambda s: g[s]["box"])
    rng = cat(lambda s: g[s]["range"])
    pbox = cat(lambda s: det[s]["box"])
    sel = has
    err = np.linalg.norm(pred - uv, axis=-1)
    valid = vis & sel[:, None]
    bin_idx = np.digitize(rng, np.array(bins_edges)[1:-1])
    out = {"n_frames": int(len(has)), "detection_rate": float(has.mean()),
           "box_iou_median": float(np.median(box_iou(pbox[has], box[has]))) if has.any() else None,
           "box_iou_below_0.5": float((box_iou(pbox[has], box[has]) < 0.5).mean()) if has.any() else None}

    def qs(v):
        return None if len(v) == 0 else {"n": int(len(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                                          "mean": float(v.mean())}

    out["error_px_all"] = qs(err[valid])
    out["error_px_by_keypoint"] = {n: qs(err[:, k][valid[:, k]]) for k, n in enumerate(cfg.KEYPOINT_NAMES)}
    out["error_px_by_bin"] = {}
    for b in range(len(bins_edges) - 1):
        m = (bin_idx == b)[:, None] & valid
        out["error_px_by_bin"][f"{bins_edges[b]:g}-{bins_edges[b + 1]:g} m"] = qs(err[m])
    # swaps
    lr = partner_swap(pred, uv, vis & sel[:, None], FLIP)
    ed = partner_swap(pred, uv, vis & sel[:, None], edge_partner())
    out["lr_swap_rate"] = float(np.nanmean(lr))
    out["lr_swap_rate_by_keypoint"] = {n: (None if np.isnan(lr[:, k]).all() else float(np.nanmean(lr[:, k])))
                                       for k, n in enumerate(cfg.KEYPOINT_NAMES) if FLIP[k] != k}
    out["lr_swap_rate_by_bin"] = {f"{bins_edges[b]:g}-{bins_edges[b + 1]:g} m":
                                  (None if np.isnan(lr[bin_idx == b]).all() else float(np.nanmean(lr[bin_idx == b])))
                                  for b in range(len(bins_edges) - 1)}
    out["edge_swap_rate"] = float(np.nanmean(ed))
    out["edge_swap_rate_by_keypoint"] = {n: (None if np.isnan(ed[:, k]).all() else float(np.nanmean(ed[:, k])))
                                         for k, n in enumerate(cfg.KEYPOINT_NAMES) if edge_partner()[k] != k}
    # frames with most keypoints swapped (whole-aircraft mirror)
    frame_lr = np.nanmean(np.where(np.isnan(lr), np.nan, lr), axis=1)
    out["frames_with_majority_lr_swap"] = float(np.nanmean(frame_lr[sel] > 0.5))
    # error vs keypoint confidence
    edges = [0, 0.9, 0.97, 0.985, 0.99, 0.995, 1.01]
    out["error_by_confidence"] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = valid & (kconf >= lo) & (kconf < hi)
        out["error_by_confidence"][f"{lo:g}-{min(hi, 1):g}"] = qs(err[m])
    # OKS (mean over visible keypoints of exp(-e)) of the detections, and with oracle corrections
    pk = np.where(sel[:, None, None], pred, 0.0)
    o_plain = oks(pk, uv, vis, box)
    o_mirror = oks(pk[:, FLIP], uv, vis, box)                       # whole aircraft mirrored
    best_mirror = np.maximum(o_plain, o_mirror)

    def best_pair_oks(partner):
        # per pair, keep whichever labelling gives the smaller error (oracle): picks the better of own / partner
        pr = pk.copy()
        for i, j in enumerate(partner):
            if j > i:
                own = np.linalg.norm(pk[:, i] - uv[:, i], axis=-1) + np.linalg.norm(pk[:, j] - uv[:, j], axis=-1)
                swp = np.linalg.norm(pk[:, i] - uv[:, j], axis=-1) + np.linalg.norm(pk[:, j] - uv[:, i], axis=-1)
                use = swp < own
                pr[use, i], pr[use, j] = pk[use, j], pk[use, i]
        return oks(pr, uv, vis, box)
    o_lr, o_edge = best_pair_oks(FLIP), best_pair_oks(edge_partner())
    out["oks"] = {}
    for name, o in (("as_detected", o_plain), ("oracle_mirror_whole_aircraft", best_mirror), ("oracle_pairwise_lr_swap_fix", o_lr),
                    ("oracle_pairwise_leading_trailing_fix", o_edge)):
        o = o[sel]
        out["oks"][name] = {"mean": float(o.mean()), "frac_gt_0.5": float((o > 0.5).mean()), "frac_gt_0.75": float((o > 0.75).mean()),
                            "frac_gt_0.9": float((o > 0.9).mean())}
    # how strict is OKS here: pixel error that gives OKS 0.5 on a typical box, and the box sizes
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    sc = np.sqrt(area[sel])
    out["gt_box_width_px_median"] = float(np.median((box[:, 2] - box[:, 0])[sel]))
    out["gt_box_height_px_median"] = float(np.median((box[:, 3] - box[:, 1])[sel]))
    out["px_error_for_oks_0.5_median_box"] = float(np.sqrt(-np.log(0.5) * (2 * OKS_SIGMA) ** 2 * 2 * np.median(area[sel])))
    out["px_error_for_oks_0.75_median_box"] = float(np.sqrt(-np.log(0.75) * (2 * OKS_SIGMA) ** 2 * 2 * np.median(area[sel])))
    return out


# ---------------------------------------------------------------------------
# Pose methods on detected keypoints
# ---------------------------------------------------------------------------
def _pose_from(ok, rvec, tvec):
    t = np.asarray(tvec).ravel()
    if not ok or not np.all(np.isfinite(t)) or t[2] <= 0 or not meth.MIN_RANGE_M <= np.linalg.norm(t) <= meth.MAX_RANGE_M:
        return None
    return np.concatenate([t, meth.rotation_to_pose_angles(cv2.Rodrigues(rvec)[0])])


def _reproj(pose, idx, img, K):
    R = meth.pose_to_rotation(pose)
    proj, _ = io.project_points(cfg.KEYPOINTS[idx], R, pose[:3], K)
    return np.linalg.norm(proj - img, axis=1)


def solve_frame(uv, conf, K, kp_thr, method, symmetric, ransac_px=8.0):
    """
    method 'sqpnp' or 'ransac'; symmetric: also try the labelling with every left/right pair swapped and keep the
    candidate with the smaller reprojection cost (mean of min(err, cap)^2, cap = ransac_px for RANSAC, none for SQPnP).
    Returns the pose vector (6,) or None.
    """
    perms = [np.arange(13)] + ([np.array(FLIP)] if symmetric else [])
    best, best_cost = None, np.inf
    for perm in perms:
        p_uv, p_conf = uv[perm], conf[perm]
        idx = np.flatnonzero(p_conf >= kp_thr)
        if len(idx) < max(meth.MIN_POINTS, 5 if method == "ransac" else 0):
            continue
        obj = np.ascontiguousarray(cfg.KEYPOINTS[idx], dtype=np.float64)
        img = np.ascontiguousarray(p_uv[idx], dtype=np.float64)
        try:
            if method == "ransac":
                cv2.setRNGSeed(0)
                ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP, reprojectionError=ransac_px,
                                                         iterationsCount=RANSAC_ITERS, confidence=0.99)
                if inl is None or len(inl) < meth.MIN_POINTS:
                    continue
            else:
                ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            continue
        pose = _pose_from(ok, rvec, tvec)
        if pose is None:
            continue
        e = _reproj(pose, idx, img, K)
        cost = np.mean(np.minimum(e, ransac_px) ** 2) if method == "ransac" else np.mean(e ** 2)
        if cost < best_cost:
            best, best_cost = pose, cost
    return best


def pose_sequence(det, K, kp_thr, method, symmetric, ransac_px=8.0):
    """Pose vectors (T, 6), NaN where there is no detection or no solution."""
    T = len(det["has"])
    out = np.full((T, 6), np.nan)
    for i in range(T):
        if det["has"][i]:
            p = solve_frame(det["kpts"][i], det["kconf"][i], K, kp_thr, method, symmetric, ransac_px)
            if p is not None:
                out[i] = p
    return out


def kalman_sequence(z, range_bins, dt=cfg.DT):
    """Run UAVKalman (full re-initialisation) over a sequence of PnP measurements (T,6) with NaN = missing."""
    kf = meth.UAVKalman(range_bins, dt, "full")
    est = np.full((len(z), 6), np.nan)
    for i in range(len(z)):
        m = None if not np.isfinite(z[i]).all() else z[i]
        est[i] = kf.initialise(m) if i == 0 else kf.step(m)
    offered = len(z) - 1 - kf.n_missing
    return est, dict(gated=kf.n_gated, offered=max(offered, 0), reinit=kf.n_reinit)


def errors(est, g):
    """Translation (m) and rotation (deg) error per frame (NaN where there is no estimate)."""
    T = len(est)
    pos, rot = np.full(T, np.nan), np.full(T, np.nan)
    for i in range(T):
        if np.isfinite(est[i]).all():
            pos[i], rot[i] = meth.pose_errors(est[i], g["R"][i], g["t"][i])
    return pos, rot


def failure(pos, rot, rng):
    return (~np.isfinite(pos)) | (pos > meth.GROSS_REL_POS * rng) | (rot > meth.GROSS_ROT_DEG)


def summarise(per_scene, key):
    """per_scene: scene -> dict(range, <key>_pos, <key>_rot). Mean / median errors (frames with a solution), failure rate, by bin."""
    cat = lambda k: np.concatenate([r[k] for r in per_scene.values()])
    rng, pos, rot = cat("range"), cat(f"{key}_pos"), cat(f"{key}_rot")
    fail = failure(pos, rot, rng)
    b = np.digitize(rng, np.array(cfg.RANGE_BINS_M)[1:-1])
    labels = ["150-250 m", "250-350 m", "350-450 m", "450+ m"]
    out = {}
    for tag, m in [("all", np.ones(len(rng), bool))] + [(labels[k], b == k) for k in range(4)]:
        if not m.any():
            continue
        p, r = pos[m], rot[m]
        ok = np.isfinite(p)
        out[tag] = {"n_frames": int(m.sum()), "no_solution_pct": float(100 * (~ok).mean()), "fail_pct": float(100 * fail[m].mean()),
                    "pos_mean": float(p[ok].mean()) if ok.any() else None, "pos_median": float(np.median(p[ok])) if ok.any() else None,
                    "rot_mean": float(r[ok].mean()) if ok.any() else None, "rot_median": float(np.median(r[ok])) if ok.any() else None,
                    "pos_rel_mean_pct": float(np.mean(100 * p[ok] / rng[m][ok])) if ok.any() else None}
    return out


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------
def nearest_train_rotation_deg(R, train_scenes):
    """Per frame: geodesic distance (deg) from the true object rotation R (T,3,3) to the closest training-frame rotation."""
    from scipy.spatial.transform import Rotation
    cache = io.build_keypoint_cache(cfg.KEYPOINTS)
    tr = Rotation.from_matrix(np.concatenate([cache[s]["R"] for s in train_scenes]))
    return np.degrees(np.array([(tr.inv() * r).magnitude().min() for r in Rotation.from_matrix(R)]))


def synthetic_pnp(c, noise_px, seed):
    """Phase 7a protocol, PnP only: projected keypoints + Gaussian noise, SQPnP on the (dilated-mask) visible ones."""
    T = len(c["frames"])
    rng = np.random.default_rng(seed)
    pos, rot = np.full(T, np.nan), np.full(T, np.nan)
    for i in range(T):
        uv = c["uv"][i] + (rng.normal(0.0, noise_px, c["uv"][i].shape) if noise_px > 0 else 0.0)
        z = meth.pnp_measurement(uv, c["vis_dilated"][i], c["K"])
        if z is not None:
            pos[i], rot[i] = meth.pose_errors(z, c["R"][i], c["t"][i])
    return pos, rot
