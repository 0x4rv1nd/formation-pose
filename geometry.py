"""
Geometry: the ONLY place that converts poses to pixels.

Frames (see config.py for the full conventions):
    leader body  --R-->  follower body  --R_c-->  camera (OpenCV)
"""

import numpy as np
from scipy.spatial.transform import Rotation

import config

# Distance tolerance (metres) so points lying on a surface don't hide themselves
OCCLUSION_EPS_M = 1e-3


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------
def euler_to_R(roll, pitch, yaw):
    """Aerospace ZYX Euler angles (deg) -> rotation matrix (leader body -> follower body)."""
    return Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=True).as_matrix()


def R_to_euler(R):
    """Rotation matrix (leader body -> follower body) -> (roll, pitch, yaw) in degrees."""
    yaw, pitch, roll = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)
    return roll, pitch, yaw


def camera_rotation(boresight):
    """
    Rotation from the follower body frame to the camera frame.

    boresight: vector (follower body frame) from the camera to the target.
    The camera z-axis is the boresight, the camera y-axis ("image down") is
    the follower z-axis projected perpendicular to the boresight, and
    x = y cross z completes a right-handed frame (x right).
    """
    z_c = np.asarray(boresight, dtype=float)
    z_c = z_c / np.linalg.norm(z_c)

    down_ref = np.array([0.0, 0.0, 1.0])  # follower body z (down)
    y_c = down_ref - np.dot(down_ref, z_c) * z_c
    if np.linalg.norm(y_c) < 1e-9:
        # Degenerate: looking straight up/down. Use follower "backward" as image down.
        down_ref = np.array([-1.0, 0.0, 0.0])
        y_c = down_ref - np.dot(down_ref, z_c) * z_c
    y_c = y_c / np.linalg.norm(y_c)
    x_c = np.cross(y_c, z_c)

    # Rows are the camera axes expressed in the follower frame
    return np.vstack([x_c, y_c, z_c])


def boresight_unit(pose, noise_deg=0.0, rng=None):
    """
    Unit vector from the follower to the leader, in follower body axes:
    -[x, y, z] / norm. This is where the gimbal is pointing, which a real
    gimbal knows. A 3D unit vector is used instead of azimuth/elevation
    because azimuth wraps at +/-180 deg when the leader is behind (x > 0).

    If noise_deg > 0 the vector is rotated by a small random rotation
    (random axis, Gaussian angle with std noise_deg) to mimic gimbal error.
    """
    b = -np.asarray(pose[:3], dtype=float)
    b = b / np.linalg.norm(b)
    if noise_deg > 0:
        if rng is None:
            rng = np.random.default_rng()
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(rng.normal(0.0, noise_deg))
        b = Rotation.from_rotvec(angle * axis).apply(b)
    return b


def pose_to_camera(pose, boresight=None):
    """
    Pose -> (R_cam, t_cam) with X_cam = R_cam @ P_leader + t_cam.

    This is exactly the (rvec, tvec) pair cv2.solvePnP estimates from
    leader-frame keypoints (rvec = Rodrigues(R_cam)).

    boresight: where the gimbal really points (follower axes). Default: the
    pose's own direction to its own leader, which is what the real camera
    does for the TRUE pose. Pass the measured boresight to convert a wrong
    (e.g. coarse classifier) pose consistently with how the image was formed.
    """
    x, y, z, roll, pitch, yaw = pose
    p = np.array([x, y, z])        # follower position, follower axes
    R = euler_to_R(roll, pitch, yaw)

    # Leader CG in the follower frame is simply -p (attitude never moves it).
    # A leader point P is at R @ P - p in the follower frame.
    R_c = camera_rotation(-p if boresight is None else boresight)

    R_cam = R_c @ R
    t_cam = -R_c @ p
    return R_cam, t_cam


def camera_to_pose(R_cam, t_cam, boresight):
    """
    Inverse of pose_to_camera: camera-frame (R_cam, t_cam), e.g. from
    cv2.solvePnP, -> pose (6,). The follower->camera rotation is built from
    the MEASURED boresight (the gimbal direction), not from the estimate.
    R = R_c^T @ R_cam (leader -> follower), p = -R_c^T @ t_cam.
    """
    R_c = camera_rotation(boresight)
    R = R_c.T @ R_cam
    p = -R_c.T @ np.asarray(t_cam, dtype=float).ravel()
    return np.array([*p, *R_to_euler(R)])


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def project_points(pose, points=None):
    """Pinhole projection of the leader keypoints -> (N, 2) pixel coordinates."""
    if points is None:
        points = config.KEYPOINTS
    R_cam, t_cam = pose_to_camera(pose)
    X_cam = points @ R_cam.T + t_cam            # (N, 3)
    uvw = X_cam @ config.K.T                    # homogeneous pixels
    return uvw[:, :2] / uvw[:, 2:3]


def project_points_batch(poses, boresight, points=None):
    """
    Project the leader keypoints for N poses at once -> (N, 14, 2) pixels.

    Unlike project_points, the camera orientation comes from ONE given
    boresight (where the gimbal really points), not from each pose's own
    direction to its own leader. Particles that are wrong in position
    therefore do not all put the leader at the image centre.
    With boresight = the pose's own unit boresight this equals project_points.
    """
    if points is None:
        points = config.KEYPOINTS
    poses = np.atleast_2d(np.asarray(poses, dtype=float))
    p = poses[:, :3]                                            # (N, 3)
    R = Rotation.from_euler(
        "ZYX", poses[:, [5, 4, 3]], degrees=True).as_matrix()   # (N, 3, 3)
    R_c = camera_rotation(boresight)                            # same for all particles

    # Keypoints in the follower frame: R @ P - p, then into the camera frame
    pts_f = np.einsum("nij,kj->nki", R, points) - p[:, None, :]  # (N, K, 3)
    X_cam = pts_f @ R_c.T
    # Points behind the camera get a tiny positive depth: a huge pixel error
    # instead of a division by zero or a mirrored image.
    z = np.maximum(X_cam[..., 2], 1e-6)
    u = config.FOCAL_PX * X_cam[..., 0] / z + config.CX
    v = config.FOCAL_PX * X_cam[..., 1] / z + config.CY
    return np.stack([u, v], axis=-1)


# ---------------------------------------------------------------------------
# Occlusion
# ---------------------------------------------------------------------------
def _ray_hits_ellipsoid(origin, dirs, center, semi_axes, s_max):
    """
    Rays origin + s * dirs[i]. True where the ray enters the ellipsoid at
    0 < s < s_max[i]. Solved by scaling space so the ellipsoid is a unit sphere.
    """
    o = (origin - center) / semi_axes           # (3,)
    d = dirs / semi_axes                        # (N, 3)
    a = np.sum(d * d, axis=1)
    b = 2.0 * d @ o
    c = np.dot(o, o) - 1.0
    disc = b * b - 4 * a * c
    hit = disc > 0
    sqrt_disc = np.sqrt(np.where(hit, disc, 0.0))
    s_entry = (-b - sqrt_disc) / (2 * a)        # first intersection
    return hit & (s_entry > 0) & (s_entry < s_max)


def _ray_hits_triangle(origin, dirs, tri, s_max):
    """Vectorised Moller-Trumbore: True where ray i hits triangle at 0 < s < s_max[i]."""
    v0, v1, v2 = tri
    e1, e2 = v1 - v0, v2 - v0
    pvec = np.cross(dirs, e2)                   # (N, 3)
    det = pvec @ e1
    ok = np.abs(det) > 1e-12                    # ray not parallel to the plane
    inv_det = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)

    tvec = origin - v0
    u = (pvec @ tvec) * inv_det
    qvec = np.cross(tvec, e1)                   # (3,)
    v = (dirs @ qvec) * inv_det
    s = (qvec @ e2) * inv_det
    return ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (s > 0) & (s < s_max)


def visible_mask(pose, uv):
    """
    (14,) bool: keypoint is visible if it is in front of the camera, inside
    the image, and the ray from the camera to it is not blocked by the
    fuselage ellipsoid or a wing triangle.

    Ray tests are done in the leader frame. The follower (camera) origin is
    R^T @ p there, since a leader point P sits at R @ P - p in the follower
    frame. Ray: cam + s * (P - cam), s in (0, 1).
    """
    pts = config.KEYPOINTS
    R = euler_to_R(*pose[3:])
    cam = R.T @ np.asarray(pose[:3], dtype=float)
    dirs = pts - cam
    # Stop the test slightly before the point itself (epsilon in metres)
    s_max = 1.0 - OCCLUSION_EPS_M / np.linalg.norm(dirs, axis=1)

    occluded = _ray_hits_ellipsoid(cam, dirs, config.FUSELAGE_CENTER,
                                   config.FUSELAGE_SEMI_AXES, s_max)
    for tri in config.WING_TRIANGLES:
        occluded |= _ray_hits_triangle(cam, dirs, tri, s_max)

    # In front of the camera and inside the image
    R_cam, t_cam = pose_to_camera(pose)
    depth = (pts @ R_cam.T + t_cam)[:, 2]
    in_image = ((uv[:, 0] >= 0) & (uv[:, 0] < config.IMAGE_WIDTH) &
                (uv[:, 1] >= 0) & (uv[:, 1] < config.IMAGE_HEIGHT))

    return ~occluded & (depth > 0) & in_image


def observe(pose, noise_px=0.0, rng=None):
    """Simulated perfect keypoint detector (+ optional Gaussian pixel noise) -> (uv, visible)."""
    uv = project_points(pose)
    visible = visible_mask(pose, uv)
    if noise_px > 0:
        if rng is None:
            rng = np.random.default_rng()
        uv = uv + rng.normal(0.0, noise_px, size=uv.shape)
    return uv, visible
