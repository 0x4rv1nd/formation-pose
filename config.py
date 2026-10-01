"""
Configuration and conventions for formation-pose.

Reproduction of "Vision-Based Precision Pose Estimation for Autonomous
Formation Flying" (Punnoose, Stanford). A follower aircraft carries a
gimballed camera that always points at a leader aircraft; we estimate the
leader's relative pose from the pixel locations of 14 keypoints.

Conventions
-----------
Pose = 6 numbers [x, y, z, roll, pitch, yaw].

* x, y, z : position of the FOLLOWER relative to the LEADER in metres,
  expressed in the FOLLOWER's body axes (x forward, y right, z down).
  The leader's position in the follower frame is therefore -[x, y, z].
  Example: x = -60 means the follower is 60 m behind the leader,
  z = -20 means the follower is 20 m above the leader.
  The attitude below is used only to rotate the leader's keypoints,
  never the position.
* roll, pitch, yaw : attitude of the LEADER relative to the FOLLOWER in
  degrees, aerospace ZYX Euler order (yaw about z, then pitch about the new
  y, then roll about the new x). The resulting rotation matrix R maps
  vectors from the leader body frame to the follower body frame:
      v_follower = R @ v_leader
  All rotations use scipy.spatial.transform.Rotation.

Aircraft body frame: x forward, y right, z down, origin at the CG.

Camera frame: OpenCV convention (x right, y down, z forward along the
boresight). The camera sits at the follower's CG and always points at the
leader's CG, so the leader CG always projects to the principal point.
Image "down" is kept aligned with the follower's body z-axis (projected
perpendicular to the boresight). If the boresight is parallel to the
follower z-axis (leader directly above/below) the follower's -x axis is
used as the "down" reference instead.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 960
FOCAL_PX = 800.0
CX = IMAGE_WIDTH / 2.0   # principal point = image centre
CY = IMAGE_HEIGHT / 2.0
K = np.array([[FOCAL_PX, 0.0, CX],
              [0.0, FOCAL_PX, CY],
              [0.0, 0.0, 1.0]])

# ---------------------------------------------------------------------------
# Keypoints (metres, leader body frame) - approximate F-16 geometry
# ---------------------------------------------------------------------------
KEYPOINT_NAMES = [
    "nose", "tail_cone",
    "left_wingtip", "right_wingtip",
    "left_wing_root", "right_wing_root",
    "left_hstab_tip", "right_hstab_tip",
    "fin_top", "fin_root",
    "canopy", "intake",
    "left_store", "right_store",
]
KEYPOINTS = np.array([
    [7.5, 0.0, 0.0],     # nose
    [-7.5, 0.0, 0.0],    # tail_cone
    [-1.5, -4.9, 0.0],   # left_wingtip
    [-1.5, 4.9, 0.0],    # right_wingtip
    [2.5, -1.2, 0.0],    # left_wing_root
    [2.5, 1.2, 0.0],     # right_wing_root
    [-6.0, -2.8, 0.0],   # left_hstab_tip
    [-6.0, 2.8, 0.0],    # right_hstab_tip
    [-6.0, 0.0, -4.0],   # fin_top
    [-3.5, 0.0, -1.0],   # fin_root
    [4.0, 0.0, -1.3],    # canopy
    [3.0, 0.0, 1.2],     # intake
    [-0.5, -3.0, 0.5],   # left_store
    [-0.5, 3.0, 0.5],    # right_store
])
N_KEYPOINTS = len(KEYPOINT_NAMES)

# ---------------------------------------------------------------------------
# Occlusion shapes (leader body frame)
# ---------------------------------------------------------------------------
FUSELAGE_CENTER = np.zeros(3)
FUSELAGE_SEMI_AXES = np.array([7.5, 1.0, 1.0])

LEFT_WING = np.array([[2.5, -1.0, 0.0], [-3.5, -1.0, 0.0], [-1.5, -4.9, 0.0]])
RIGHT_WING = LEFT_WING * np.array([1.0, -1.0, 1.0])   # mirror in y
WING_TRIANGLES = np.stack([LEFT_WING, RIGHT_WING])     # (2, 3, 3)

# ---------------------------------------------------------------------------
# Pose label grid (paper Table I). Pitch and yaw are not part of the label.
# ---------------------------------------------------------------------------
GRID_X = np.linspace(-100.0, 50.0, 10)
GRID_Y = np.linspace(-50.0, 50.0, 10)
GRID_Z = np.linspace(-20.0, 20.0, 8)
GRID_ROLL = np.linspace(-45.0, 45.0, 6)
GRID = [GRID_X, GRID_Y, GRID_Z, GRID_ROLL]
GRID_SHAPE = tuple(len(g) for g in GRID)   # (10, 10, 8, 6)
N_LABELS = int(np.prod(GRID_SHAPE))        # 4800

# ---------------------------------------------------------------------------
# Simulation / dataset settings
# ---------------------------------------------------------------------------
PITCH_YAW_RANGE_DEG = 10.0   # pitch and yaw sampled uniformly in +/- this
MIN_RANGE_M = 25.0           # reject poses closer than this
KEYPOINT_NOISE_PX = 0.0      # Gaussian pixel noise std (used in later phases)
GIMBAL_NOISE_DEG = 0.0       # gimbal pointing error std (used in later phases)
N_FEATURES = 3 * N_KEYPOINTS + 3   # 42 keypoint features + 3 gimbal direction = 45
SEED = 0
DATA_DIR = "data"
RESULTS_DIR = "results"
