"""Pose sampling and trajectories for the formation-flying simulation."""

import numpy as np

import config


def sample_random_poses(n, rng):
    """
    n random poses: x, y, z, roll uniform within the label-grid limits,
    pitch and yaw uniform in +/- PITCH_YAW_RANGE_DEG. Poses closer than
    MIN_RANGE_M are rejected and redrawn.
    """
    lows = [g[0] for g in config.GRID] + [-config.PITCH_YAW_RANGE_DEG] * 2
    highs = [g[-1] for g in config.GRID] + [config.PITCH_YAW_RANGE_DEG] * 2

    poses = np.empty((0, 6))
    while len(poses) < n:
        batch = rng.uniform(lows, highs, size=(n, 6))
        keep = np.linalg.norm(batch[:, :3], axis=1) >= config.MIN_RANGE_M
        poses = np.vstack([poses, batch[keep]])
    return poses[:n]


def approach_trajectory(duration=10.0, dt=0.1):
    """
    Follower approach similar to the paper's Fig. 6: closes from 90 m to 40 m
    behind the leader while the leader gently rolls, pitches and yaws.
    Returns (times (T,), poses (T, 6)).
    """
    t = np.arange(0.0, duration + dt / 2, dt)
    frac = t / duration
    x = -90.0 + 50.0 * frac
    y = np.full_like(t, 20.0)
    z = 15.0 - 5.0 * frac
    roll = 15.0 * np.sin(2 * np.pi * t / 5.0)
    pitch = 2.0 * np.sin(2 * np.pi * t / 7.0)
    yaw = 3.0 * np.sin(2 * np.pi * t / 6.0)
    return t, np.column_stack([x, y, z, roll, pitch, yaw])
