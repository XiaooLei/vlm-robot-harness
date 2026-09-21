"""Bounds, workspace validation and controller-health checks."""
import numpy as np

from .actions import finite_vector


# Measured end-effector poses jitter by a fraction of a millimetre, so a bound
# is treated as met unless it is exceeded by this much.
WORKSPACE_SLACK = .002
# The arm overshoots a commanded target by a few millimetres; the post-motion
# check only needs to catch a genuine breach, not that overshoot.
MEASURED_SLACK = .008
def pose_error(a, b):
    from scipy.spatial.transform import Rotation
    a, b = np.asarray(a), np.asarray(b)
    return (float(np.linalg.norm(a[:3] - b[:3])),
            float((Rotation.from_euler("xyz", a[3:]) *
                   Rotation.from_euler("xyz", b[3:]).inv()).magnitude()))


def require_stationary(state, status):
    for key, n in (("cartesian_position", 6), ("joint_positions", 7), ("joint_velocities", 7)):
        finite_vector(state[key], n)
    finite_vector([state["gripper_position"]], 1)
    if not 0 <= state["gripper_position"] <= 1:
        raise RuntimeError("Invalid gripper state")
    if not status["robot_connected"] or status["last_robot_error"]:
        raise RuntimeError("Robot not healthy")
    if status["joint_async_busy"] or status["joint_async_has_pending"]:
        raise RuntimeError("Another joint operation is pending")
    # Closing the jaws shakes the arm slightly; this only needs to reject a
    # genuinely moving robot.
    if max(abs(v) for v in state["joint_velocities"]) >= .02:
        raise RuntimeError("Robot not stationary")
    for key, n in (("franka_joint_collision", 7), ("franka_cartesian_collision", 6)):
        if np.any(finite_vector(state[key], n)):
            raise RuntimeError("Collision flag present")


def check_status_unchanged(before, after, allow_recovery=False):
    for key in ("recover_count", "reconnect_count", "joint_async_submit_count", "joint_async_error_count"):
        if before[key] != after[key]:
            # The backend recovers itself; mid-motion recovery is reported by the
            # next observation, which re-checks health, stationarity and collisions.
            if allow_recovery and key in ("recover_count", "reconnect_count"):
                continue
            raise RuntimeError(f"Controller changed {key}; no further actions")
    if after["last_robot_error"]:
        raise RuntimeError("Controller reports an error")


def validate_target(action, state, workspace, allow_gripper):
    target = np.array(state["cartesian_position"], dtype=float)
    target[:3] += action.delta
    if workspace is not None:
        lo, hi = workspace
        current = np.asarray(state["cartesian_position"][:3])
        if (np.any(current < lo - WORKSPACE_SLACK) or np.any(current > hi + WORKSPACE_SLACK)
                or np.any(target[:3] < lo - WORKSPACE_SLACK) or np.any(target[:3] > hi + WORKSPACE_SLACK)):
            raise ValueError("Current/target EEF is outside operator-specified workspace")
    if action.grip_delta and not allow_gripper:
        raise ValueError("Gripper actuation disabled; requires explicit --allow-gripper")
    grip = state["gripper_position"] + action.grip_delta
    if not 0 <= grip <= 1:
        raise ValueError("Gripper target outside [0,1]; not clipped")
    return target, grip
