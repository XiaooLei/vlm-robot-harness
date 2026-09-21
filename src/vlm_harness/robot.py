"""Robot control over zerorpc to the existing DROID backend on the NUC."""
import time

import numpy as np

from .actions import finite_vector
from .safety import (MEASURED_SLACK, check_status_unchanged, pose_error, require_stationary,
                     validate_target)


class DroidRobot:
    def __init__(self, host, timeout=8):
        import zerorpc
        self.rpc = zerorpc.Client(timeout=timeout)
        self.rpc.connect(f"tcp://{host}:4242")

    def observe(self):
        state, stamp = self.rpc.get_robot_state()
        status = self.rpc.get_connection_status()
        require_stationary(state, status)
        return state, status, stamp

    def execute(self, action, observed, observed_status, workspace, allow_gripper, timeout, deadline,
                max_joint_step=0., speed_factor=.01, arrival_tolerance=.01, hold_orientation=None):
        fresh, status, _ = self.observe()
        check_status_unchanged(observed_status, status)
        pe, re = pose_error(fresh["cartesian_position"], observed["cartesian_position"])
        if pe > .003 or re > .05 or abs(fresh["gripper_position"] - observed["gripper_position"]) > .03:
            raise RuntimeError("Robot moved during inference; stale action rejected")
        target, grip = validate_target(action, fresh, workspace, allow_gripper)
        if hold_orientation is not None:
            # The backend tracks a commanded orientation only loosely, and because
            # each target copies the measured pose the error accumulates over a
            # long run. Re-aim at the session's reference orientation instead.
            target = np.asarray(target, dtype=float)
            target[3:] = np.asarray(hold_orientation, dtype=float)
        grip_status = self.rpc.get_gripper_async_status()
        if grip_status["gripper_async_busy"] or grip_status["gripper_async_has_pending"]:
            raise RuntimeError("Another gripper operation is pending")
        if action.grip_delta and not grip_status["gripper_available"]:
            raise RuntimeError("Gripper unavailable")
        if time.monotonic() >= deadline:
            raise RuntimeError("Session deadline reached before dispatch")
        tracking_delta = np.asarray(action.delta, dtype=float).copy()
        joint_step_scale = 1.0
        if np.any(action.delta):
            # Existing EEF -> IK API; only send seven arm joints, never a gripper side-effect.
            computed = self.rpc.create_action_dict(target.tolist() + [fresh["gripper_position"]],
                                                   "cartesian_position", "position", fresh)
            joints = finite_vector(computed["joint_position"], 7)
            joint_start = finite_vector(fresh["joint_positions"], 7)
            joint_delta = joints - joint_start
            largest = float(np.max(np.abs(joint_delta)))
            if max_joint_step > 0 and largest > max_joint_step:
                joint_step_scale = max_joint_step / largest
                joints = joint_start + joint_delta * joint_step_scale
                tracking_delta *= joint_step_scale
                target = np.asarray(target, dtype=float).copy()
                target[:3] = np.asarray(fresh["cartesian_position"][:3]) + tracking_delta
            if time.monotonic() >= deadline:
                raise RuntimeError("Session deadline reached before dispatch")
            move = (self.rpc.move_to_joint_positions_checked if status.get("checked_motion_supported")
                    else self.rpc.move_to_joint_positions)
            if not move(joints.tolist(), speed_factor):
                raise RuntimeError("Arm submission failed; client does not retry")
        elif action.grip_delta:
            if action.grip_delta < 0:
                if not grip_status.get("explicit_release_supported", False):
                    raise RuntimeError("Backend requires explicit release support; refusing legacy auto-grip path")
                submitted = self.rpc.release_gripper(grip)
            else:
                submitted = self.rpc.update_gripper(grip, False, False)
            if not submitted:
                # False also means "already at that target" (deadband), not only a
                # fault; the next observation shows whether the jaws actually moved.
                return dict(state=fresh, controller_status=status, result="gripper_no_change")
        else:
            return dict(state=fresh, controller_status=status, result="no_op")
        # Do not send the next action until actual pose/gripper has settled. The
        # backend's EEF->IK path does not reproduce a requested Cartesian delta
        # exactly (measured: 60-85% of the requested magnitude), and every step
        # re-plans from a fresh observation, so a step also counts as done when
        # the arm has clearly moved the commanded way and stopped.
        end = min(time.monotonic() + timeout, deadline)
        stable = None
        while time.monotonic() < end:
            state, _ = self.rpc.get_robot_state()
            current_status = self.rpc.get_connection_status()
            check_status_unchanged(status, current_status)
            if not current_status["robot_connected"]:
                raise RuntimeError("Robot disconnected during motion")
            for key, n in (("cartesian_position", 6), ("joint_positions", 7), ("joint_velocities", 7),
                           ("franka_joint_collision", 7), ("franka_cartesian_collision", 6)):
                values = finite_vector(state[key], n)
                if "collision" in key and np.any(values):
                    raise RuntimeError("Collision flag during motion; no further commands")
            finite_vector([state["gripper_position"]], 1)
            if workspace is not None:
                actual = np.asarray(state["cartesian_position"][:3])
                if np.any(actual < workspace[0] - MEASURED_SLACK) or np.any(actual > workspace[1] + MEASURED_SLACK):
                    raise RuntimeError("Measured EEF outside workspace; no further commands")
            gs = self.rpc.get_gripper_async_status()
            expected_submits = grip_status["gripper_async_submit_count"] + bool(action.grip_delta)
            if gs["gripper_async_error_count"] != grip_status["gripper_async_error_count"] or gs["gripper_async_submit_count"] != expected_submits:
                raise RuntimeError("Gripper error/competing command; no further commands")
            pe, re = pose_error(state["cartesian_position"], target)
            travelled = np.asarray(state["cartesian_position"][:3], dtype=float) - np.asarray(fresh["cartesian_position"][:3], dtype=float)
            progress = (float(travelled @ tracking_delta) / float(tracking_delta @ tracking_delta)) if np.any(action.delta) else 1.
            stationary = (max(abs(x) for x in state["joint_velocities"]) < .005
                          and not current_status["joint_async_busy"]
                          and not current_status["joint_async_has_pending"])
            distance = float(np.linalg.norm(tracking_delta))
            # Absolute tolerance must not swallow a small request before it moves.
            tolerance = min(arrival_tolerance, max(.0015, .15 * distance)) if distance else .003
            lateral_error = float(np.linalg.norm(travelled - progress * tracking_delta)) if distance else 0.
            partial = distance > 0 and .25 <= progress <= 1.25 and lateral_error <= max(.0015, .15 * distance)
            moved_enough = not distance or progress >= .25
            reached = stationary and re <= .02 and moved_enough and (pe <= tolerance or partial)
            # Small, bounded IK deviations warrant fresh vision, never blind continuation.
            reobserve = (not reached and stationary and re <= .02 and
                         0 < distance <= .02 and .5 <= progress <= 1.25 and
                         lateral_error <= max(.003, .25 * distance) and pe <= max(.006, .5 * distance))
            reached = reached or reobserve
            if action.grip_delta:
                # An object between the jaws stops the gripper short of the
                # commanded position, so also accept a stalled gripper that has
                # moved a good part of the way and is no longer busy.
                moved = abs(state["gripper_position"] - fresh["gripper_position"])
                reached = reached and not gs["gripper_async_busy"] and not gs["gripper_async_has_pending"] and (
                    abs(state["gripper_position"] - grip) <= .015 or
                    (action.grip_delta > 0 and state["gripper_position"] > fresh["gripper_position"]
                     and moved >= .5 * action.grip_delta))
            elif abs(state["gripper_position"] - fresh["gripper_position"]) > .015:
                raise RuntimeError("Unexpected gripper movement")
            if reached:
                stable = time.monotonic() if stable is None else stable
                if time.monotonic() - stable >= .15:
                    return dict(state=state, controller_status=self.rpc.get_connection_status(), result=("reobserve" if reobserve else "settled" if pe <= tolerance else "partial_step"),
                                position_error_m=pe, rotation_error_rad=re, joint_step_scale=joint_step_scale,
                                observed_delta_m=(travelled.tolist() if np.any(action.delta) else None),
                                fraction_of_requested=round(progress, 3))
            else:
                stable = None
            time.sleep(.1)
        raise RuntimeError(f"Arrival timeout: no further commands; position_error={pe:.6f}m, "
                           f"progress={progress:.3f}, lateral_error={lateral_error:.6f}m, "
                           f"rotation_error={re:.6f}rad, stationary={stationary}")

    def close(self):
        self.rpc.close()
