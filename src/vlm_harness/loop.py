"""Session loop and CLI: observe, ask the model, validate, dispatch, verify."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np

from .actions import Action, finite_vector, normalize_action, parse_action
from .cameras import ThreeCameras
from .history import attach_after_observation, compact_history, load_history
from .prompts import SYSTEM, VIEW_LABELS, wrist_view_axes  # noqa: F401
from .robot import DroidRobot
from .safety import (MEASURED_SLACK, WORKSPACE_SLACK, check_status_unchanged,  # noqa: F401
                     pose_error, require_stationary, validate_target)
from .vision import VisionClient, thinking_body


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    # No endpoint or model is baked in: give one explicitly, or via the
    # environment, so a checkout carries no deployment detail.
    p.add_argument("--base-url", default=os.getenv("VLM_BASE_URL"),
                   help="OpenAI-compatible base URL, e.g. https://host/v1 (or $VLM_BASE_URL)")
    p.add_argument("--model", default=os.getenv("VLM_MODEL"),
                   help="Model name to send (or $VLM_MODEL)")
    p.add_argument("--allow-http", action="store_true",
                   help="Permit a plaintext http endpoint (camera images and the API key travel unencrypted)")
    p.add_argument("--thinking", choices=("auto", "dashscope", "deepseek", "none"), default="auto",
                   help="Provider switch that suppresses the reasoning budget; auto picks by endpoint host")
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--vision-confirmed", action="store_true", help="Operator confirms chosen endpoint/model supports image_url inputs")
    p.add_argument("--nuc", default=os.getenv("VLM_NUC"),
                   help="Host running the robot RPC server on port 4242 (or $VLM_NUC)")
    p.add_argument("--cameras", nargs=3, default=(os.getenv("VLM_CAMERAS") or "").split() or None,
                   metavar=("WRIST", "EXTERIOR1", "EXTERIOR2"),
                   help="Three distinct ZED serials, wrist first (or $VLM_CAMERAS)")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--allow-gripper", action="store_true", help="Opt in to existing backend's gripper behavior; no force cap in this RPC")
    p.add_argument("--workspace-min", nargs=3, type=float)
    p.add_argument("--workspace-max", nargs=3, type=float)
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--max-step-m", type=float, default=.1, help="Max EEF translation norm per step, metres")
    p.add_argument("--min-step-m", type=float, default=0., help="Min EEF translation norm for a moving step; 0 disables")
    p.add_argument("--sub-step-m", type=float, default=.02,
                   help="Largest single dispatch inside one action; the backend's IK caps a dispatch near this")
    p.add_argument("--max-sub-steps", type=int, default=1, help="Dispatches allowed to serve one model action")
    p.add_argument("--max-gripper-step", type=float, default=.5,
                   help="Max normalized gripper change per step; 0.5 closes half the stroke in one command")
    p.add_argument("--max-joint-step-rad", type=float, default=0., help="Max per-joint IK increment; 0 disables the check")
    p.add_argument("--speed-factor", type=float, default=.01, help="Franky relative dynamics factor for arm dispatch")
    p.add_argument("--arrival-tolerance-m", type=float, default=.01, help="Measured position error accepted as arrived")
    p.add_argument("--max-seconds", type=float, default=300)
    p.add_argument("--max-travel-m", type=float, default=.3)
    p.add_argument("--max-gripper-travel", type=float, default=3.)
    p.add_argument("--max-observation-age", type=float, default=30)
    p.add_argument("--api-timeout", type=float, default=30)
    p.add_argument("--arrival-timeout", type=float, default=30)
    p.add_argument("--log-dir", type=Path, default=Path("runs/agent_harness"))
    p.add_argument("--record-video", action="store_true", help="Record three RGB MP4s using shared camera handles")
    p.add_argument("--resume-log", type=Path, help="Previous events.jsonl; reuse motion history, never replay actions")
    args = p.parse_args()
    key = os.getenv("VLM_API_KEY")
    missing = [name for name, value in (("VLM_API_KEY", key), ("--base-url", args.base_url),
                                        ("--model", args.model), ("--nuc", args.nuc),
                                        ("--cameras", args.cameras)) if not value]
    if missing or not args.vision_confirmed:
        p.error("Missing " + ", ".join(missing + ([] if args.vision_confirmed else ["--vision-confirmed"])) +
                "; all are required (--vision-confirmed asserts the endpoint accepts image_url inputs)")
    if len(args.cameras) != 3 or len(set(args.cameras)) != 3 or not all(s.isdigit() for s in args.cameras):
        p.error("Three distinct numeric camera serials required")
    if args.max_steps < 1 or any(not math.isfinite(x) or x <= 0 for x in (
            args.max_seconds, args.max_travel_m, args.max_gripper_travel,
            args.max_observation_age, args.api_timeout, args.arrival_timeout,
            args.max_step_m, args.max_gripper_step, args.sub_step_m,
            args.speed_factor, args.arrival_tolerance_m)) or args.max_joint_step_rad < 0 or args.min_step_m < 0:
        p.error("Budgets/timeouts must be finite and positive")
    if (args.max_sub_steps < 1 or args.max_tokens < 1 or not math.isfinite(args.min_step_m)
            or not math.isfinite(args.max_joint_step_rad) or args.min_step_m > args.max_step_m
            or args.speed_factor > 1):
        p.error("Invalid sub-step count, token budget, step limit or dynamics factor")
    workspace = None
    if args.workspace_min is not None and args.workspace_max is not None:
        lo, hi = finite_vector(args.workspace_min, 3), finite_vector(args.workspace_max, 3)
        if np.any(lo >= hi):
            p.error("Invalid workspace bounds")
        workspace = (lo, hi)
    elif args.workspace_min is not None or args.workspace_max is not None:
        p.error("--workspace-min and --workspace-max must be given together")
    client = VisionClient(args.base_url, args.model, key, args.api_timeout, args.thinking, args.max_tokens,
                          args.allow_http)
    root = args.log_dir / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, mode=0o700)
    # Prevent two copies of THIS harness on PC2; not a robot-wide ownership lock.
    lock_path = Path.home() / ".cache" / "droid-vlm-harness.lock"
    lock_path.parent.mkdir(exist_ok=True)
    robot = cameras = None
    history, travel, grip_travel = load_history(args.resume_log), 0., 0.
    no_progress = 0
    hold_orientation = None
    rejects = 0
    deadline = time.monotonic() + args.max_seconds
    with lock_path.open("a") as lock, (root / "events.jsonl").open("a") as log:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def record(event):
            log.write(json.dumps(dict(time=time.time(), **event)) + "\n")
            log.flush()

        try:
            robot = DroidRobot(args.nuc)
            cameras = ThreeCameras(args.cameras)
            if args.record_video:
                cameras.start_recording(root / "video")
                record(dict(event="recording_started", path=str(root / "video")))
            for step in range(args.max_steps):
                if time.monotonic() >= deadline:
                    record(dict(event="budget_stop", reason="time")); break
                folder = root / f"step-{step:04d}"
                folder.mkdir()
                before, before_status, _ = robot.observe()
                images = cameras.capture(folder)
                state, status, stamp = robot.observe()
                if hold_orientation is None:
                    hold_orientation = np.asarray(state["cartesian_position"][3:], dtype=float)
                check_status_unchanged(before_status, status)
                pe, re = pose_error(before["cartesian_position"], state["cartesian_position"])
                if pe > .003 or re > .05:
                    raise RuntimeError("Robot moved during camera acquisition")
                current_uv = next((x.get("yellow_uv") for x in images if x["role"] == "wrist"), None)
                attach_after_observation(history, state, current_uv)
                limits = dict(max_translation_norm_m=args.max_step_m, min_translation_norm_m=args.min_step_m,
                              max_abs_gripper_delta=args.max_gripper_step,
                              gripper_enabled=args.allow_gripper, remaining_travel_m=args.max_travel_m-travel,
                              max_single_dispatch_m=args.sub_step_m, max_dispatches=args.max_sub_steps)
                if workspace is not None:
                    limits["workspace_min_m"] = workspace[0].tolist()
                    limits["workspace_max_m"] = workspace[1].tolist()
                record(dict(event="observation", step=step, state=state, robot_timestamp=stamp,
                            images=[{k:v for k,v in x.items() if k != "url"} for x in images]))
                try:
                    raw = client.decide(args.task, state, images, history, limits)
                except (ValueError, RuntimeError) as err:
                    # A malformed, truncated or failed answer costs one step rather
                    # than the session: the next step re-observes and re-plans.
                    rejects += 1
                    history.append(dict(action=None, result="model_failed", error=str(err)))
                    record(dict(event="model_failed", step=step, error=str(err)))
                    print(json.dumps(dict(step=step, model_failed=str(err))), flush=True)
                    continue
                record(dict(event="model_response", step=step, text=raw))
                try:
                    action = normalize_action(raw, args.max_step_m, args.max_gripper_step)
                    record(dict(event="normalized_action", step=step, action=action.as_dict()))
                except (ValueError, TypeError) as err:
                    # A single malformed answer should not end a session the robot
                    # is already positioned for; the next step re-plans.
                    rejects += 1
                    history.append(dict(action=None, result="rejected", error=str(err)))
                    record(dict(event="action_rejected", step=step, error=str(err), text=raw[:600]))
                    print(json.dumps(dict(step=step, rejected=str(err))), flush=True)
                    continue
                if action.stop:
                    record(dict(event="model_requested_stop", verified_success=False, reason=action.reason, phase=action.phase)); break
                if time.monotonic() - min(x["captured_monotonic"] for x in images) > args.max_observation_age:
                    record(dict(event="reobserve", step=step, reason="stale images; no action sent"))
                    continue
                distance = float(np.linalg.norm(action.delta))
                if travel + distance > args.max_travel_m + 1e-12 or grip_travel + abs(action.grip_delta) > args.max_gripper_travel + 1e-12:
                    raise RuntimeError("Cumulative action budget exceeded")
                try:
                    validate_target(action, state, workspace, args.allow_gripper)
                except ValueError as err:
                    # A step that would leave the operator's workspace or drive the
                    # gripper into a limit costs one step, not the session.
                    rejects += 1
                    history.append(dict(action=action.as_dict(), result="rejected", error=str(err),
                                        wrist_yellow_uv=next((x.get("yellow_uv") for x in images
                                                              if x["role"] == "wrist"), None)))
                    record(dict(event="action_rejected", step=step, error=str(err)))
                    print(json.dumps(dict(step=step, rejected=str(err))), flush=True)
                    continue
                rejects = 0
                print(json.dumps(dict(step=step, execute=args.execute, **action.as_dict())), flush=True)
                record(dict(event="dispatch_intent" if args.execute else "dry_run", step=step, action=action.as_dict()))
                observed_before_action = state
                if args.execute:
                    # The backend's IK caps one dispatch at roughly 0.05 rad on the
                    # most loaded joint (~2 cm of tool motion here), so a larger
                    # requested step is served by repeated dispatches, each
                    # re-solved from the measured state, like streaming teleop.
                    start_pose = np.asarray(state["cartesian_position"], dtype=float)
                    remaining = np.asarray(action.delta, dtype=float)
                    result = dict(state=state, controller_status=status, result="no_op")
                    # A gripper action has no translation to split but must still
                    # reach execute(). A translation stops once the residual is
                    # within tolerance rather than chasing millimetres with more
                    # dispatches, whose coupling would drag the tool off target.
                    tolerance = min(.003, max(.0015, .1 * float(np.linalg.norm(action.delta))))
                    attempt, done, dispatches = 0, False, 0
                    needs_reobserve = False
                    for attempt in range(1 if not np.any(action.delta) else args.max_sub_steps):
                        if not np.any(action.delta):
                            sub = action
                        elif dispatches > 0 and float(np.linalg.norm(remaining)) <= tolerance:
                            done = True
                            break
                        else:
                            sub = Action(remaining * min(1., args.sub_step_m / float(np.linalg.norm(remaining))),
                                         0., False, action.reason, action.phase)
                        result = robot.execute(sub, state, status, workspace, args.allow_gripper,
                                               args.arrival_timeout,
                                               deadline,
                                               max_joint_step=args.max_joint_step_rad, speed_factor=args.speed_factor,
                                               arrival_tolerance=args.arrival_tolerance_m,
                                               hold_orientation=hold_orientation)
                        dispatches += 1
                        if result["result"] not in ("settled", "partial_step", "reobserve", "no_op", "gripper_no_change"):
                            raise RuntimeError(f"Execution did not settle: {result['result']}")
                        state, status = result["state"], result["controller_status"]
                        achieved = np.asarray(result.get("observed_delta_m") or [0., 0., 0.], dtype=float)
                        if np.any(action.delta):
                            remaining = remaining - achieved
                        record(dict(event="sub_step", step=step, attempt=attempt, wanted=sub.delta.tolist(),
                                    achieved=achieved.tolist(), remaining=remaining.tolist(),
                                    execution_result=result["result"], joint_step_scale=result.get("joint_step_scale", 1.0)))
                        if result["result"] == "reobserve" or result.get("joint_step_scale", 1.0) < 1.0:
                            needs_reobserve = True
                            break
                    if np.any(action.delta):
                        total = np.asarray(state["cartesian_position"][:3], dtype=float) - start_pose[:3]
                        result = dict(state=state, controller_status=status, dispatches=dispatches,
                                      observed_delta_m=total.tolist(),
                                      outstanding_m=float(np.linalg.norm(remaining)),
                                      result=("reobserve" if needs_reobserve else "settled" if done or float(np.linalg.norm(remaining)) <= tolerance
                                              else "partial_step"))
                    travel += distance
                    grip_travel += abs(action.grip_delta)
                else:
                    result = dict(state=state, result="dry_run_NOT_executed")
                history.append(dict(action=action.as_dict(),
                                    state_before=observed_before_action,
                                    images_before=[{"role": i["role"], "path": i["path"]} for i in images],
                                    wrist_yellow_uv_before=current_uv,
                                    wrist_yellow_uv_after=None,
                                    **result))
                record(dict(event="result", step=step, **history[-1]))
                actual = float(np.linalg.norm(result.get("observed_delta_m") or [0., 0., 0.]))
                grip_moved = abs(state["gripper_position"] - observed_before_action["gripper_position"])
                no_progress = no_progress + 1 if args.execute and actual < .0003 and grip_moved < .005 else 0
                if no_progress >= 3:
                    record(dict(event="reobserve", step=step, reason="no progress; replan alignment"))
                    history.append(dict(result="no_progress", error="Reassess alignment using fresh images; do not repeat the same action"))
                    no_progress = 0
            else:
                record(dict(event="budget_stop", reason="steps"))
        except BaseException as err:
            record(dict(event="abort", error=str(err), exception=type(err).__name__))
            print("Loop stopped; no further commands. This is NOT a hardware emergency stop.", flush=True)
            raise
        finally:
            if cameras is not None:
                cameras.close()
            if robot is not None:
                robot.close()
    print(f"Logs: {root}")
