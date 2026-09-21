# vlm-robot-harness

A small, self-contained harness that lets a vision-language model drive a Franka
arm through an existing robot RPC backend, using three cameras per step.

It was extracted from a DROID/FR3 tabletop setup, but it contains **no import of
the robot vendor code**: the robot side is spoken over zerorpc, and cameras are
spoken to through the ZED SDK. Point it at your own endpoints and it works
unchanged.

## What one step does

```
read robot state ─┐
                  ├─ capture wrist + left + right views
read state again ─┘        (reject if the arm moved during capture)
        │
        ├─ ask the model for ONE action: eef_delta_m / gripper_delta / stop / phase
        ├─ normalize, validate against limits + operator workspace
        ├─ dispatch, wait for measured arrival, verify progress and orientation
        └─ feed the measured displacement and the new views into the next step
```

Actions are single steps, never trajectories: at most one translation *or* one
gripper change per decision, orientation held at the session reference.

## Layout

| Module | Responsibility |
| --- | --- |
| `actions.py` | action JSON parsing and normalization (`Action`, `parse_action`, `normalize_action`) |
| `prompts.py` | the system prompt, view labels, wrist-view geometry hints |
| `safety.py` | bounds, workspace validation, controller-health checks |
| `robot.py` | robot control: zerorpc client, dispatch, measured arrival, joint scaling |
| `cameras.py` | ZED acquisition: three views, JPEG + blob feedback, optional MP4 recording |
| `vision.py` | OpenAI-compatible vision client, prior-frame comparison, transport retries |
| `history.py` | before/after observation pairing, cross-run resume |
| `loop.py` | the session loop and CLI |

## Requirements

* Python 3.9+
* `numpy` (core), `scipy` (pose math)
* Robot side: `zerorpc` client, plus a backend exposing the RPCs described below
* Cameras: `pyzed` (ZED SDK) and `opencv-python`

```bash
pip install -e .
# robot-side extras, per machine:
pip install zerorpc opencv-python    # pyzed comes with the ZED SDK
```

## Backend contract

The harness is a client. It expects a zerorpc server at `--nuc <host>:4242`
exposing these methods:

| RPC | Used for | Optional |
| --- | --- | --- |
| `get_robot_state()` | `(state_dict, timestamp)`; needs `cartesian_position`, `joint_positions`, `joint_velocities`, `gripper_position`, `franka_joint_collision`, `franka_cartesian_collision` | required |
| `get_connection_status()` | `robot_connected`, `last_robot_error`, `recover_count`, `reconnect_count`, `joint_async_busy`, `joint_async_has_pending`, `joint_async_submit_count`, `joint_async_error_count` | required |
| `get_gripper_async_status()` | `gripper_available`, busy/pending and submit/error counters | required |
| `create_action_dict(action, action_space, gripper_action_space, robot_state)` | EEF→joint IK; the harness reads `joint_position` and dispatches only those 7 joints | required |
| `move_to_joint_positions(joints, speed_factor)` | the single arm dispatch per sub-step | required |
| `update_gripper(position, velocity=False, blocking=False)` | absolute gripper target, 0=open 1=closed | required |
| `move_to_joint_positions_checked(...)` | dispatch that reports real controller completion | optional, advertised via `checked_motion_supported` |
| `release_gripper(position)` | explicit open after contact, bypassing a gripper library's object-detection filter | optional, advertised via `explicit_release_supported` |

Anything the server does beyond this — its own recovery, retries, collision
reflexes — is left untouched by the harness.

## Quickstart

```bash
# Nothing deployment-specific is baked in. Provide it, or set these:
export VLM_API_KEY=...                      # never commit this
export VLM_BASE_URL=https://your-host/v1
export VLM_MODEL=your-vision-model
export VLM_NUC=your-robot-host              # runs the RPC server on :4242
export VLM_CAMERAS="11111111 22222222 33333333"   # wrist, exterior_1, exterior_2

# dry run: reads the robot and the cameras, calls the model, dispatches nothing
python -m vlm_harness \
  --task "Pick up the yellow banana and put it in the beige box" \
  --base-url "$VLM_BASE_URL" --model "$VLM_MODEL" \
  --vision-confirmed --max-steps 1

# live, bounded, with an operator-supplied workspace box
python -m vlm_harness \
  --task "Pick up the yellow banana and put it in the beige box" \
  --base-url "$VLM_BASE_URL" --model "$VLM_MODEL" \
  --vision-confirmed --execute --allow-gripper \
  --workspace-min -0.30 -0.60 0.03 --workspace-max 0.90 0.20 0.90 \
  --max-steps 15 --max-travel-m 0.3 --max-seconds 300
```

Every run writes `events.jsonl`, per-step JPEGs and (with `--record-video`) three
MP4s under `runs/agent_harness/<timestamp>/`. `--resume-log <events.jsonl>`
reuses a previous run's motion history without replaying any command.

## Safety

This is research code that moves real hardware. It defaults to **dry run**,
refuses to dispatch without explicit flags, and stops rather than retrying when
motion does not settle. It does not implement collision avoidance, a force
limit, or an emergency stop: keep a physical stop within reach and supervise
every live run.

The operator-supplied workspace box is a bound on the *commanded* end-effector
pose, not finger or table clearance. A camera can make the jaws look like they
straddle an object while they are still above it.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The suite is hardware-free: no robot, no camera and no network call is made.

## Status

Extracted from a working FR3 tabletop experiment. Pick-and-place has been run
end to end on real hardware, but reliability is not there yet: the model's
image-to-base spatial reasoning is the limiting factor, not the transport.

See `docs/operating.md` for the measured characteristics of the backend this was
built against (dynamics caps, gripper conventions, workspace behaviour).
