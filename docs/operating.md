# Operating notes

These are the measured characteristics of the rig this harness was extracted
from. They are **observations, not guarantees**; re-measure on your own setup.

## Endpoint and cameras

Set `VLM_API_KEY` and give `--base-url` and `--model` (or `VLM_BASE_URL`,
`VLM_MODEL`) for every run: nothing deployment-specific is baked into the code.
The same applies to the robot host and the three camera serials
(`--nuc`/`VLM_NUC`, `--cameras`/`VLM_CAMERAS`).
The model must accept multiple `image_url` inputs and return JSON;
`--vision-confirmed` is an operator assertion, not a capability check.
Provider-specific thinking switches are chosen with
`--thinking auto|dashscope|deepseek|none`.

Plaintext endpoints are refused unless `--allow-http` is passed, because the
camera images and the API key travel to that endpoint.

Each step sends three views: wrist, exterior_1 and exterior_2. When executed
history exists, the two *previous* exterior frames are attached as BEFORE
references next to the three CURRENT frames, so the model can distinguish "the
banana moved with the tool" from "the tool moved and the banana stayed put".
After physically moving a camera, start fresh history: old and new camera poses
are not comparable.

## Action contract

```json
{"eef_delta_m":[0,0,-0.005],"gripper_delta":0,"stop":false,"phase":"descend"}
```

XYZ is translation in **robot base** coordinates, metres, +Z up. Orientation is
held at the session reference. `gripper_delta` is normalized: positive closes,
0=open and 1=closed absolute; 0.5 is half the stroke and is *not* proof of
contact. Translation and gripper motion are separate steps. Phases: `align`,
`descend`, `grasp`, `lift`, `transfer`, `release`, `verify`, `blocked`.

The loop normalizes fenced JSON and single-element array wrappers, scales
oversized translation/gripper requests to the limits, drops unknown phase labels
and optional text, and turns `stop` into zero motion. Raw model responses and
normalized actions are both logged.

## Execution behaviour

* The backend's EEF→IK path does **not** reproduce a requested Cartesian delta
  exactly. Measured on this rig: a request saturates at about 0.05 rad on the
  most loaded joint, i.e. roughly 2 cm of tool motion per dispatch. Everything
  beyond that is silently clamped by the IK, which is why a 10 cm request used
  to produce ~2 cm of motion.
* `--sub-step-m` splits one model action into repeated dispatches that are each
  re-solved from the measured state, which restores the full requested step.
  With `--max-sub-steps 1` (the default) one model action is one dispatch, and
  the model is expected to plan from the resulting fresh images.
* IK increments above `--max-joint-step-rad` are scaled uniformly to the limit,
  logged as `joint_step_scale`, and followed by fresh vision rather than a blind
  dispatch of the remainder.
* Arrival requires actual progress, low measured velocity, no pending controller
  motion, and bounded cross-track and orientation error. Small stalled steps can
  return `reobserve`, which stops the remaining sub-steps and demands new images.
* Arrival timeout, collision flags, controller recovery/reconnect, competing
  commands and workspace breaches stop the run. The client never retries a
  motion command and never recovers the robot itself.
* Three steps without measured progress request a fresh observation. Three
  malformed or rejected model answers no longer abort the session.

## Gripper conventions (measured)

`gripper_position` in the robot state is normalized with **0 = open,
1 = closed**; the state docstring in one backend build claims the opposite and
is wrong. Verified on hardware:

| Command | Result |
| --- | --- |
| `update_gripper(0.0, velocity=False)` | fully open (`0.0118`) |
| `update_gripper(1.0, velocity=False)` | fully closed (`0.9020`) |

Repeated identical targets are dropped by the backend deadband and return
`False`, which the harness reports as `gripper_no_change` rather than a fault.
Opening after contact needs an explicit release path on backends whose gripper
library filters the first open request after a grip.

## Safety envelope

The workspace box bounds the commanded end-effector pose. It is **not** finger
or table clearance, and it is not collision detection: on this rig the tool
contacts the table around `z ≈ 0.02 m`, which was found the hard way.

Stop is not a hardware emergency stop. Ctrl-C stops future submissions; it does
not undo motion the backend has already accepted, and the backend may recover
internally. Keep a physical stop in reach and supervise every live run.

## Model behaviour worth knowing

* The wrist camera sits beside the tool, so the tool axis is neither the image
  centre nor a fixed pixel. Aiming "the object at the image centre" is wrong;
  the jaws are visible in the wrist view and are the correct reference.
* Base-frame orientation must be given to the model explicitly. On this rig
  +X points away from the robot into the table and +Y is the robot's left; both
  were confirmed with single-axis probes watched by an operator.
* Two operator-watched probes (each a few centimetres along one base axis, with
  the resulting image shift measured) settle the local image→base mapping far
  more reliably than any fixed equation: the mapping direction is stable but its
  magnitude changes with height and configuration.
