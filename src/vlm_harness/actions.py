"""Parsing and normalization of the model's action JSON."""
from dataclasses import dataclass
import json
import math

import numpy as np


def finite_vector(value, length):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"Expected a {length}-element list")
    if any(type(x) not in (float, int) or not math.isfinite(x) for x in value):
        raise ValueError("Action components must be finite numbers, not booleans")
    return np.asarray(value, dtype=float)


@dataclass(frozen=True)
class Action:
    delta: np.ndarray
    grip_delta: float
    stop: bool
    reason: str
    phase: str = "unspecified"

    def as_dict(self):
        result = dict(eef_delta_m=self.delta.tolist(), gripper_delta=self.grip_delta,
                      stop=self.stop, phase=self.phase)
        if self.reason:
            result["reason"] = self.reason  # preserve older logs; not requested from model
        return result


def parse_action(text, max_step=.1, max_grip=.5, min_step=0.):
    def reject_constant(value):
        raise ValueError(f"Invalid JSON number: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    if not isinstance(text, str) or len(text) > 8192:
        raise ValueError("Missing/oversized model JSON")
    obj = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        obj = obj[0]  # some providers wrap the object in a one-element array
    # Unknown keys are ignored: providers sometimes echo their own wrappers
    # (e.g. "type": "json_object"), and only these fields are ever read.
    if not isinstance(obj, dict) or not ({"eef_delta_m", "gripper_delta"} & set(obj)):
        raise ValueError("Unexpected action schema")
    delta = finite_vector(obj.get("eef_delta_m", [0, 0, 0]), 3)
    grip = finite_vector([obj.get("gripper_delta", 0)], 1)[0]
    stop = obj.get("stop", False)
    if type(stop) is not bool:
        raise ValueError("Invalid stop flag")
    reason = obj.get("reason", "")
    phase = obj.get("phase", "unspecified")
    if not isinstance(reason, str) or len(reason) > 600:
        raise ValueError("Invalid action reason")
    if not isinstance(phase, str) or phase not in ("unspecified", "align", "descend", "grasp", "lift", "transfer", "release", "verify", "blocked"):
        raise ValueError("Invalid action phase")
    moving = bool(np.any(delta != 0))
    # Tolerance, not truncation: a model aiming at the limit rounds just over it.
    step_slack, grip_slack = max(.002, .02 * max_step), max(.005, .02 * max_grip)
    if np.linalg.norm(delta) > max_step + step_slack or abs(grip) > max_grip + grip_slack:
        raise ValueError("Action exceeds limits; rejected, NOT silently clipped")
    if moving and min_step > 0 and np.linalg.norm(delta) < min_step - 1e-12:
        raise ValueError(f"Translation below the {min_step} m minimum; step at least that far or hold still")
    if moving and grip != 0:
        raise ValueError("Translate and operate gripper in separate steps")
    if stop and (moving or grip != 0):
        raise ValueError("Stop action must have zero deltas")
    return Action(delta, float(grip), stop, reason, phase)


def normalize_action(text, max_step, max_grip):
    """Normalize model presentation and bound requests before strict dispatch parsing."""
    if not isinstance(text, str) or len(text) > 8192:
        raise ValueError("Missing/oversized model JSON")
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines[0].lower() in ("```", "```json"):
            text = "\n".join(lines[1:-1])
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    obj = json.loads(text, object_pairs_hook=unique)
    if isinstance(obj, list) and len(obj) == 1:
        obj = obj[0]
    if not isinstance(obj, dict) or not ({"eef_delta_m", "gripper_delta"} & obj.keys()):
        raise ValueError("Missing action fields")
    delta = finite_vector(obj.get("eef_delta_m", [0, 0, 0]), 3)
    grip = float(finite_vector([obj.get("gripper_delta", 0)], 1)[0])
    if type(obj.get("stop", False)) is not bool:
        raise ValueError("Invalid stop flag")
    # Scale in two stages to avoid overflow on large but finite model values.
    largest = float(np.max(np.abs(delta)))
    if largest:
        unit = delta / largest
        norm = float(np.linalg.norm(unit))
        if largest > max_step / norm:
            delta = unit * (max_step / norm)
    grip = float(np.clip(grip, -max_grip, max_grip))
    phase = obj.get("phase", "unspecified")
    phases = ("unspecified", "align", "descend", "grasp", "lift", "transfer", "release", "verify", "blocked")
    if not isinstance(phase, str) or phase not in phases:
        phase = "unspecified"
    if obj.get("stop", False):
        delta, grip = np.zeros(3), 0.
    elif np.any(delta):
        grip = 0.  # Translate first; next fresh observation can request grip.
    return parse_action(json.dumps(dict(eef_delta_m=delta.tolist(), gripper_delta=grip,
        stop=obj.get("stop", False), phase=phase)), max_step, max_grip, 0.)
