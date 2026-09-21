"""VLM-in-the-loop robot harness.

A single-action-at-a-time agent loop for a Franka arm driven through the
existing DROID zerorpc backend, with three ZED camera views per step.

The public surface is re-exported here so callers can use either::

    import vlm_harness
    vlm_harness.parse_action(text)

or the narrower modules (``vlm_harness.robot``, ``vlm_harness.cameras``, ...).
"""
from .actions import Action, finite_vector, normalize_action, parse_action
from .cameras import ThreeCameras
from .history import attach_after_observation, compact_history, load_history
from .loop import main
from .prompts import SYSTEM, VIEW_LABELS, wrist_view_axes
from .robot import DroidRobot
from .safety import (MEASURED_SLACK, WORKSPACE_SLACK, check_status_unchanged,
                     pose_error, require_stationary, validate_target)
from .vision import VisionClient, thinking_body

__version__ = "0.1.0"

__all__ = [
    "Action", "DroidRobot", "MEASURED_SLACK", "SYSTEM", "ThreeCameras", "VIEW_LABELS",
    "VisionClient", "WORKSPACE_SLACK", "attach_after_observation", "check_status_unchanged",
    "compact_history", "finite_vector", "load_history", "main", "normalize_action",
    "parse_action", "pose_error", "require_stationary", "thinking_body", "validate_target",
    "wrist_view_axes",
]
