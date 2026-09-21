"""Motion history: before/after observation pairing and cross-run resume."""
import json
from pathlib import Path

from .safety import pose_error


def attach_after_observation(history, state, uv):
    if not history or not history[-1].get("state"):
        return
    previous = history[-1]
    pe, re = pose_error(previous["state"]["cartesian_position"], state["cartesian_position"])
    if pe <= .003 and re <= .02:
        previous["wrist_yellow_uv_after"] = uv
        previous["after_observation_available"] = True


def compact_history(history):
    keys = ("action", "result", "observed_delta_m", "outstanding_m", "error",
            "wrist_yellow_uv_before", "wrist_yellow_uv_after", "after_observation_available")
    result = []
    for entry in history[-4:]:
        item = {k: entry[k] for k in keys if k in entry}
        for name in ("state_before", "state"):
            if name in entry:
                item[name] = {k: entry[name][k] for k in ("cartesian_position", "gripper_position")}
        if item.get("action"):
            item["action"] = {k: v for k, v in item["action"].items() if k != "reason"}
        result.append(item)
    return result


def load_history(path):
    if path is None:
        return []
    history = []
    observation_images = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "result" and event.get("result") != "dry_run_NOT_executed":
            event.setdefault("wrist_yellow_uv_before", event.get("wrist_yellow_uv"))
            event.setdefault("images_before", observation_images)
            history.append(event)
        elif event.get("event") == "observation":
            observation_images = [{"role": i["role"], "path": i["path"]} for i in event.get("images", [])]
            uv = next((i.get("yellow_uv") for i in event.get("images", []) if i["role"] == "wrist"), None)
            attach_after_observation(history, event["state"], uv)
    return history[-12:]
