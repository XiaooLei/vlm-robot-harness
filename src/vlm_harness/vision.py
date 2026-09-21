"""OpenAI-compatible vision client: three views per step, prior-frame comparison."""
import base64
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from .history import compact_history
from .prompts import SYSTEM, VIEW_LABELS, wrist_view_axes


def thinking_body(style, base_url):
    """Provider-specific switch that keeps the answer out of a reasoning budget."""
    host = urllib.parse.urlsplit(base_url).hostname or ""
    if style == "auto":
        style = "dashscope" if "dashscope" in host else ("deepseek" if "deepseek" in host else "none")
    if style == "deepseek":
        return {"thinking": {"type": "disabled"}}
    if style == "dashscope":
        return {"enable_thinking": False}
    return {}


class VisionClient:
    def __init__(self, base_url, model, key, timeout, thinking="auto", max_tokens=160, allow_http=False):
        url = urllib.parse.urlsplit(base_url)
        schemes = ("http", "https") if allow_http else ("https",)
        if url.scheme not in schemes or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("Model endpoint must use HTTPS; camera images and API credentials are sent to it."
                             " Pass --allow-http to override for a trusted plaintext endpoint.")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.key, self.timeout = model, key, timeout
        self.thinking, self.max_tokens = thinking_body(thinking, base_url), max_tokens

    def decide(self, task, state, images, history, limits):
        wrist = next((x for x in images if x["role"] == "wrist"), None)
        content = [{"type": "text", "text": json.dumps(dict(task=task, robot_state={k: state[k] for k in ("cartesian_position", "gripper_position")},
                    recent_steps=compact_history(history), limits=limits, orientation="hold current",
                    wrist_view=wrist_view_axes(state["cartesian_position"]),
                    wrist_yellow_uv=None if wrist is None else wrist.get("yellow_uv")))}]
        # Compare fixed-camera BEFORE frames with CURRENT frames to distinguish
        # a grasp from a stationary banana merely overlapping the moving jaws.
        previous = next((x for x in reversed(history) if x.get("images_before") and x.get("action")), None)
        if previous:
            content.append({"type": "text", "text": "PREVIOUS external views BEFORE the last executed action: " +
                            json.dumps(dict(action=previous["action"], observed_delta_m=previous.get("observed_delta_m")))})
            for old in previous["images_before"]:
                path = Path(old["path"])
                if old["role"] != "wrist" and path.is_file():
                    content.extend([{"type": "text", "text": "PREVIOUS " + VIEW_LABELS.get(old["role"], old["role"])},
                                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                                     base64.b64encode(path.read_bytes()).decode()}}])
        content.append({"type": "text", "text": "CURRENT live views below. Compare banana movement with tool movement, especially after lifting or release."})
        for image in images:
            content.extend([{"type": "text", "text": f"Camera {VIEW_LABELS.get(image['role'], image['role'])}, serial {image['serial']}"},
                            {"type": "image_url", "image_url": {"url": image["url"]}}])
        # Both tested providers default to a reasoning mode whose tokens are
        # spent before any content appears, so the switch is set explicitly.
        body = dict(model=self.model, messages=[{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": content}], max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}, stream=False, **self.thinking)
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + self.key,
                                              "Content-Type": "application/json"})
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                raise RuntimeError("Model API redirect refused; credentials/images not forwarded")

        # Transport errors mean no request completed and nothing was dispatched,
        # so retrying those is safe. There is still no fallback to a text-only
        # model, and never a retry of a motion command.
        for attempt in range(3):
            try:
                with urllib.request.build_opener(NoRedirect()).open(req, timeout=self.timeout) as response:
                    data = json.loads(response.read(2_000_000))
                break
            except urllib.error.HTTPError as err:
                raise RuntimeError(f"Model API HTTP {err.code}; check vision/JSON support, URL and model. No action sent.") from None
            except urllib.error.URLError as err:
                if attempt == 2:
                    raise RuntimeError(f"Model API connection failed ({err.reason}); no action sent") from None
                time.sleep(1.5)
        choice = data["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("Model answer incomplete or rejected")
        return choice["message"]["content"]
