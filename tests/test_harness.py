import json
from pathlib import Path
import sys
import time
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import vlm_harness as h  # noqa: E402

# ``main`` reads its collaborators as module globals of ``vlm_harness.loop``, so
# that is where the loop-level patches below have to land.
L = h.loop


def state():
    return dict(cartesian_position=[.36, 0., .34, 0., 0., 0.], joint_positions=[0.] * 7,
                joint_velocities=[0.] * 7, gripper_position=.1,
                franka_joint_collision=[0.] * 7, franka_cartesian_collision=[0.] * 6)


def status():
    return dict(robot_connected=True, last_robot_error=None, recover_count=0, reconnect_count=0,
                joint_async_busy=False, joint_async_has_pending=False,
                joint_async_submit_count=0, joint_async_error_count=0)


def grip_status():
    return dict(gripper_available=True, gripper_async_busy=False, gripper_async_has_pending=False,
                gripper_async_submit_count=0, gripper_async_error_count=0)


def action(**updates):
    obj = dict(eef_delta_m=[.001, 0, 0], gripper_delta=0, stop=False, reason="test")
    obj.update(updates)
    return json.dumps(obj)


class HarnessTests(unittest.TestCase):
    def test_normalizer_scales_and_ignores_advisory_fields(self):
        raw = '```json\n' + action(eef_delta_m=[.06,.08,0], phase=123, reason=None) + '\n```'
        a = h.normalize_action(raw, .02, .8)
        np.testing.assert_allclose(a.delta,[.012,.016,0])
        self.assertEqual(a.phase, "unspecified")
        self.assertEqual(a.reason, "")
        a = h.normalize_action(action(eef_delta_m=[0,0,0],gripper_delta=-2),.02,.8)
        self.assertEqual(a.grip_delta,-.8)

    def test_normalizer_preserves_stop_and_separates_gripper(self):
        a=h.normalize_action(action(stop=True,gripper_delta=.4),.02,.8)
        self.assertTrue(a.stop);self.assertFalse(np.any(a.delta));self.assertEqual(a.grip_delta,0)
        a=h.normalize_action(action(gripper_delta=.4),.02,.8)
        self.assertTrue(np.any(a.delta));self.assertEqual(a.grip_delta,0)

    def test_normalizer_never_accepts_nonfinite_or_ambiguous_actions(self):
        for raw in (action(eef_delta_m=[float('nan'),0,0]),action(gripper_delta=float('inf')),
                    action(stop='false'), '{"eef_delta_m":[0,0,0],"stop":true,"stop":false}'):
            with self.assertRaises(ValueError):h.normalize_action(raw,.02,.8)
        a=h.normalize_action(action(eef_delta_m=[1e308,1e308,0]),.02,.8)
        self.assertLessEqual(np.linalg.norm(a.delta),.02000001)

    def test_valid_action(self):
        a = h.parse_action(action())
        np.testing.assert_array_equal(a.delta, [.001, 0, 0])

    def test_reject_untrusted_formats_and_limits(self):
        bad = ["```json\n" + action() + "\n```", "[]", "null", "{}",
               action(eef_delta_m=[True, 0, 0]), action(eef_delta_m=[float("nan"), 0, 0]),
               action(eef_delta_m=[float("inf"), 0, 0]), action(eef_delta_m=[.08, .08, 0]),
               action(gripper_delta=.52, eef_delta_m=[0, 0, 0]),
               action(stop=True), action(stop="false"),
               action(gripper_delta=.01), action(eef_delta_m=[0, 0]),
               action().replace('"stop": false', '"stop": false, "stop": true')]
        for text in bad:
            with self.subTest(text=text), self.assertRaises((ValueError, TypeError)):
                h.parse_action(text)

    def test_step_limits_are_configurable(self):
        # A translation defaults to the 1-10 cm band: the cap still rejects overshoot.
        np.testing.assert_allclose(h.parse_action(action(eef_delta_m=[.1, 0, 0])).delta, [.1, 0, 0])
        for text in (action(eef_delta_m=[.105, 0, 0]), action(eef_delta_m=[.08, .08, 0])):
            with self.subTest(text=text), self.assertRaises(ValueError):
                h.parse_action(text)
        # A model that rounds a step onto the limit is accepted, and a one-element
        # array wrapper is unwrapped.
        h.parse_action(action(eef_delta_m=[-.071, -.071, 0]))
        np.testing.assert_allclose(h.parse_action("[" + action(eef_delta_m=[.02, 0, 0]) + "]").delta, [.02, 0, 0])
        h.parse_action(action(eef_delta_m=[.02, 0, 0]), .03, .1)
        with self.assertRaises(ValueError):
            h.parse_action(action(eef_delta_m=[.04, 0, 0]), .03, .1)

    def test_minimum_step_is_enforced_only_for_moving_steps(self):
        for text in (action(eef_delta_m=[.009, 0, 0]), action(eef_delta_m=[.005, .005, 0])):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "minimum"):
                h.parse_action(text, .05, .1, .01)
        h.parse_action(action(eef_delta_m=[.01, 0, 0]), .05, .1, .01)
        # A held position, and a gripper-only step, stay legal.
        h.parse_action(action(eef_delta_m=[0, 0, 0]), .05, .1, .01)
        h.parse_action(action(eef_delta_m=[.001, 0, 0]), .05, .1, 0.)

    def test_stop_is_not_motion(self):
        self.assertTrue(h.parse_action(action(stop=True, eef_delta_m=[0, 0, 0])).stop)

    def test_extra_keys_are_ignored_not_fatal(self):
        a = h.parse_action(action(extra="code", type="json_object"))
        np.testing.assert_allclose(a.delta, [.001, 0, 0])
        self.assertFalse(a.stop)

    def test_workspace_and_gripper_bounds(self):
        bounds = (np.array([.3, -.1, .3]), np.array([.4, .1, .4]))
        target, _ = h.validate_target(h.parse_action(action()), state(), bounds, False)
        self.assertAlmostEqual(target[0], .361)
        with self.assertRaises(ValueError):
            h.validate_target(h.parse_action(action()), state(), (np.zeros(3), np.ones(3) * .1), False)
        a = h.parse_action(action(eef_delta_m=[0, 0, 0], gripper_delta=-.03))
        with self.assertRaises(ValueError):
            h.validate_target(a, state(), None, False)
        s = state(); s["gripper_position"] = .01
        with self.assertRaises(ValueError):
            h.validate_target(a, s, None, True)

    def test_stationary_preflight(self):
        h.require_stationary(state(), status())
        for field, value in (("joint_velocities", [.05] * 7), ("franka_joint_collision", [1.] * 7),
                             ("cartesian_position", [float("nan")] * 6)):
            s = state(); s[field] = value
            with self.assertRaises((ValueError, RuntimeError)):
                h.require_stationary(s, status())
        st = status(); st["joint_async_has_pending"] = True
        with self.assertRaises(RuntimeError):
            h.require_stationary(state(), st)

    def robot(self):
        robot = h.DroidRobot.__new__(h.DroidRobot)
        robot.rpc = Mock()
        robot.rpc.get_robot_state.return_value = (state(), {})
        robot.rpc.get_connection_status.return_value = status()
        robot.rpc.get_gripper_async_status.return_value = grip_status()
        robot.rpc.create_action_dict.return_value = {"joint_position": [.001] * 7}
        robot.rpc.move_to_joint_positions.return_value = True
        return robot

    def run_action(self, robot, a=None, observed=None, deadline=None, guard=0.):
        return robot.execute(a or h.parse_action(action()), observed or state(), status(),
                             (np.array([.3, -.1, .3]), np.array([.4, .1, .4])), False, 1,
                             deadline if deadline is not None else time.monotonic() + 10,
                             max_joint_step=guard)

    def test_stale_state_and_deadline_do_not_dispatch(self):
        for stale in (True, False):
            r = self.robot(); observed = state()
            if stale:
                observed["cartesian_position"][0] += .01
            with self.assertRaises(RuntimeError):
                self.run_action(r, observed=observed, deadline=0 if not stale else None)
            r.rpc.move_to_joint_positions.assert_not_called()

    def test_checked_backend_fault_is_propagated_without_legacy_fallback(self):
        r = self.robot(); st = status(); st['checked_motion_supported'] = True
        r.rpc.get_connection_status.return_value = st
        r.rpc.move_to_joint_positions_checked.side_effect = RuntimeError('control fault')
        with self.assertRaisesRegex(RuntimeError, 'control fault'):
            self.run_action(r)
        r.rpc.move_to_joint_positions_checked.assert_called_once()
        r.rpc.move_to_joint_positions.assert_not_called()

    def test_invalid_ik_does_not_dispatch(self):
        r = self.robot(); r.rpc.create_action_dict.return_value = {"joint_position": [float("nan")] * 7}
        with self.assertRaises(ValueError):
            self.run_action(r, guard=.1)
        r.rpc.move_to_joint_positions.assert_not_called()

    def test_submission_error_not_retried(self):
        r = self.robot(); r.rpc.move_to_joint_positions.side_effect = RuntimeError("lost")
        with self.assertRaisesRegex(RuntimeError, "lost"):
            self.run_action(r)
        r.rpc.move_to_joint_positions.assert_called_once()
        r.rpc.recover_robot.assert_not_called()

    def test_settled_action_uses_low_speed_and_no_gripper_command(self):
        r = self.robot()
        after = state(); after["cartesian_position"][0] += .001; after["joint_positions"] = [.001] * 7
        r.rpc.get_robot_state.side_effect = [(state(), {}), (state(), {})] + [(after, {})] * 10
        result = self.run_action(r)
        self.assertEqual(result["result"], "settled")
        r.rpc.move_to_joint_positions.assert_called_once_with([.001] * 7, .01)
        r.rpc.update_gripper.assert_not_called()

    def test_joint_guard_and_speed_factor_are_configurable(self):
        # A 0.15 rad IK increment is dispatched by default: the guard is off unless asked for.
        r = self.robot(); r.rpc.create_action_dict.return_value = {"joint_position": [.15] * 7}
        after = state(); after["cartesian_position"][0] += .001; after["joint_positions"] = [.15] * 7
        r.rpc.get_robot_state.side_effect = [(state(), {}), (state(), {})] + [(after, {})] * 10
        result = r.execute(h.parse_action(action()), state(), status(),
                           (np.array([.3, -.1, .3]), np.array([.4, .1, .4])), False, 1,
                           time.monotonic() + 10, speed_factor=.02)
        self.assertEqual(result["result"], "settled")
        r.rpc.move_to_joint_positions.assert_called_once_with([.15] * 7, .02)
        r = self.robot(); r.rpc.create_action_dict.return_value = {"joint_position": [.15] * 7}
        after = state(); after["cartesian_position"][0] += .001 * (2/3)
        r.rpc.get_robot_state.side_effect = [(state(), {})] + [(after, {})] * 20
        result = self.run_action(r, guard=.1)
        self.assertAlmostEqual(result["joint_step_scale"], 2/3)
        np.testing.assert_allclose(r.rpc.move_to_joint_positions.call_args.args[0], [.1] * 7)

    def test_recovery_before_or_during_motion_aborts(self):
        changed = status(); changed["recover_count"] = 1
        for statuses in ([changed], [status(), changed]):
            r = self.robot(); r.rpc.get_connection_status.side_effect = statuses
            with self.assertRaisesRegex(RuntimeError, "recover_count"):
                self.run_action(r)

    def test_no_motion_and_large_sideways_error_never_count_as_arrived(self):
        for delta in ([0., 0., 0.], [.001, .04, 0.]):
            r = self.robot()
            after = state(); after["cartesian_position"] = (np.asarray(after["cartesian_position"]) + np.array(list(delta) + [0, 0, 0])).tolist()
            r.rpc.get_robot_state.side_effect = [(state(), {})] + [(after, {})] * 100
            with self.assertRaisesRegex(RuntimeError, "Arrival timeout"):
                r.execute(h.parse_action(action(eef_delta_m=[.004, 0, 0])), state(), status(),
                          None, False, .02, time.monotonic() + 10)
            r.rpc.move_to_joint_positions.assert_called_once()

    def test_millimetre_tracking_error_with_real_progress_is_accepted(self):
        r = self.robot()
        after = state(); after["cartesian_position"][0] -= .0005
        after["cartesian_position"][1] += .00485
        after["cartesian_position"][2] -= .00095
        r.rpc.get_robot_state.side_effect = [(state(), {})] + [(after, {})] * 20
        result = r.execute(h.parse_action(action(eef_delta_m=[0, .005, 0])), state(), status(),
                           None, False, 1, time.monotonic() + 10)
        self.assertEqual(result["result"], "settled")
        r.rpc.move_to_joint_positions.assert_called_once()

    def test_small_ik_deviation_requests_reobservation(self):
        for wanted, actual in (([.01, 0, .002], [.0067, .0004, -.00022]),
                               ([0, .005, 0], [-.00073, .00487, -.00198]),
                               ([0, 0, .02], [.00356, 0, .01358])):
            r = self.robot(); after = state()
            after["cartesian_position"][:3] = (np.array(after["cartesian_position"][:3]) + actual).tolist()
            r.rpc.get_robot_state.side_effect = [(state(), {})] + [(after, {})] * 30
            result = r.execute(h.parse_action(action(eef_delta_m=wanted)), state(), status(),
                               None, False, 1, time.monotonic()+10)
            self.assertEqual(result["result"], "reobserve")
            r.rpc.move_to_joint_positions.assert_called_once()

    def test_release_requires_actual_target_and_explicit_rpc(self):
        for actual, succeeds in ((.02, True), (.4, False), (.78, False)):
            r = self.robot(); before = state(); before["gripper_position"] = .75
            after = state(); after["gripper_position"] = actual
            r.rpc.get_robot_state.side_effect = [(before, {})] + [(after, {})] * 100
            gs = grip_status(); gs["explicit_release_supported"] = True
            sent = dict(gs, gripper_async_submit_count=1)
            r.rpc.get_gripper_async_status.side_effect = [gs] + [sent] * 100
            a = h.Action(np.zeros(3), -.73, False, "release", "release")
            if succeeds:
                result = r.execute(a,before,status(),None,True,1,time.monotonic()+10)
                self.assertEqual(result["result"], "settled")
            else:
                with self.assertRaisesRegex(RuntimeError, "Arrival timeout"):
                    r.execute(a,before,status(),None,True,.02,time.monotonic()+10)
            r.rpc.release_gripper.assert_called_once()
            r.rpc.update_gripper.assert_not_called()
            r.rpc.move_to_joint_positions.assert_not_called()

    def test_legacy_backend_release_fails_before_dispatch(self):
        r = self.robot()
        a = h.Action(np.zeros(3), -.05, False, "release", "release")
        with self.assertRaisesRegex(RuntimeError, "explicit release support"):
            r.execute(a,state(),status(),None,True,1,time.monotonic()+10)
        r.rpc.update_gripper.assert_not_called()
        r.rpc.release_gripper.assert_not_called()

    def test_action_without_reason_is_accepted_and_not_emitted(self):
        obj = json.loads(action()); obj.pop("reason")
        a = h.parse_action(json.dumps(obj))
        self.assertNotIn("reason", a.as_dict())

    def test_scaling_preserves_joint_direction_and_measured_progress(self):
        r = self.robot()
        qdelta = np.array([.10, -.05, .02, 0, -.03, .01, 0])
        r.rpc.create_action_dict.return_value = {"joint_position": qdelta.tolist()}
        after = state(); after["cartesian_position"][0] += .005
        r.rpc.get_robot_state.side_effect = [(state(), {})] + [(after, {})] * 20
        result = r.execute(h.parse_action(action(eef_delta_m=[.01,0,0])), state(), status(),
                           None, False, 1, time.monotonic()+10, max_joint_step=.05)
        self.assertEqual(result["joint_step_scale"], .5)
        np.testing.assert_allclose(r.rpc.move_to_joint_positions.call_args.args[0], qdelta*.5)
        self.assertEqual(result["result"], "settled")

    def test_reason_and_phase_are_retained(self):
        a = h.parse_action(action(reason="banana still on table", phase="lift"))
        self.assertEqual(a.as_dict()["reason"], "banana still on table")
        self.assertEqual(a.as_dict()["phase"], "lift")
        for value in (2, [], None):
            with self.assertRaises(ValueError):
                h.parse_action(action(reason=value))

    def test_history_pairs_before_and_after_without_old_state_torques(self):
        history = [dict(action={}, state=state(), wrist_yellow_uv_before=[.4, .5])]
        h.attach_after_observation(history, state(), [.5, .6])
        item = h.compact_history(history)[0]
        self.assertEqual(item["wrist_yellow_uv_before"], [.4, .5])
        self.assertEqual(item["wrist_yellow_uv_after"], [.5, .6])
        self.assertTrue(item["after_observation_available"])
        self.assertNotIn("joint_positions", item["state"])
        moved = state(); moved["cartesian_position"][0] += .1
        h.attach_after_observation(history, moved, [.9, .9])
        self.assertEqual(history[0]["wrist_yellow_uv_after"], [.5, .6])

    def test_gripper_delta_uses_absolute_target_without_arm_motion(self):
        r = self.robot()
        after = state(); after["gripper_position"] = .13
        r.rpc.get_robot_state.side_effect = [(state(), {})] + [(after, {})] * 10
        after_grip = grip_status(); after_grip["gripper_async_submit_count"] = 1
        r.rpc.get_gripper_async_status.side_effect = [grip_status()] + [after_grip] * 10
        r.rpc.update_gripper.return_value = True
        a = h.parse_action(action(eef_delta_m=[0, 0, 0], gripper_delta=.03))
        result = r.execute(a, state(), status(), None, True, 2, time.monotonic() + 10)
        self.assertEqual(result["result"], "settled")
        r.rpc.update_gripper.assert_called_once_with(.13, False, False)
        r.rpc.move_to_joint_positions.assert_not_called()

    def test_busy_gripper_blocks_arm_dispatch(self):
        r = self.robot(); gs = grip_status(); gs["gripper_async_busy"] = True
        r.rpc.get_gripper_async_status.return_value = gs
        with self.assertRaisesRegex(RuntimeError, "gripper operation"):
            self.run_action(r)
        r.rpc.move_to_joint_positions.assert_not_called()

    def test_model_request_has_three_images_and_no_redirects(self):
        client = h.VisionClient("https://example.invalid/v1", "vision-model", "secret", 5)
        response = Mock(); response.__enter__ = Mock(return_value=response); response.__exit__ = Mock()
        response.read.return_value = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": action()}}]}).encode()
        opener = Mock(); opener.open.return_value = response
        images = [dict(role=r, serial=str(i), url="data:image/jpeg;base64,YQ==") for i, r in enumerate(("wrist", "exterior_1", "exterior_2"))]
        with patch.object(h.vision.urllib.request, "build_opener", return_value=opener) as builder:
            h.parse_action(client.decide("test", state(), images, [], {}))
        request = opener.open.call_args.args[0]
        body = json.loads(request.data)
        self.assertNotIn("thinking", body)
        self.assertEqual(sum(x["type"] == "image_url" for x in body["messages"][1]["content"]), 3)
        with self.assertRaisesRegex(RuntimeError, "redirect"):
            builder.call_args.args[0].redirect_request()

    def test_thinking_switch_follows_the_provider(self):
        self.assertEqual(h.thinking_body("auto", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                         {"enable_thinking": False})
        self.assertEqual(h.thinking_body("auto", "https://api.deepseek.com"), {"thinking": {"type": "disabled"}})
        self.assertEqual(h.thinking_body("auto", "https://example.invalid"), {})
        self.assertEqual(h.thinking_body("none", "https://dashscope.aliyuncs.com"), {})
        self.assertEqual(h.thinking_body("deepseek", "https://dashscope.aliyuncs.com"), {"thinking": {"type": "disabled"}})

    def test_reject_insecure_endpoint(self):
        for url in ("http://example.invalid", "https://user:pass@example.invalid", "https://", "https://example.invalid?key=secret"):
            with self.assertRaises(ValueError):
                h.VisionClient(url, "m", "k", 1)
        # Plaintext is allowed only when the operator asks for it.
        self.assertEqual(h.VisionClient("http://10.0.0.5:8317/v1", "m", "k", 1, allow_http=True).url,
                         "http://10.0.0.5:8317/v1/chat/completions")

    def test_default_dry_run_complete_loop_never_dispatches(self):
        with tempfile.TemporaryDirectory() as directory:
            robot, cameras, client = Mock(), Mock(), Mock()
            robot.observe.return_value = (state(), status(), {"robot_timestamp_seconds": 1})
            cameras.capture.side_effect = lambda folder: [dict(role=str(i), serial=str(i), path="fake.jpg",
                captured_monotonic=time.monotonic(), url="data:image/jpeg;base64,YQ==") for i in range(3)]
            client.decide.return_value = action(eef_delta_m=[.02, 0, 0])
            argv = ["harness", "--task", "test", "--base-url", "https://example.invalid/v1", "--model", "vision-test", "--vision-confirmed",
                    "--max-steps", "1", "--log-dir", directory]
            with patch.object(sys, "argv", argv), patch.dict(L.os.environ, {"VLM_API_KEY": "test-key", "VLM_NUC": "127.0.0.1", "VLM_CAMERAS": "1 2 3"}), \
                 patch.object(L, "DroidRobot", return_value=robot), patch.object(L, "ThreeCameras", return_value=cameras), \
                 patch.object(L, "VisionClient", return_value=client), patch.object(L.Path, "home", return_value=Path(directory)):
                L.main()
            robot.execute.assert_not_called()
            robot.close.assert_called_once()
            cameras.close.assert_called_once()
            events = [json.loads(line) for line in next(Path(directory).glob("*/events.jsonl")).read_text().splitlines()]
            self.assertIn("dry_run", [e["event"] for e in events])
            self.assertEqual(events[-1]["event"], "budget_stop")

    def test_three_bad_model_answers_are_followed_by_fresh_replan(self):
        with tempfile.TemporaryDirectory() as directory:
            robot, cameras, client = Mock(), Mock(), Mock()
            robot.observe.return_value = (state(), status(), {})
            cameras.capture.side_effect = lambda folder: [dict(role=str(i), serial=str(i), path="fake.jpg",
                captured_monotonic=time.monotonic(), url="data:image/jpeg;base64,YQ==") for i in range(3)]
            client.decide.side_effect = ["invalid", "invalid", "invalid", action()]
            argv = ["harness", "--task", "test", "--base-url", "https://example.invalid/v1", "--model", "test", "--vision-confirmed",
                    "--max-steps", "4", "--log-dir", directory]
            with patch.object(sys, "argv", argv), patch.dict(L.os.environ, {"VLM_API_KEY": "test", "VLM_NUC": "127.0.0.1", "VLM_CAMERAS": "1 2 3"}), \
                 patch.object(L, "DroidRobot", return_value=robot), patch.object(L, "ThreeCameras", return_value=cameras), \
                 patch.object(L, "VisionClient", return_value=client), patch.object(L.Path, "home", return_value=Path(directory)):
                L.main()
            self.assertEqual(client.decide.call_count, 4)
            self.assertEqual(cameras.capture.call_count, 4)
            robot.execute.assert_not_called()
            events = [json.loads(line) for path in Path(directory).glob("*/events.jsonl") for line in path.read_text().splitlines()]
            self.assertEqual(sum(e['event']=='action_rejected' for e in events),3)
            self.assertTrue(any(e['event']=='dry_run' for e in events))

    def test_large_action_is_served_by_sub_steps(self):
        """One 6 cm action must become repeated dispatches of at most 2 cm."""
        with tempfile.TemporaryDirectory() as directory:
            robot, cameras, client = Mock(), Mock(), Mock()
            pose = {"cartesian_position": [.3, 0., .3, 0., 0., 0.]}

            def current_state():
                s = state(); s["cartesian_position"] = list(pose["cartesian_position"]); return s

            def execute(action, observed, observed_status, workspace, allow_gripper, timeout, deadline, **kw):
                pose["cartesian_position"] = [a + b for a, b in zip(pose["cartesian_position"], list(action.delta) + [0, 0, 0])]
                return dict(state=current_state(), controller_status=status(),
                            observed_delta_m=list(action.delta), result="settled")

            robot.observe.side_effect = lambda: (current_state(), status(), {"robot_timestamp_seconds": 1})
            robot.execute.side_effect = execute
            cameras.capture.side_effect = lambda folder: [dict(role=str(i), serial=str(i), path="fake.jpg",
                captured_monotonic=time.monotonic(), url="data:image/jpeg;base64,YQ==") for i in range(3)]
            client.decide.return_value = action(eef_delta_m=[.06, 0, 0])
            argv = ["harness", "--task", "test", "--base-url", "https://example.invalid/v1", "--model", "vision-test", "--vision-confirmed", "--execute",
                    "--max-steps", "1", "--sub-step-m", "0.02", "--max-sub-steps", "10", "--log-dir", directory]
            with patch.object(sys, "argv", argv), patch.dict(L.os.environ, {"VLM_API_KEY": "test-key", "VLM_NUC": "127.0.0.1", "VLM_CAMERAS": "1 2 3"}), \
                 patch.object(L, "DroidRobot", return_value=robot), patch.object(L, "ThreeCameras", return_value=cameras), \
                 patch.object(L, "VisionClient", return_value=client), patch.object(L.Path, "home", return_value=Path(directory)):
                L.main()
            wanted = [call.args[0].delta for call in robot.execute.call_args_list]
            self.assertEqual(len(wanted), 3)
            for step in wanted:
                self.assertLessEqual(float(np.linalg.norm(step)), .02 + 1e-9)
            self.assertAlmostEqual(sum(float(np.linalg.norm(s)) for s in wanted), .06, places=6)

    def test_reobserve_interrupts_substeps_and_is_logged(self):
        """One 6 cm action must become repeated dispatches of at most 2 cm."""
        with tempfile.TemporaryDirectory() as directory:
            robot, cameras, client = Mock(), Mock(), Mock()
            pose = {"cartesian_position": [.3, 0., .3, 0., 0., 0.]}

            def current_state():
                s = state(); s["cartesian_position"] = list(pose["cartesian_position"]); return s

            def execute(action, observed, observed_status, workspace, allow_gripper, timeout, deadline, **kw):
                pose["cartesian_position"] = [a + b for a, b in zip(pose["cartesian_position"], list(action.delta) + [0, 0, 0])]
                return dict(state=current_state(), controller_status=status(),
                            observed_delta_m=list(action.delta), result="reobserve")

            robot.observe.side_effect = lambda: (current_state(), status(), {"robot_timestamp_seconds": 1})
            robot.execute.side_effect = execute
            cameras.capture.side_effect = lambda folder: [dict(role=str(i), serial=str(i), path="fake.jpg",
                captured_monotonic=time.monotonic(), url="data:image/jpeg;base64,YQ==") for i in range(3)]
            client.decide.return_value = action(eef_delta_m=[.06, 0, 0])
            argv = ["harness", "--task", "test", "--base-url", "https://example.invalid/v1", "--model", "vision-test", "--vision-confirmed", "--execute",
                    "--max-steps", "1", "--sub-step-m", "0.02", "--max-sub-steps", "10", "--log-dir", directory]
            with patch.object(sys, "argv", argv), patch.dict(L.os.environ, {"VLM_API_KEY": "test-key", "VLM_NUC": "127.0.0.1", "VLM_CAMERAS": "1 2 3"}), \
                 patch.object(L, "DroidRobot", return_value=robot), patch.object(L, "ThreeCameras", return_value=cameras), \
                 patch.object(L, "VisionClient", return_value=client), patch.object(L.Path, "home", return_value=Path(directory)):
                L.main()
            wanted = [call.args[0].delta for call in robot.execute.call_args_list]
            self.assertEqual(len(wanted), 1)
            for step in wanted:
                self.assertLessEqual(float(np.linalg.norm(step)), .02 + 1e-9)
            self.assertAlmostEqual(sum(float(np.linalg.norm(s)) for s in wanted), .02, places=6)

            events = [json.loads(line) for path in Path(directory).glob("*/events.jsonl") for line in path.read_text().splitlines()]
            results = [e for e in events if e["event"] == "result"]
            self.assertEqual(results[0]["result"], "reobserve")

    def test_four_mm_action_reaches_execute_once(self):
        """A 4 mm request must actually reach the controller."""
        with tempfile.TemporaryDirectory() as directory:
            robot, cameras, client = Mock(), Mock(), Mock()
            pose = {"cartesian_position": [.3, 0., .3, 0., 0., 0.]}

            def current_state():
                s = state(); s["cartesian_position"] = list(pose["cartesian_position"]); return s

            def execute(action, observed, observed_status, workspace, allow_gripper, timeout, deadline, **kw):
                pose["cartesian_position"] = [a + b for a, b in zip(pose["cartesian_position"], list(action.delta) + [0, 0, 0])]
                return dict(state=current_state(), controller_status=status(),
                            observed_delta_m=list(action.delta), result="settled")

            robot.observe.side_effect = lambda: (current_state(), status(), {"robot_timestamp_seconds": 1})
            robot.execute.side_effect = execute
            cameras.capture.side_effect = lambda folder: [dict(role=str(i), serial=str(i), path="fake.jpg",
                captured_monotonic=time.monotonic(), url="data:image/jpeg;base64,YQ==") for i in range(3)]
            client.decide.return_value = action(eef_delta_m=[.004, 0, 0])
            argv = ["harness", "--task", "test", "--base-url", "https://example.invalid/v1", "--model", "vision-test", "--vision-confirmed", "--execute",
                    "--max-steps", "1", "--sub-step-m", "0.02", "--max-sub-steps", "10", "--log-dir", directory]
            with patch.object(sys, "argv", argv), patch.dict(L.os.environ, {"VLM_API_KEY": "test-key", "VLM_NUC": "127.0.0.1", "VLM_CAMERAS": "1 2 3"}), \
                 patch.object(L, "DroidRobot", return_value=robot), patch.object(L, "ThreeCameras", return_value=cameras), \
                 patch.object(L, "VisionClient", return_value=client), patch.object(L.Path, "home", return_value=Path(directory)):
                L.main()
            wanted = [call.args[0].delta for call in robot.execute.call_args_list]
            self.assertEqual(len(wanted), 1)
            for step in wanted:
                self.assertLessEqual(float(np.linalg.norm(step)), .02 + 1e-9)
            self.assertAlmostEqual(sum(float(np.linalg.norm(s)) for s in wanted), .004, places=6)

    def test_no_vision_config_does_not_connect_hardware(self):
        with patch.object(sys, "argv", ["harness", "--task", "test"]), \
             patch.dict(L.os.environ, {}, clear=True), patch.object(L, "DroidRobot") as robot:
            with self.assertRaises(SystemExit):
                L.main()
            robot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
