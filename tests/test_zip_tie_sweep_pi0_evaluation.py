"""Check sweep success boundaries, action clipping and trace alignment."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/deploy"))
import deploy_zip_tie_sweep_pi0 as deploy


class FakeArm:
    def __init__(self, home_error=.25, collected_count=2, sponge_returned=True,
                 camera_height=.02, table_contact=False):
        self.home_error = home_error
        self.collected_count = collected_count
        self.sponge_returned = sponge_returned
        self.table_contact = table_contact
        self.limits = np.tile([-1., 1.], (6, 1))
        self.model = SimpleNamespace(camera=lambda name: SimpleNamespace(id=0))
        self.data = SimpleNamespace(
            body=lambda name: SimpleNamespace(xpos=np.array([0., 0., 0.])),
            cam_xpos=np.array([[0., 0., camera_height]]))
        self.mj = SimpleNamespace(mjtState=SimpleNamespace(mjSTATE_INTEGRATION=1),
                                  mj_stateSize=lambda m, s: 1,
                                  mj_getState=lambda m, d, value, s: value.fill(self.t))

    def reset(self, rng, fixed_eval):
        self.t = -deploy.FPS
        self.last_contact = {"robot_table_contact": False}
        return np.r_[np.zeros(6), .101]

    def state(self):
        return np.r_[np.full(6, self.home_error if self.t > 0 else 0.), .101]

    def step(self, action):
        self.t += 1
        self.last_contact["robot_table_contact"] = self.table_contact and self.t > 0

    def finger_contacts(self):
        return 2 if self.t == 1 else 0

    def result_metrics(self):
        return {"collected_count": self.collected_count,
                "sponge_returned": self.sponge_returned and self.t >= 2,
                "home_joint_max_error_rad": self.home_error}


class SweepEvaluationTests(unittest.TestCase):
    def run_fake(self, arm=None, max_steps=26, policy=None):
        renderer = SimpleNamespace(update_scene=lambda *a, **k: None,
                                   render=lambda: np.zeros((2, 2, 3), dtype=np.uint8))
        if policy is None:
            policy = SimpleNamespace(infer=lambda obs: {
                "actions": np.tile(np.r_[np.full(6, 2.), .2], (8, 1))})
        self.timings = {"policy_inference_time_ms": []}
        return deploy.run_episode(arm or FakeArm(), policy, renderer, renderer, None,
                                  np.random.default_rng(42), True, max_steps, 8,
                                  timings=self.timings)

    def test_home_boundary_success_and_trace_alignment(self):
        result, states, actions, ties, sponge, snapshots, measured, raw, requests = self.run_fake()
        self.assertTrue(result["success"])
        self.assertTrue(result["returned_home"])
        self.assertEqual(result["stable_success_frames"], 25)
        self.assertEqual(snapshots.shape, (27, 1))
        self.assertEqual(measured.shape, (27, 7))
        self.assertEqual(ties.shape, (26, 2, 3))
        self.assertEqual(sponge.shape, (26, 3))
        np.testing.assert_allclose(states, measured[:-1], atol=1e-8)
        np.testing.assert_array_equal(raw[:, :6], 2.)
        np.testing.assert_array_equal(actions[:, :6], 1.)
        np.testing.assert_allclose(actions[:, 6], .101)
        np.testing.assert_array_equal(requests, np.arange(26) // 8)
        self.assertEqual(len(self.timings["policy_inference_time_ms"]), 4)

    def test_home_error_over_boundary_prevents_success(self):
        result = self.run_fake(FakeArm(home_error=.251))[0]
        self.assertFalse(result["success"])
        self.assertFalse(result["returned_home"])
        self.assertEqual(result["reason"], "timeout")

    def test_other_sweep_requirements_still_prevent_success(self):
        for values in ({"collected_count": 1}, {"sponge_returned": False},
                       {"camera_height": .004}):
            with self.subTest(values=values):
                self.assertFalse(self.run_fake(FakeArm(**values))[0]["success"])
        result = self.run_fake(FakeArm(table_contact=True))[0]
        self.assertEqual(result["steps"], 1)
        self.assertEqual(result["reason"], "Robot arm touched the tabletop")

    def test_hold_one_frame_short_times_out(self):
        result = self.run_fake(max_steps=25)[0]
        self.assertFalse(result["success"])
        self.assertEqual(result["stable_success_frames"], 24)

    def test_policy_error_preserves_initial_frame_and_empty_actions(self):
        def infer(obs):
            raise OSError("connection lost")
        result, states, actions, _, _, snapshots, measured, raw, requests = self.run_fake(
            policy=SimpleNamespace(infer=infer))
        self.assertEqual(result["reason"], "connection lost")
        self.assertEqual(states.shape, (0, 7))
        self.assertEqual(actions.shape, (0, 7))
        self.assertEqual(raw.shape, (0, 7))
        self.assertEqual(requests.shape, (0,))
        self.assertEqual(snapshots.shape, (1, 1))
        self.assertEqual(measured.shape, (1, 7))

    def test_summary_counts_failures_and_excludes_viewer_closure(self):
        rows = [{"episode": i, "success": reason == "success", "reason": reason,
                 "robot_table_contact": reason == "Robot arm touched the tabletop"}
                for i, reason in enumerate(["success", "timeout", "connection lost",
                                            "Robot arm touched the tabletop", "Viewer closed"])]
        with tempfile.TemporaryDirectory() as temp:
            deploy.write_summary(Path(temp), rows, 5)
            summary = json.loads((Path(temp) / "summary.json").read_text())
        self.assertEqual(summary["success_rate"], .25)
        self.assertEqual(summary["timeouts"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["robot_table_contact_episodes"], 1)
        self.assertEqual(summary["interrupted_episodes"], 1)

    def test_step_limit_rejected_before_model_or_connection(self):
        with patch.object(sys, "argv", ["deploy", "--max_steps", "5001"]), \
                patch.object(deploy, "CleanupArm") as arm, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                deploy.main()
            self.assertEqual(error.exception.code, 2)
            arm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
