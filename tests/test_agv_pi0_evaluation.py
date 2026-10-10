"""Check success/timeout boundaries and recorded episode alignment."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/deploy"))
import deploy_agv_dummyx_pi0 as deploy


class FakeArm:
    def __init__(self, return_at=1, invalid_at=None, home_offset=0.):
        self.return_at = return_at
        self.invalid_at = invalid_at
        self.home_offset = home_offset
        self.limits = np.tile([-3., 3.], (6, 1))
        self.model = None
        self.data = SimpleNamespace(body=lambda name: SimpleNamespace(xpos=np.array([0., 0., self.z])))
        self.mj = SimpleNamespace(mjtState=SimpleNamespace(mjSTATE_INTEGRATION=1),
                                  mj_stateSize=lambda m, s: 1,
                                  mj_getState=lambda m, d, value, s: value.fill(self.t))
        self.reset()

    def reset(self, *args):
        self.t, self.z = -deploy.FPS, 0.
        return np.r_[np.zeros(6), .101]

    def state(self):
        q = 0. if self.t < 1 else (self.home_offset if self.t >= self.return_at else deploy.HOME_TOLERANCE_RAD + .01)
        return np.r_[np.full(6, q), .101]

    def step(self, action):
        self.t += 1
        self.z = .1 if self.t > 0 else 0.

    def finger_contacts(self):
        return 2 if self.t == 1 else 0

    def placement_metrics(self):
        return {"inside_bin": self.t != self.invalid_at, "height_ok": True,
                "bin_contact": True, "object_speed_m_s": 0., "finger_contacts": 0,
                "gripper_width_m": .101}


class EvaluationTests(unittest.TestCase):
    def run_fake(self, arm, max_steps):
        renderer = SimpleNamespace(update_scene=lambda *a, **k: None,
                                   render=lambda: np.zeros((2, 2, 3), dtype=np.uint8))
        self.policy_calls = 0

        def infer(obs):
            self.policy_calls += 1
            return {"actions": np.tile(np.r_[np.zeros(6), .101], (8, 1))}

        policy = SimpleNamespace(infer=infer)
        with patch.object(deploy, "load_spawn_region", return_value={}):
            return deploy.run_episode(arm, policy, renderer, renderer, None,
                                      np.random.default_rng(42), True, max_steps, 8)

    def test_placement_without_home_times_out(self):
        result = self.run_fake(FakeArm(return_at=4000), 3000)[0]
        self.assertEqual(result["reason"], "timeout")
        self.assertEqual(result["steps"], 3000)
        self.assertFalse(result["success"])
        self.assertFalse(result["returned_home"])
        self.assertEqual(result["retention_frames"], 0)
        self.assertEqual(self.policy_calls, 375)

    def test_success_on_last_allowed_step_and_trace_alignment(self):
        values = self.run_fake(FakeArm(return_at=26), 50)
        result, states, actions, objects, snapshots, measured, raw, requests = values
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 50)
        self.assertEqual(result["retention_frames"], 25)
        self.assertEqual(snapshots.shape, (51, 1))
        self.assertEqual(measured.shape, (51, 7))
        self.assertEqual(actions.shape, (50, 7))
        self.assertEqual(objects.shape, (50, 3))
        np.testing.assert_allclose(states, measured[:-1], atol=1e-8)
        self.assertTrue(np.isfinite(raw).all())
        np.testing.assert_array_equal(actions, raw)
        self.assertTrue((requests >= 0).all())
        np.testing.assert_array_equal(requests, np.arange(50) // 8)
        self.assertEqual(self.policy_calls, 7)

    def test_hold_resets_when_placement_is_lost(self):
        result = self.run_fake(FakeArm(invalid_at=24), 48)[0]
        self.assertEqual(result["reason"], "timeout")
        self.assertEqual(result["retention_frames"], 24)

    def test_placement_and_home_require_half_second_hold(self):
        result = self.run_fake(FakeArm(home_offset=.075), 25)[0]
        self.assertTrue(result["success"])
        self.assertTrue(result["returned_home"])
        self.assertEqual(result["retention_frames"], 25)

    def test_home_error_prevents_success(self):
        result = self.run_fake(FakeArm(home_offset=.201), 25)[0]
        self.assertEqual(result["reason"], "timeout")
        self.assertFalse(result["success"])
        self.assertFalse(result["returned_home"])
        self.assertEqual(result["retention_frames"], 0)

    def test_relaxed_home_tolerance_accepts_boundary(self):
        for offset in (.15, .20):
            with self.subTest(offset=offset):
                result = self.run_fake(FakeArm(home_offset=offset), 25)[0]
                self.assertTrue(result["success"])
                self.assertTrue(result["returned_home"])
                self.assertEqual(result["retention_frames"], 25)

    def test_one_joint_over_home_tolerance_prevents_success(self):
        result = self.run_fake(FakeArm(home_offset=np.r_[np.zeros(5), .201]), 25)[0]
        self.assertEqual(result["reason"], "timeout")
        self.assertFalse(result["success"])
        self.assertFalse(result["returned_home"])
        self.assertEqual(result["retention_frames"], 0)

    def test_one_frame_short_of_home_hold_times_out(self):
        result = self.run_fake(FakeArm(), 24)[0]
        self.assertEqual(result["reason"], "timeout")
        self.assertEqual(result["retention_frames"], 24)

    def test_summary_counts_errors_and_excludes_manual_interruptions(self):
        rows = [{"episode": i, "success": reason == "success", "reason": reason}
                for i, reason in enumerate(["success", "timeout", "server error", "Viewer closed"])]
        with tempfile.TemporaryDirectory() as temp:
            deploy.write_summary(Path(temp), rows, 4)
            result = json.loads((Path(temp) / "summary.json").read_text())
            self.assertEqual(result["success_rate"], 1 / 3)
            self.assertEqual(result["timeouts"], 1)
            self.assertEqual(result["interrupted_episodes"], 1)

    def test_perception_failure_stops_before_policy_action(self):
        arm = FakeArm()
        renderer = SimpleNamespace(update_scene=lambda *a, **k: None,
                                   render=lambda: np.zeros((2, 2, 3), dtype=np.uint8))
        policy = SimpleNamespace(infer=lambda obs: self.fail("Policy must not run after perception failure"))
        def fail_perception(arm, evidence_dir):
            raise ValueError("no valid obstacle")
        with patch.object(deploy, "load_spawn_region", return_value={}):
            result, _, actions, *_ = deploy.run_episode(
                arm, policy, renderer, renderer, None, np.random.default_rng(42),
                True, 8, 8, perception=fail_perception, perception_dir=Path("unused"))
        self.assertEqual(result["steps"], 0)
        self.assertEqual(actions.shape, (0, 7))
        self.assertIn("Perception failed", result["reason"])
        self.assertEqual(arm.t, 0)  # The reset hold completed; no policy action followed.


class LegacyAPFParityTests(unittest.TestCase):
    """Compare mapped AGV targets to the original visual APF control law."""
    def setUp(self):
        import deploy_screwdriver_client as legacy
        self.legacy = legacy
        self.names = {"link6_1": 1, "link6": 1, "link7": 2,
                      "real_screwdriver": 3, "dynamic_pillar": 4, "link0": 5}
        self.q = np.array([.02, -.03, .01, .04, -.02, .05])
        self.jacobian = np.array([[.3, -.1, .5, .15, -.12, .02],
                                 [.08, .4, -.1, .12, .16, -.03],
                                 [.1, -.14, .03, .2, .1, .3]])
        self.columns = np.array([8, 1, 11, 4, 10, 6])
        self.old_model = self.make_model(np.arange(6))
        self.model = self.make_model(self.columns)
        self.data = SimpleNamespace(qpos=np.r_[self.q, .01, .01],
                                    site_xpos=np.array([[.30, -.12, .40]]),
                                    xpos=np.zeros((6, 3)),
                                    xmat=np.tile(np.eye(3).ravel(), (6, 1)))
        self.data.xpos[1] = [.30, -.12, .48]
        self.data.xpos[2] = self.data.site_xpos[0]
        self.data.xpos[3] = [.30, -.12, .39]
        self.data.xpos[4] = [.42, -.12, .40]
        self.data.body = lambda name: SimpleNamespace(
            xpos=self.data.xpos[self.names[name]], xmat=self.data.xmat[self.names[name]])
        self.arm = SimpleNamespace(model=self.model, data=self.data, mj=legacy.mujoco,
                                   dadr=self.columns, state=lambda: np.r_[self.q, .0],
                                   limits=np.tile([-10., 10.], (6, 1)),
                                   apf_last_q_dot=np.zeros(6), target_body=3)
        self.cap = {"p1": np.array([.40, -.12, .20]),
                    "p2": np.array([.40, -.12, .60]), "r": .025,
                    "placement_goal": np.array([0., -.1, .24])}
        self.action = np.r_[self.q + [.12, -.03, .08, -.02, -.01, .01], .01]
        legacy.last_q_dot = np.zeros(6)
        self.addCleanup(setattr, legacy, "last_q_dot", np.zeros(6))
        name_patch = patch.object(legacy.mujoco, "mj_name2id", side_effect=self.name_id)
        jac_patch = patch.object(legacy.mujoco, "mj_jacBody", side_effect=self.body_jacobian)
        name_patch.start()
        jac_patch.start()
        self.addCleanup(name_patch.stop)
        self.addCleanup(jac_patch.stop)

    def make_model(self, columns):
        return SimpleNamespace(nv=int(max(columns)) + 5, parity_columns=np.asarray(columns),
                               site=lambda name: SimpleNamespace(id=0),
                               body=lambda name: SimpleNamespace(id=self.names[name]),
                               site_bodyid=np.array([2]))

    def name_id(self, model, kind, name):
        return 0 if name == "tcp_site" else self.names.get(name, -1)

    def body_jacobian(self, model, data, position, rotation, body_id):
        self.assertEqual(body_id, 2)
        position[:] = .777  # Decoy columns catch accidental use of the first six DOFs.
        position[:, model.parity_columns] = self.jacobian
        if rotation is not None:
            rotation[:] = 0.

    def compare(self, step=30, action=None, cap=None):
        """Keep the original 8 cm envelope as a control-law parity baseline."""
        action = self.action if action is None else action
        cap = self.cap if cap is None else cap
        # The legacy policy uses single-finger units; AGV retains total-opening targets.
        old_action = np.r_[action[:6], action[6] / 2.]
        expected, _, _, active, debug = self.legacy.process_apf_action(
            self.old_model, self.data, old_action, True, step,
            obstacle_source="groundingdino", perceived_capsule=cap)
        with patch.object(deploy, "APF_MAX_DISTANCE_M", .08):
            actual = deploy.process_apf_action(self.arm, action, cap, step, .22, {})
        expected_joints = np.clip(expected[:6], self.arm.limits[:, 0], self.arm.limits[:, 1])
        np.testing.assert_allclose(actual[:6], expected_joints, rtol=0., atol=1e-12)
        np.testing.assert_allclose(self.arm.apf_last_q_dot, debug["q_dot"], rtol=0., atol=1e-12)
        self.assertEqual(actual.shape, (7,))
        self.assertEqual(actual[6], action[6])
        return actual, active

    def test_lift_gate_and_far_obstacle_match_legacy_smoothing(self):
        self.data.xpos[3, 2] = .23
        output, active = self.compare()
        self.assertFalse(active)
        np.testing.assert_allclose(output[:6], self.q + .6 * (self.action[:6] - self.q))
        self.data.xpos[3, 2] = .39
        far = dict(self.cap, p1=np.array([1.5, -.12, .2]), p2=np.array([1.5, -.12, .6]))
        _, active = self.compare(cap=far)
        self.assertFalse(active)

    def test_inward_and_outward_motion_match_original_visual_apf(self):
        _, active = self.compare()
        self.assertTrue(active)
        outward = np.r_[self.q - (self.action[:6] - self.q), self.action[6]]
        _, active = self.compare(action=outward)
        self.assertTrue(active)

    def test_reduced_range_uses_capsule_surface_clearance(self):
        self.assertEqual(deploy.APF_MAX_DISTANCE_M, .05)
        expected_without_avoidance = self.q + .6 * (self.action[:6] - self.q)
        for clearance, expected_active in ((.06, False), (.05, False), (.03, True)):
            with self.subTest(surface_clearance_m=clearance):
                self.arm.apf_last_q_dot[:] = 0.
                obstacle_x = .30 + deploy.ARM_CAPSULE_R + self.cap["r"] + clearance
                cap = dict(self.cap, p1=np.array([obstacle_x, -.12, .20]),
                           p2=np.array([obstacle_x, -.12, .60]))
                with patch.object(self.arm.mj, "mj_jacBody", side_effect=self.body_jacobian) as jacobian:
                    actual = deploy.process_apf_action(self.arm, self.action, cap, 30, .22)
                self.assertEqual(jacobian.called, expected_active)
                if expected_active:
                    self.assertGreater(np.linalg.norm(actual[:6] - expected_without_avoidance), 1e-5)
                else:
                    np.testing.assert_allclose(actual[:6], expected_without_avoidance, rtol=0., atol=1e-12)
                self.assertEqual(actual[6], self.action[6])

    def test_continuous_low_pass_state_and_start_fade_match(self):
        for step, delta in zip((0, 1, 5, 15, 30, 70), (-.1, -.03, -.3, .08, -.2, 1.)):
            with self.subTest(step=step):
                self.compare(step=step, action=np.r_[self.q + delta, .01])

    def test_tool_capsule_uses_single_finger_gate_and_original_length(self):
        tool_near = dict(self.cap, p1=np.array([.30, .01, .2]), p2=np.array([.30, .01, .6]))
        for opening in (.04, .04000001):
            with self.subTest(total_opening=opening):
                self.compare(action=np.r_[self.action[:6], opening], cap=tool_near)
        self.data.xpos[3, 0] = .60
        self.compare(cap=tool_near)

    def test_agv_limits_clip_legacy_output_and_preserve_gripper(self):
        self.arm.limits[:] = [-.025, .025]
        action = np.r_[self.q + [10., -10., 10., -10., 10., -10.], .073]
        output, _ = self.compare(action=action)
        self.assertTrue(np.all(output[:6] >= self.arm.limits[:, 0]))
        self.assertTrue(np.all(output[:6] <= self.arm.limits[:, 1]))

    def test_visual_geometry_does_not_follow_oracle_pillar(self):
        baseline, _ = self.compare()
        self.legacy.last_q_dot = np.zeros(6)
        self.arm.apf_last_q_dot[:] = 0.
        self.data.xpos[4] = [10., 10., 10.]
        moved, _ = self.compare()
        np.testing.assert_array_equal(baseline, moved)
        with self.assertRaises(RuntimeError):
            deploy.process_apf_action(self.arm, self.action, None, 30, .22)

    def test_geometry_and_damped_pseudoinverse_match_legacy(self):
        for points in (
            ([0., 0., 0.], [0., 0., 0.], [1., 0., 0.], [1., 0., 0.]),
            ([0., 0., 0.], [1., 0., 0.], [.5, 0., 0.], [.5, 1., 0.]),
            ([0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]),
            ([.3, .2, .1], [.3, .2, .5], [.4, .1, .2], [.5, .3, .6]),
        ):
            for actual, expected in zip(deploy.segment_to_segment_distance(*points),
                                        self.legacy.segment_to_segment_distance(*points)):
                np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-12)
        np.testing.assert_allclose(deploy.damped_pinv(self.jacobian, .05),
                                   self.legacy.damped_pinv(self.jacobian, .05), rtol=0., atol=1e-12)


class CollisionMetricsTests(unittest.TestCase):
    def test_rates_count_contact_and_tilt_independently_of_success(self):
        rows = [
            dict(episode=0, reason="success", success=True, steps=25,
                 obstacle_present=True, contact_collision=True, pillar_tilted=False),
            dict(episode=1, reason="timeout", success=False, steps=50,
                 obstacle_present=True, contact_collision=False, pillar_tilted=True),
            dict(episode=2, reason="Perception failed", success=False, steps=0,
                 obstacle_present=True, contact_collision=False, pillar_tilted=False),
            dict(episode=3, reason="Viewer closed", success=False, steps=10,
                 obstacle_present=True, contact_collision=True, pillar_tilted=True),
            dict(episode=4, reason="success", success=True, steps=25,
                 obstacle_present=False, contact_collision=False, pillar_tilted=False),
        ]
        with tempfile.TemporaryDirectory() as temp:
            deploy.write_summary(Path(temp), rows, 5)
            result = json.loads((Path(temp) / "summary.json").read_text())
            self.assertEqual(result["collision_evaluated_episodes"], 2)
            self.assertEqual(result["contact_collision_rate"], .5)
            self.assertEqual(result["pillar_tilt_rate"], .5)
            self.assertIn("contact_collision", (Path(temp) / "episodes.csv").read_text())

    def test_no_obstacle_rate_is_not_applicable(self):
        with tempfile.TemporaryDirectory() as temp:
            deploy.write_summary(Path(temp), [], 1)
            result = json.loads((Path(temp) / "summary.json").read_text())
            self.assertIsNone(result["contact_collision_rate"])
            self.assertIsNone(result["pillar_tilt_rate"])

    def test_table_contact_is_excluded_and_tool_contact_is_latched(self):
        arm = deploy.CollisionTrackedArm(deploy.ROOT / "models/agv_dummyx/scene_obstacle.xml")
        arm.reset()
        arm.start_collision_recording()
        arm.observe_collision()
        self.assertFalse(arm.contact_collision)
        self.assertFalse(arm.pillar_tilted)
        arm.data.joint("pillar_joint").qpos[:3] = arm.data.body("real_screwdriver").xpos.copy()
        arm.mj.mj_forward(arm.model, arm.data)
        arm.observe_collision()
        self.assertTrue(arm.contact_collision)
        arm.reset()
        arm.observe_collision()
        self.assertTrue(arm.contact_collision)
        arm.start_collision_recording()
        self.assertFalse(arm.contact_collision)
        angle = np.deg2rad(30.)
        arm.data.joint("pillar_joint").qpos[3:7] = [np.cos(angle / 2), np.sin(angle / 2), 0., 0.]
        arm.mj.mj_forward(arm.model, arm.data)
        arm.observe_collision()
        self.assertTrue(arm.pillar_tilted)

    def test_transient_substep_contact_and_tilt_are_not_lost(self):
        arm = deploy.CollisionTrackedArm.__new__(deploy.CollisionTrackedArm)
        arm.steps = 2
        arm.pillar_geom, arm.pillar_body = 1, 1
        arm.collision_geoms = {2}
        arm.tracking_collisions = True
        arm.contact_collision = arm.pillar_tilted = False
        arm.model = None
        arm.data = SimpleNamespace(contact=[], xmat=np.zeros((2, 9)),
                                   qpos=np.zeros(1), warning=SimpleNamespace(number=np.zeros(1)))
        arm.command = lambda action: None
        calls = []
        def physics_step(model, data):
            calls.append(1)
            data.contact = ([SimpleNamespace(geom=(1, 2), dist=-.001)] if len(calls) == 1 else [])
            data.xmat[1, 8] = .8 if len(calls) == 1 else 1.
        arm.mj = SimpleNamespace(mj_step=physics_step, mj_forward=lambda model, data: None)
        arm.step(np.zeros(7))
        self.assertEqual(arm.data.contact, [])
        self.assertEqual(arm.data.xmat[1, 8], 1.)
        self.assertTrue(arm.contact_collision)
        self.assertTrue(arm.pillar_tilted)

    def test_substep_observation_preserves_physics(self):
        scene = deploy.ROOT / "models/agv_dummyx/scene_obstacle.xml"
        original, tracked = deploy.HardstopArm(scene), deploy.CollisionTrackedArm(scene)
        action = original.reset()
        tracked.reset()
        tracked.start_collision_recording()
        for _ in range(20):
            original.step(action)
            tracked.step(action)
        np.testing.assert_array_equal(original.data.qpos, tracked.data.qpos)
        np.testing.assert_array_equal(original.data.qvel, tracked.data.qvel)


if __name__ == "__main__":
    unittest.main()
