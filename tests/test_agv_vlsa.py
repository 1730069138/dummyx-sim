"""Shared-perception VLSA mapping and comparison checks; no policy service."""
import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/deploy"))
import agv_vlsa
import compare_agv_pi0_experiments as comparison
import deploy_agv_dummyx_pi0 as deploy
import deploy_screwdriver_client_vlsa as reference
from test_agv_pi0_evaluation import FakeArm


CAPSULE = {"p1": np.array([.46, .049, .81]),
           "p2": np.array([.46, .049, .95]), "r": .033}


class AdapterTests(unittest.TestCase):
    def test_ellipsoid_encloses_capsule_for_arbitrary_orientation(self):
        rng = np.random.default_rng(42)
        for direction in (np.array([0., 0., 1.]), np.array([1., .2, -.4])):
            capsule = {"p1": np.array([.3, .4, .8]), "p2": np.array([.3, .4, .8]) + direction * .1, "r": .033}
            center, rotation, axes = agv_vlsa.capsule_ellipsoid(capsule)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)
            unit = rng.normal(size=(2000, 3))
            unit /= np.linalg.norm(unit, axis=1)[:, None]
            for t in (0., .25, .5, .75, 1.):
                points = capsule["p1"] * (1-t) + capsule["p2"] * t + capsule["r"] * unit
                self.assertLessEqual(np.max(np.sum(((points-center) @ rotation / axes) ** 2, axis=1)), 1.)

    def test_agv_joint_addresses_actual_velocity_and_gripper_are_preserved(self):
        q = np.array([.02, -.03, .01, .04, -.02, .05])
        addresses = np.arange(6, 12)
        arm = SimpleNamespace(model=SimpleNamespace(body=lambda name: SimpleNamespace(id=16)),
                              data=SimpleNamespace(qvel=np.arange(14) * .01), dadr=addresses,
                              state=lambda: np.r_[q, .101], limits=np.tile([-3., 3.], (6, 1)))
        jacp = np.arange(42).reshape(3, 14) / 100.
        jacr = np.flip(jacp, axis=1).copy()
        target = np.r_[q + [.04, -.08, .07, -.03, .02, -.01], .037]
        with patch.object(reference, "get_eef_ellipsoid_pose", return_value=(np.array([.2, .1, 1.]), np.eye(3))), \
                patch.object(reference, "get_point_jacobians", return_value=(jacp, jacr)):
            adapter = agv_vlsa.SharedPerceptionVLSA(arm, CAPSULE, .02)
            def filter_twist(center, rotation, v, omega, actual_v, actual_omega):
                np.testing.assert_allclose(v, jacp[:, addresses] @ ((target[:6]-q) / .02))
                np.testing.assert_allclose(actual_v, jacp[:, addresses] @ arm.data.qvel[addresses])
                np.testing.assert_allclose(actual_omega, jacr[:, addresses] @ arm.data.qvel[addresses])
                return v, omega, {"qp_ok": False, "qp_ms": 2.5, "h": -.02}
            adapter.layer.filter = filter_twist
            timing = {}
            output = adapter.process(target, timing)
        np.testing.assert_allclose(output, target)
        self.assertEqual(timing["solver_time_ms"], 2.5)
        self.assertEqual(adapter.qp_failures, 1)
        self.assertEqual(adapter.min_h, -.02)

    def test_original_qp_on_agv_static_state(self):
        arm = deploy.CollisionTrackedArm(scene=deploy.ROOT / "models/agv_dummyx/scene_obstacle.xml")
        action = arm.reset()
        adapter = agv_vlsa.SharedPerceptionVLSA(arm, CAPSULE, 1 / deploy.FPS)
        timing = {}
        output = adapter.process(action, timing)
        self.assertEqual(output.shape, (7,))
        self.assertTrue(np.isfinite(output).all())
        self.assertEqual(output[6], action[6])
        self.assertTrue(np.all(output[:6] >= arm.limits[:, 0]))
        self.assertTrue(np.all(output[:6] <= arm.limits[:, 1]))
        self.assertGreater(timing["solver_time_ms"], 0)
        self.assertEqual(adapter.qp_failures, 0)
        self.assertFalse(adapter.geometry_metadata()["protects_carried_tool"])

    def test_correction_matches_original_cartesian_adapter(self):
        arm = deploy.CollisionTrackedArm(scene=deploy.ROOT / "models/agv_dummyx/scene_obstacle.xml")
        action = arm.reset()
        action[:6] += np.array([.01, -.02, .015, .01, -.005, .02])
        adapter = agv_vlsa.SharedPerceptionVLSA(arm, CAPSULE, .02)
        center, rotation = adapter.eef_pose()
        jacp, jacr = reference.get_point_jacobians(arm.model, arm.data, center, adapter.eef_body)
        legacy_data = SimpleNamespace(qpos=arm.state()[:6], qvel=arm.data.qvel[arm.dadr])
        legacy_layer = reference.AEGISFullActionLayer(
            adapter.layer.p_obs, adapter.layer.r_obs, adapter.layer.q_obs, .02,
            eef_axes=adapter.layer.eef_axes, max_linear_correction=.08, max_angular_correction=.08)
        legacy_layer.reset(center)
        output = adapter.process(action)
        with patch.object(reference, "get_eef_ellipsoid_pose", return_value=(center, rotation)), \
                patch.object(reference, "get_point_jacobians", return_value=(jacp[:, arm.dadr], jacr[:, arm.dadr])):
            legacy_output, _ = reference.adapt_joint_target_through_vlsa(
                None, legacy_data, action, legacy_layer, adapter.eef_body, .02)
        np.testing.assert_allclose(output[:6], np.clip(legacy_output[:6], arm.limits[:, 0], arm.limits[:, 1]), atol=1e-10)
        self.assertEqual(output[6], action[6])

    def test_episode_routes_vlsa_and_records_solver_timing(self):
        arm = FakeArm()
        renderer = SimpleNamespace(update_scene=lambda *a, **k: None,
                                   render=lambda: np.zeros((2, 2, 3), dtype=np.uint8))
        policy = SimpleNamespace(infer=lambda obs: {"actions": np.tile(np.r_[np.zeros(6), .101], (50, 1))})
        adapter = SimpleNamespace(qp_failures=2, min_h=-.01)
        def process(action, timing):
            timing["solver_time_ms"] = 1.25
            return action
        adapter.process = process
        timings = {key: [] for key in ("policy_inference_time_ms", "safety_layer_time_ms", "solver_time_ms")}
        with patch.object(deploy, "load_spawn_region", return_value={}), \
                patch.object(agv_vlsa, "SharedPerceptionVLSA", return_value=adapter), \
                patch.object(deploy, "process_apf_action", side_effect=AssertionError("VLSA must not call APF")):
            result, _, actions, *_ = deploy.run_episode(
                arm, policy, renderer, renderer, None, np.random.default_rng(42), True, 25, 50,
                perception=lambda arm, directory: (CAPSULE, {}), timings=timings, safety_mode="vlsa")
        self.assertTrue(result["success"])
        self.assertEqual(len(actions), 25)
        self.assertEqual(timings["solver_time_ms"], [1.25] * 25)
        self.assertEqual(result["vlsa_qp_failures"], 2)
        self.assertEqual(result["vlsa_min_h"], -.01)


class ScheduleTests(unittest.TestCase):
    def test_vlsa_only_cli_selects_one_mode_and_100_rounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "dino.py"; config.touch()
            weights = Path(temporary) / "dino.pth"; weights.touch()
            with patch.object(sys, "argv", ["deploy_agv_dummyx_pi0.py", "--vlsa_only", "--num_rounds", "100",
                                            "--fixed_eval", "--record_experiment", "--device", "cpu",
                                            "--groundingdino_config", str(config),
                                            "--groundingdino_checkpoint", str(weights)]), \
                    patch.object(deploy, "run_comparison_rounds") as run:
                deploy.main()
            args = run.call_args.args[0]
            self.assertEqual(args.mode_order, ("vlsa",))
            self.assertEqual(args.num_rounds, 100)
            self.assertEqual(args.chunk_steps, 50)
            self.assertEqual(args.seed, 42)
            self.assertEqual(args.output, deploy.ROOT / "论文ppt材料")

    def test_four_modes_vlsa_only_and_legacy_order(self):
        for modes in (deploy.COMPARISON_MODES, ("vlsa",), ("apf", "no_obstacle", "obstacle")):
            with self.subTest(modes=modes), tempfile.TemporaryDirectory() as temporary:
                args = SimpleNamespace(mode_order=modes, resume=None, output=Path(temporary),
                                       num_rounds=2, seed=42, host="127.0.0.1", port=8000,
                                       max_steps=5000, chunk_steps=50, task_prompt=deploy.INSTRUCTION,
                                       device="cpu", obstacle_text="red pillar.", box_threshold=.35,
                                       text_threshold=.25, groundingdino_config="config.py",
                                       groundingdino_checkpoint="weights.pth", fixed_eval=True,
                                       headless=False, realtime=True, record_experiment=True, export_video=False)
                calls = []
                def command(argv):
                    calls.append(argv)
                    if "--eval_mode" not in argv:
                        return
                    folder = Path(argv[argv.index("--output") + 1]) / "run_test"
                    (folder / "ep_0000").mkdir(parents=True)
                    (folder / "ep_0000/result.json").write_text(json.dumps(
                        {"success": True, "reason": "success", "steps": 25,
                         "obstacle_present": True, "contact_collision": False, "pillar_tilted": False}))
                with patch.object(deploy, "_run_comparison_command", side_effect=command), redirect_stdout(io.StringIO()):
                    deploy.run_comparison_rounds(args)
                    output, = Path(temporary).glob("comparison_*")
                    args.resume = output
                    deploy.run_comparison_rounds(args)
                child_calls = [call for call in calls if "--eval_mode" in call]
                self.assertEqual([call[call.index("--eval_mode") + 1] for call in child_calls], list(modes) * 2)
                self.assertEqual([int(call[call.index("--seed") + 1]) for call in child_calls],
                                 [42] * len(modes) + [43] * len(modes))
                self.assertTrue(all("--defer_plots" in call and "--export_video" not in call for call in child_calls))
                schedule = json.loads((output / "schedule.json").read_text())
                self.assertEqual(schedule["status"], "complete")
                self.assertEqual(schedule["completed_episodes"], 2 * len(modes))


class ComparisonTests(unittest.TestCase):
    def test_merge_four_modes_and_reject_unmatched_poses(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, extra = Path(temporary) / "old", Path(temporary) / "vlsa"
            base.mkdir(); extra.mkdir()
            base_entries, extra_entries = [], []
            for round_index in range(2):
                for mode in (*comparison.MODES, "vlsa"):
                    folder = Path(temporary) / f"case_{round_index}_{mode}"
                    experiment = folder / "experiment/ep_0000"
                    experiment.mkdir(parents=True)
                    scene = "plain.xml" if mode == "no_obstacle" else "obstacle.xml"
                    config = {"defer_plots": True, "evaluation_mode": mode, "scene": scene,
                              "model_manifest": {"robot.xml": "robot", scene: scene},
                              **{key: value for key, value in zip(
                                  ("seed", "chunk_steps", "task_prompt", "fixed_eval", "max_steps", "fps",
                                   "success_criteria", "collision_criteria", "comparison_scene_signature"),
                                  (42+round_index, 50, deploy.INSTRUCTION, True, 5000, 50, {}, {}, "same"))}}
                    (folder / "run_config.json").write_text(json.dumps(config))
                    (folder / "experiment/run_config.json").write_text(json.dumps({"server_metadata": {"model": "same"}}))
                    metric = {"episode": 0, "evaluation_mode": mode, "task": "screwdriver",
                              "task_prompt": deploy.INSTRUCTION, "initial_xy": [.5, 0.], "initial_yaw": 0.,
                              "obstacle_present": mode != "no_obstacle", "success": True, "reason": "success",
                              "contact_collision": False, "pillar_tilted": False, "successful_completion_time_s": 2.}
                    (experiment / "metrics.json").write_text(json.dumps(metric))
                    entry = {"round": round_index, "evaluation_mode": mode, "run_path": str(folder)}
                    (extra_entries if mode == "vlsa" else base_entries).append(entry)
            (base / "schedule.json").write_text(json.dumps({"mode_order": comparison.MODES, "episodes": base_entries}))
            (extra / "schedule.json").write_text(json.dumps({"mode_order": ["vlsa"], "episodes": extra_entries}))
            runs, matched = comparison.load_comparison_rounds(base, extra)
            self.assertEqual(len(runs), 4)
            self.assertEqual(matched["matched_rounds"], 2)
            self.assertTrue(all(len(run["episodes"]) == 2 for run in runs))
            vlsa_runs, _ = comparison.load_comparison_rounds(extra)
            self.assertEqual([run["mode"] for run in vlsa_runs], ["vlsa"])
            metric_path = Path(extra_entries[0]["run_path"]) / "experiment/ep_0000/metrics.json"
            bad = json.loads(metric_path.read_text()); bad["initial_xy"][0] += .01
            metric_path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "unmatched initial object pose"):
                comparison.load_comparison_rounds(base, extra)


if __name__ == "__main__":
    unittest.main()
