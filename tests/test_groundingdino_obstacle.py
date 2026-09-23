"""Run with: python -m unittest discover -s tests -v (MuJoCo environment)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "deploy"))
import groundingdino_obstacle as perception
import deploy_screwdriver_client as deploy


def pillar_cloud():
    x, y = np.meshgrid(np.linspace(-0.018, 0.018, 12), np.linspace(-0.018, 0.018, 12))
    top = np.stack((x + 0.11, y - 0.10, np.full_like(x, 0.36)), axis=2)
    sx, sz = np.meshgrid(np.linspace(-0.018, 0.018, 12), np.linspace(0.22, 0.35, 20))
    side = np.stack((sx + 0.11, np.full_like(sx, -0.08), sz), axis=2)
    return np.concatenate((top, side), axis=0)


class GeometryTests(unittest.TestCase):
    def test_center_and_corner_enclosure(self):
        capsule, _ = perception.fit_known_pillar(pillar_cloud())
        np.testing.assert_allclose((capsule["p1"] + capsule["p2"]) / 2, [0.11, -0.1, 0.28])
        self.assertGreater(capsule["r"], np.hypot(0.02, 0.02))

    def test_invalid_partial_and_table_clouds_rejected(self):
        cloud = pillar_cloud()
        table = cloud.copy()
        table[:, :, 2] = 0.20
        for invalid in (np.full_like(cloud, np.nan), cloud[:, :4], table, cloud[:12]):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                perception.fit_known_pillar(invalid)

    def test_multiple_pillars_rejected(self):
        first = pillar_cloud()
        second = first + [0.2, 0, 0]
        separator = np.full((first.shape[0], 2, 3), np.nan)
        with self.assertRaises(ValueError):
            perception.fit_known_pillar(np.concatenate((first, separator, second), axis=1))

    def test_camera_axes_and_invalid_depth(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><camera name="rear_cam" pos="1 2 3" fovy="90"/></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        depth = np.full((3, 3), 2.0)
        points = perception.bbox_depth_to_world_points(model, data, "rear_cam", depth, (0, 0, 3, 3))
        np.testing.assert_allclose(points[4], [1, 2, 1])
        self.assertLess(points[0, 0], points[4, 0])
        self.assertGreater(points[0, 1], points[4, 1])
        depth[:] = np.nan
        self.assertEqual(len(perception.bbox_depth_to_world_points(model, data, "rear_cam", depth, (0, 0, 3, 3))), 0)

    def test_no_detection_saves_failure(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><camera name="rear_cam"/></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        class EmptyDetector:
            def detect_all_boxes(self, *args):
                return []
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "failed"
            with patch.object(perception, "render_rgb_depth", return_value=(np.zeros((3, 3, 3), np.uint8), np.ones((3, 3)))):
                with self.assertRaises(ValueError):
                    perception.localize_pillar(model, data, None, EmptyDetector(), "red pillar.", 0.35, 0.25, output)
            metadata = json.loads((output / "perception.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertIn("0 detections", metadata["reason"])
            self.assertTrue((output / "rgb.png").is_file())


class APFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "dummyx_apf_scene.xml"))

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        deploy.reset_scene(self.model, self.data, True, (0.28, -0.1))
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "fj_screwdriver")
        self.data.qpos[self.model.jnt_qposadr[jid] + 2] = 0.30
        mujoco.mj_forward(self.model, self.data)
        deploy.last_q_dot = np.zeros(6)

    def test_visual_geometry_does_not_follow_oracle(self):
        capsule, _ = perception.fit_known_pillar(pillar_cloud())
        before = deploy.process_apf_action(self.model, self.data, np.zeros(7), True, 20, "groundingdino", capsule)
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pillar_joint")
        self.data.qpos[self.model.jnt_qposadr[jid]:self.model.jnt_qposadr[jid] + 3] = [2, 3, 4]
        mujoco.mj_forward(self.model, self.data)
        deploy.last_q_dot = np.zeros(6)
        after = deploy.process_apf_action(self.model, self.data, np.zeros(7), True, 20, "groundingdino", capsule)
        np.testing.assert_allclose(before[0], after[0])
        self.assertIs(after[2], capsule)

    def test_missing_visual_geometry_never_falls_back(self):
        with self.assertRaises(RuntimeError):
            deploy.process_apf_action(self.model, self.data, np.zeros(7), True, 20, "groundingdino")

    def test_oracle_default_and_no_obstacle(self):
        result = deploy.process_apf_action(self.model, self.data, np.zeros(7), True, 20)
        np.testing.assert_allclose((result[2]["p1"] + result[2]["p2"]) / 2, [0.11, -0.1, 0.38])
        self.assertEqual(result[2]["r"], 0.025)
        result = deploy.process_apf_action(self.model, self.data, np.zeros(7), False, 20, "groundingdino")
        self.assertIsNone(result[2])


if __name__ == "__main__":
    unittest.main()
