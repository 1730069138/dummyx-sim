"""Cross-check the MJCF adapter against source URDF and collection contracts."""
import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/collect"))
from collect_hardstop_screwdriver import HardstopArm, SOURCE, sample_pose


def urdf_origin(element):
    """Independent URDF Rz(yaw) Ry(pitch) Rx(roll) implementation."""
    result = np.eye(4)
    r, p, y = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    sr, sp, sy = np.sin([r, p, y])
    cr, cp, cy = np.cos([r, p, y])
    result[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]]
    result[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    return result


def source_forward(urdf, values):
    transforms = {"link0": np.eye(4)}
    transforms["link0"][:3, 3] = [.12866757754516, -.001, .8019146091938019]
    pending = list(urdf.findall("joint"))
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            motion = np.eye(4)
            if joint.get("type") != "fixed":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                q = values[joint.get("name")]
                if joint.get("type") == "prismatic":
                    motion[:3, 3] = axis * q
                else:
                    x, y, z = axis
                    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    motion[:3, :3] = np.eye(3) + np.sin(q) * skew + (1 - np.cos(q)) * skew @ skew
            transforms[joint.find("child").get("link")] = transforms[parent] @ urdf_origin(joint.find("origin")) @ motion
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("Invalid source URDF tree")
    return transforms


class HardstopArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arm = HardstopArm()
        cls.urdf = ET.parse(SOURCE / "urdf/arm_portable.urdf").getroot()

    def test_random_pose_samples_full_rectangle_and_forward_yaw_range(self):
        rng = np.random.default_rng(12)
        region = {"x_range_m": [.415, .515], "y_range_m": [-.28, -.08]}
        poses = [sample_pose(rng, region) for _ in range(4000)]
        xy = np.array([p[0] for p in poses])
        yaw = np.array([p[1] for p in poses])
        self.assertTrue(np.all((xy[:, 0] >= .415) & (xy[:, 0] < .515)))
        self.assertTrue(np.all((xy[:, 1] >= -.28) & (xy[:, 1] < -.08)))
        self.assertTrue(np.all((yaw >= np.pi / 3) & (yaw < 2 * np.pi / 3)))
        self.assertLess(yaw.min(), np.deg2rad(61))
        self.assertGreater(yaw.max(), np.deg2rad(119))
        np.testing.assert_allclose(sample_pose(rng, region, True)[0], [.415, -.16])
        self.assertAlmostEqual(sample_pose(rng, region, True)[1], np.pi / 2)

    def test_source_forward_kinematics_all_links(self):
        arm = self.arm
        rng = np.random.default_rng(91)
        poses = [json.loads((SOURCE / "config" / name).read_text())["joint_positions"]
                 for name in ("initial_pose.json", "straight7_pose.json")]
        for _ in range(8):
            pose = {}
            for joint in self.urdf.findall("joint"):
                if joint.get("type") != "fixed":
                    lim = joint.find("limit")
                    pose[joint.get("name")] = rng.uniform(float(lim.get("lower")), float(lim.get("upper")))
            pose.update(arm.coupling.coupled_positions(pose["joint7"]))
            poses.append(pose)
        for values in poses:
            for name, value in values.items():
                arm.data.joint(name).qpos[:] = value
            arm.mj.mj_forward(arm.model, arm.data)
            for name, expected in source_forward(self.urdf, values).items():
                np.testing.assert_allclose(arm.data.body(name).xpos, expected[:3, 3], atol=1e-12)
                np.testing.assert_allclose(arm.data.body(name).xmat.reshape(3, 3), expected[:3, :3], atol=1e-12)

    def test_limits_and_mesh_geometry_preserved(self):
        import trimesh
        arm = self.arm
        values = json.loads((SOURCE / "config/straight7_pose.json").read_text())["joint_positions"]
        for name, value in values.items():
            arm.data.joint(name).qpos[:] = value
        arm.mj.mj_forward(arm.model, arm.data)
        frames = source_forward(self.urdf, values)
        for joint in self.urdf.findall("joint"):
            if joint.get("type") != "fixed":
                lim = joint.find("limit")
                np.testing.assert_allclose(arm.model.joint(joint.get("name")).range,
                                           [float(lim.get("lower")), float(lim.get("upper"))], atol=1e-13)
        for link in self.urdf.findall("link"):
            name = link.get("name")
            for visual in link.findall("visual"):
                filename = visual.find("geometry/mesh").get("filename")
                mesh = trimesh.load(SOURCE / "urdf" / filename, process=False)
                transform = frames[name] @ urdf_origin(visual.find("origin"))
                expected = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
                geom = arm.model.geom(f"visual_{name}")
                mesh_id = geom.dataid[0]
                start, count = arm.model.mesh_vertadr[mesh_id], arm.model.mesh_vertnum[mesh_id]
                actual = arm.model.mesh_vert[start:start + count] @ arm.data.geom(geom.id).xmat.reshape(3, 3).T + arm.data.geom(geom.id).xpos
                np.testing.assert_allclose(actual.min(0), expected.min(0), atol=2e-7)
                np.testing.assert_allclose(actual.max(0), expected.max(0), atol=2e-7)

    def test_gripper_width_and_force_limited_control(self):
        arm = self.arm
        action = arm.reset()
        np.testing.assert_allclose(arm.state()[6], 0.101, atol=1e-5)
        arm.command(action.astype(np.float32))  # saved maximum opening remains replayable
        action[6] = 0
        for _ in range(100):
            arm.step(action)
        self.assertLess(abs(arm.state()[6]), 1e-4)
        self.assertLess(abs(arm.data.joint("joint10").qpos[0] - arm.data.joint("joint12").qpos[0]), 1e-6)
        for index in (10, 12):
            np.testing.assert_allclose(arm.model.actuator_forcerange[arm.servo[index]], [-12, 12])
        for bad in (np.zeros(8), np.full(7, np.nan), np.r_[action[:6], -0.01],
                    np.r_[action[:6], 0.102], np.r_[-1., action[1:]]):
            with self.assertRaises(ValueError):
                arm.command(bad)

    def test_default_run_holds_z_and_commands_remain_absolute(self):
        arm = self.arm
        arm.mj.mj_resetData(arm.model, arm.data)
        z_pose = json.loads((SOURCE.parent / "agv_dummyx/z_pose.json").read_text())["joint_positions"]
        for name, value in z_pose.items():
            self.assertAlmostEqual(arm.data.joint(name).qpos[0], value)
        start = arm.data.qpos.copy()
        for _ in range(1000):
            arm.mj.mj_step(arm.model, arm.data)
        np.testing.assert_allclose(arm.data.qpos[7:14], start[7:14], atol=1e-4)

        action = arm.reset()
        arm.command(action)
        for i in range(1, 7):
            self.assertAlmostEqual(arm.data.ctrl[arm.servo[i]],
                                   action[i - 1] - z_pose[f"joint{i}"])

    def test_placement_accepts_rim_lean_but_rejects_tool_center_outside_bin(self):
        arm = self.arm
        arm.reset()
        # Recorded slanted placement: tip on bin bottom, handle against the rim.
        arm.data.joint("fj_screwdriver").qpos[:] = [
            .41573588, .25002803, .85288357, .67808026, .11131164, .10562596, .71879067]
        arm.mj.mj_forward(arm.model, arm.data)
        metrics = arm.placement_metrics()
        self.assertTrue(metrics["inside_bin"])
        self.assertTrue(metrics["height_ok"])
        self.assertTrue(metrics["bin_contact"])
        self.assertGreater(arm.tool_vertices()[:, 2].max(), .853)
        arm.data.joint("fj_screwdriver").qpos[0] = .55
        arm.mj.mj_forward(arm.model, arm.data)
        self.assertFalse(arm.placement_metrics()["inside_bin"])


if __name__ == "__main__":
    unittest.main()
