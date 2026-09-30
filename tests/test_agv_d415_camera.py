"""Check the physical mount and D415 RGB framing in the AGV task scene."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "models/agv_dummyx/scene.xml"


class D415CameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(SCENE))
        cls.data = mujoco.MjData(cls.model)

    def test_pole_occupies_deck_hole_and_clears_upright_arm(self):
        model, data = self.model, self.data
        mesh = trimesh.load(SCENE.parent / "base_link.STL", process=False)
        vertices = mesh.vertices
        ring = np.unique(np.round(vertices[(np.abs(vertices[:, 2] - .532) < 1e-5) &
                                           (np.abs(vertices[:, 0] + .099992) < .04) &
                                           (np.abs(vertices[:, 1] + .2) < .04), :2], 6), axis=0)
        self.assertEqual(len(ring), 14)
        np.testing.assert_allclose(np.linalg.norm(ring - [-.099992, -.2], axis=1), .03, atol=1e-6)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.body("d415_pole_mount").xpos,
                                   [-.099992, -.2, .6019146091938019], atol=1e-9)
        self.assertLess(model.geom("d415_pole_flange").size[0], .03)
        np.testing.assert_allclose(data.body("d415_camera_mount").xpos[2], 1.64, atol=1e-9)

        upright = [3.0543261909900767, 1.39618219, -.42898494, -1.95145762,
                   1.48859977, 3.38201935]
        for index, value in enumerate(upright, 1):
            data.joint(f"joint{index}").qpos[:] = value
        mujoco.mj_forward(model, data)
        arm_top_bound = max(data.geom_xpos[g, 2] + model.geom_rbound[g]
                            for g in range(model.ngeom) if model.geom(g).name.startswith("visual_link"))
        self.assertGreater(data.body("d415_camera_mount").xpos[2] - arm_top_bound, .05)

    def test_rgb_view_contains_all_tabletop_corners(self):
        model, data = self.model, self.data
        mujoco.mj_forward(model, data)
        overview = model.camera("overview").id
        alias = model.camera("d415_rgb").id
        np.testing.assert_allclose(data.cam_xpos[overview], data.cam_xpos[alias])
        np.testing.assert_allclose(data.cam_xmat[overview], data.cam_xmat[alias])
        corners = np.array([[x, y, .8] for x in (.3609761143, 1.0609761143)
                            for y in (-.6, .6)])
        local = (corners - data.cam_xpos[overview]) @ data.cam_xmat[overview].reshape(3, 3)
        depth = -local[:, 2]
        self.assertTrue(np.all(depth > 0))
        half_vertical = np.tan(np.deg2rad(model.cam_fovy[overview] / 2))
        normalized = np.column_stack([local[:, 0] / depth / (half_vertical * 640 / 360),
                                      local[:, 1] / depth / half_vertical])
        self.assertLess(np.abs(normalized).max(), .95)

    def test_d415_visual_uses_metre_mesh_and_faces_rgb_direction(self):
        mesh = trimesh.load(SCENE.parent / "d415_camera.STL", force="mesh")
        np.testing.assert_allclose(mesh.extents, [.099, .023, .020], atol=.001)

        geom = ET.parse(SCENE).find(".//geom[@name='d415_camera_visual']")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.fromstring(geom.get("quat"), sep=" "))
        mujoco.mj_forward(self.model, self.data)
        mount_rotation = self.data.body("d415_camera_mount").xmat.reshape(3, 3)
        camera_rotation = self.data.camera("overview").xmat.reshape(3, 3)
        mesh_front = mount_rotation @ rotation.reshape(3, 3) @ [0, 0, 1]
        self.assertGreater(np.dot(mesh_front, -camera_rotation[:, 2]), .99)

    def test_pole_stops_below_camera_and_stud_meets_mounting_hole(self):
        scene = ET.parse(SCENE)
        geom = scene.find(".//geom[@name='d415_camera_visual']")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.fromstring(geom.get("quat"), sep=" "))
        rotation = rotation.reshape(3, 3)
        position = np.fromstring(geom.get("pos"), sep=" ")
        mesh = trimesh.load(SCENE.parent / "d415_camera.STL", force="mesh")
        vertices = mesh.vertices @ rotation.T + position
        hole = np.array([.05, mesh.bounds[0, 1], .010])
        np.testing.assert_allclose(rotation @ hole + position, [0, 0, 0], atol=1e-6)

        pole = scene.find(".//geom[@name='d415_pole']")
        pole_top = float(pole.get("pos").split()[2]) + float(pole.get("size").split()[1])
        camera_mount = scene.find(".//body[@name='d415_camera_mount']")
        self.assertGreater(float(camera_mount.get("pos").split()[2]) + vertices[:, 2].min() - pole_top, .002)
        stud = scene.find(".//geom[@name='d415_mount_stud']")
        stud_top = float(stud.get("pos").split()[2]) + float(stud.get("size").split()[1])
        self.assertAlmostEqual(stud_top, float(camera_mount.get("pos").split()[2]), places=6)


if __name__ == "__main__":
    unittest.main()
