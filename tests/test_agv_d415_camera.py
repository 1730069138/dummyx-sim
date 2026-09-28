"""Check the physical mount and D415 RGB framing in the AGV task scene."""
from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()
