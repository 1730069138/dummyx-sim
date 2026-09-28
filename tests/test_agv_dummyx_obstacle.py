"""Check that obstacle evaluation cannot leak into the default collection scene."""
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/tools'))
from preview_agv_dummyx import load_model


class AGVObstacleSceneTests(unittest.TestCase):
    def test_neutral_z_pose_faces_vehicle_forward(self):
        for obstacle in (False, True):
            _, data = load_model(obstacle=obstacle)
            np.testing.assert_allclose(data.site('tcp_site').xmat.reshape(3, 3)[:, 2],
                                       [1., 0., 0.], atol=1e-10)
            self.assertAlmostEqual(float(data.joint('joint2').qpos[0]), 0.)

    def test_layout_centers_pillar_and_offsets_bin_five_cm_inside_left_edge(self):
        model, data = load_model(obstacle=True)
        arm = data.body('link0').xpos
        pillar = data.body('dynamic_pillar').xpos
        self.assertAlmostEqual(pillar[1], arm[1], places=9)
        self.assertGreater(pillar[0], arm[0])
        mesh = model.geom('agv_base_visual').dataid[0]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        geom = data.geom('agv_base_visual')
        vertices = model.mesh_vert[start:start + count] @ geom.xmat.reshape(3, 3).T + geom.xpos
        self.assertAlmostEqual(data.body('drop_box').xpos[1], vertices[:, 1].max() - .05, places=6)

    def test_default_scene_has_no_obstacle_body_geom_or_joint(self):
        model, _ = load_model()
        for kind, name in ((mujoco.mjtObj.mjOBJ_BODY, 'dynamic_pillar'),
                           (mujoco.mjtObj.mjOBJ_GEOM, 'pillar_geom'),
                           (mujoco.mjtObj.mjOBJ_JOINT, 'pillar_joint')):
            self.assertEqual(mujoco.mj_name2id(model, kind, name), -1)

    def test_optional_scene_preserves_robot_and_task_geometry(self):
        base, original = load_model()
        test, enabled = load_model(obstacle=True)
        self.assertEqual(test.nq, base.nq + 7)
        self.assertEqual(test.nu, base.nu)
        for i in range(base.njnt):
            name = base.joint(i).name
            np.testing.assert_array_equal(original.joint(name).qpos, enabled.joint(name).qpos)
        for i in range(base.ngeom):
            name = base.geom(i).name
            if not name:
                continue
            np.testing.assert_allclose(original.geom(name).xpos, enabled.geom(name).xpos, atol=1e-12)
            np.testing.assert_array_equal(base.geom(name).size, test.geom(name).size)
            np.testing.assert_array_equal(base.geom(name).contype, test.geom(name).contype)
        self.assertGreater(test.geom('pillar_geom').contype[0], 0)
        self.assertGreater(test.geom('pillar_geom').conaffinity[0], 0)

    def test_pillar_is_separate_from_props_and_stands_on_table(self):
        model, data = load_model(obstacle=True)
        pillar = model.geom('pillar_geom').id
        table = model.geom('workbench_top').id
        initial = data.body('dynamic_pillar').xpos.copy()
        for c in data.contact:
            if pillar in c.geom:
                self.assertIn(table, c.geom)
        for _ in range(round(1 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.body('dynamic_pillar').xpos, initial, atol=.001)
        self.assertGreater(data.body('dynamic_pillar').xmat[8], .999)
        self.assertTrue(any(pillar in c.geom and table in c.geom for c in data.contact))
        self.assertFalse(np.any(data.warning.number))


if __name__ == '__main__':
    unittest.main()
