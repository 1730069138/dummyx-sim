"""The debug rectangle must never enter independent observation renderers."""
import json
from contextlib import nullcontext
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/tools'))
from preview_agv_dummyx import load_model
import agv_spawn_region as overlay


class SpawnRegionTests(unittest.TestCase):
    def test_viewer_hides_overlay_when_switching_to_a_model_camera(self):
        model, data = load_model()
        region = overlay.load_spawn_region(model, data)
        viewer = SimpleNamespace(cam=mujoco.MjvCamera(), user_scn=mujoco.MjvScene(model, maxgeom=32),
                                 lock=nullcontext)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        overlay.update_viewer_spawn_region(viewer, region)
        self.assertEqual(viewer.user_scn.ngeom, 5)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        overlay.update_viewer_spawn_region(viewer, region)
        self.assertEqual(viewer.user_scn.ngeom, 0)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        overlay.update_viewer_spawn_region(viewer, region)
        self.assertEqual(viewer.user_scn.ngeom, 5)

    def test_rectangle_is_separate_from_model_and_on_safe_side(self):
        model, data = load_model(obstacle=True)
        region = overlay.load_spawn_region(model, data)
        before = (model.ngeom, model.nbody, model.nq, model.nv)
        qpos = data.qpos.copy()
        user_scene = mujoco.MjvScene(model, maxgeom=32)
        overlay.draw_spawn_region(user_scene, region)
        self.assertEqual(user_scene.ngeom, 5)
        self.assertEqual(before, (model.ngeom, model.nbody, model.nq, model.nv))
        np.testing.assert_array_equal(data.qpos, qpos)
        self.assertGreaterEqual(region['actual_obstacle_clearance_m'], region['obstacle_clearance_m'])
        self.assertNotIn('yaw_range_deg', region)

    def test_invalid_region_near_reserved_obstacle_is_rejected_without_pillar(self):
        model, data = load_model()
        config = json.loads(overlay.REGION_PATH.read_text())
        config['y_range_m'] = [-.10, -.025]
        # Place the temporary file beside a minimal copy of the obstacle XML.
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / 'obstacle.xml').write_bytes((overlay.REGION_PATH.parent / 'obstacle.xml').read_bytes())
            path = folder / 'spawn_region.json'
            path.write_text(json.dumps(config))
            with patch.object(overlay, 'REGION_PATH', path):
                with self.assertRaisesRegex(ValueError, 'too close'):
                    overlay.load_spawn_region(model, data)

    def test_rgb_depth_and_segmentation_do_not_contain_debug_overlay(self):
        model, data = load_model(obstacle=True)
        region = overlay.load_spawn_region(model, data)
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0
        with mujoco.Renderer(model, height=240, width=320) as sensor, \
                mujoco.Renderer(model, height=240, width=320) as debug:
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = [.45, -.1, .82]
            camera.distance, camera.azimuth, camera.elevation = 1., 135, -50
            debug.update_scene(data, camera=camera, scene_option=option)
            clean = debug.render().copy()
            overlay.draw_spawn_region(debug.scene, region)
            self.assertTrue(np.any(debug.render() != clean))
            for name in ('overview', 'wrist_cam'):
                for mode in ('rgb', 'depth', 'segmentation'):
                    sensor.disable_depth_rendering()
                    sensor.disable_segmentation_rendering()
                    if mode == 'depth':
                        sensor.enable_depth_rendering()
                    elif mode == 'segmentation':
                        sensor.enable_segmentation_rendering()
                    sensor.update_scene(data, camera=name, scene_option=option)
                    before = sensor.render().copy()
                    # Refresh the independent debug view while observing the sensor.
                    debug.update_scene(data, camera=camera, scene_option=option)
                    overlay.draw_spawn_region(debug.scene, region)
                    debug.render()
                    sensor.update_scene(data, camera=name, scene_option=option)
                    np.testing.assert_array_equal(sensor.render(), before)
            # Even reusing a debug renderer starts clean after update_scene.
            debug.update_scene(data, camera=camera, scene_option=option)
            np.testing.assert_array_equal(debug.render(), clean)


if __name__ == '__main__':
    unittest.main()
