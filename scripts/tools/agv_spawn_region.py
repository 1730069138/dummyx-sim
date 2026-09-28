"""Viewer-only spawn rectangle. Never adds geoms or sites to the physics model."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

REGION_PATH = Path(__file__).resolve().parents[2] / 'models/agv_dummyx/spawn_region.json'


def load_spawn_region(model, data):
    """Keep the debug position rectangle on the table and right of the reserved pillar.

    This validates the debug rectangle. The collector separately tests full-tool
    clearance for each sampled yaw before running physics.
    """
    region = json.loads(REGION_PATH.read_text())
    bounds = np.array([region['x_range_m'], region['y_range_m']], dtype=float)
    margin = float(region['obstacle_clearance_m'])
    if (bounds.shape != (2, 2) or not np.all(np.isfinite(bounds)) or
            np.any(bounds[:, 0] >= bounds[:, 1]) or not np.isfinite(margin) or margin < 0):
        raise ValueError('Invalid spawn rectangle or clearance')
    table = data.geom('workbench_top').xpos
    half = model.geom('workbench_top').size
    if np.any(bounds[:, 0] < table[:2] - half[:2]) or np.any(bounds[:, 1] > table[:2] + half[:2]):
        raise ValueError('The spawn rectangle extends beyond the tabletop')
    pillar = ET.parse(REGION_PATH.parent / 'obstacle.xml').find("worldbody/body[@name='dynamic_pillar']")
    pillar_pos = np.fromstring(pillar.get('pos'), sep=' ')
    pillar_half = np.fromstring(pillar.find('geom').get('size'), sep=' ')
    clearance = float(pillar_pos[1] - pillar_half[1] - bounds[1, 1])
    if clearance < margin:
        raise ValueError(f'Spawn rectangle is too close to the reserved obstacle: {clearance:.4f} m < {margin:.4f} m')
    region['surface_z_m'] = float(table[2] + half[2])
    region['actual_obstacle_clearance_m'] = clearance
    return region


def draw_spawn_region(scene, region):
    """Append five decorative geoms to viewer.user_scn or an explicit debug render only.

    Call after Renderer.update_scene for a debug screenshot. Never call this on
    the renderer used for observation images. Viewer and camera scenes are separate.
    """
    if scene.ngeom + 5 > scene.maxgeom:
        raise ValueError('Not enough debug scene capacity for the spawn rectangle')
    x0, x1 = region['x_range_m']
    y0, y1 = region['y_range_m']
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = (x1 - x0) / 2, (y1 - y0) / 2
    z = region['surface_z_m'] + .001
    fill = (.05, .75, .95, .20)
    edge = (.02, .85, 1., .95)
    boxes = [((cx, cy, z), (hx, hy, .0002), fill),
             ((x0, cy, z), (.0008, hy, .0005), edge),
             ((x1, cy, z), (.0008, hy, .0005), edge),
             ((cx, y0, z), (hx, .0008, .0005), edge),
             ((cx, y1, z), (hx, .0008, .0005), edge)]
    for pos, size, rgba in boxes:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                           np.array(size), np.array(pos), np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.objid = -1
        geom.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        scene.ngeom += 1


def update_viewer_spawn_region(viewer, region):
    """Own this tool's user scene; hide it when the Viewer selects a model camera."""
    with viewer.lock():
        viewer.user_scn.ngeom = 0
        if region is not None and viewer.cam.type != mujoco.mjtCamera.mjCAMERA_FIXED:
            draw_spawn_region(viewer.user_scn, region)
