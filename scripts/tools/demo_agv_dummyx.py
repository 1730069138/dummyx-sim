"""Contact-based screwdriver-to-bin demo in the agv_dummyx factory.

Starts at the saved Z pose. Only reset writes qpos; the episode uses position
servos, finger contact, gravity and release, without attaching the screwdriver.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/collect'))
from collect_hardstop_screwdriver import HardstopArm, DOWN_ROTATION, FPS
from agv_spawn_region import load_spawn_region, update_viewer_spawn_region

SCENE = ROOT / 'models/agv_dummyx/scene.xml'
GRASP_ROTATION = (Rotation.from_euler('y', -10, degrees=True).as_matrix() @
                  Rotation.from_euler('z', 90, degrees=True).as_matrix() @ DOWN_ROTATION)
PLACE_ROTATION = (Rotation.from_euler('y', -25, degrees=True).as_matrix() @
                 Rotation.from_euler('z', 90, degrees=True).as_matrix() @ DOWN_ROTATION)
CARRY_HEIGHT = .915  # Above the 0.853 m bin rim, below the reserved 0.960 m pillar top.
CARRY_PHASES = ('lift', 'transfer', 'above_bin', 'lower')


class AGVDemo(HardstopArm):
    def __init__(self):
        super().__init__(scene=SCENE)

    def reset(self):
        self.mj.mj_resetData(self.model, self.data)
        pose = json.loads((SCENE.parent / 'z_pose.json').read_text())['joint_positions']
        for name, value in pose.items():
            self.data.joint(name).qpos[:] = value
        self.mj.mj_forward(self.model, self.data)
        self.home = self.data.qpos[self.qadr].copy()
        return np.r_[self.home, .101]

    def placement_metrics(self):
        vertices = []
        for g in self.target_geoms:
            if not self.model.geom_contype[g]:
                continue
            mesh = self.model.geom_dataid[g]
            start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            vertices.append(self.model.mesh_vert[start:start + count] @
                            self.data.geom_xmat[g].reshape(3, 3).T + self.data.geom_xpos[g])
        v = np.concatenate(vertices)
        center = self.data.body('drop_box').xpos
        return {'inside_bin': bool(np.all(np.abs(v[:, :2] - center[:2]) < [.098, .103])),
                'height_ok': bool(v[:, 2].min() >= .806 and v[:, 2].max() <= .855),
                'object_speed_m_s': float(np.linalg.norm(self.data.joint('fj_screwdriver').qvel[:3])),
                'finger_contacts': self.finger_contacts(), 'gripper_width_m': float(self.state()[6]),
                'object_xyz': self.data.body('real_screwdriver').xpos.tolist()}


def run(arm, output, viewer=None, video=True, realtime=False, show_spawn_region=False):
    import mujoco
    import cv2
    action = arm.reset()
    d = arm.data
    frames, phases, actions, objects, contacts, times = [], [], [], [], [], []
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [.35, 0., .90]
    camera.distance, camera.azimuth, camera.elevation = 2.35, 145, -27
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    region = load_spawn_region(arm.model, d) if viewer is not None and show_spawn_region else None
    if viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = camera.distance, camera.azimuth, camera.elevation
            viewer.opt.geomgroup[:] = option.geomgroup
    renderer = mujoco.Renderer(arm.model, height=720, width=960) if video else None
    writer = cv2.VideoWriter(str(output / 'demo.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (960, 720)) if video else None
    if writer is not None and not writer.isOpened():
        raise RuntimeError('Could not open video writer')
    start_height = None
    max_height = 0.
    grasp_seen = False
    retained = 0
    last_phase = None
    max_speed = 0.

    def tick(command, phase):
        nonlocal max_height, grasp_seen, last_phase, max_speed
        start = time.monotonic()
        if viewer is not None and not viewer.is_running():
            raise RuntimeError('Viewer closed')
        if last_phase != phase:
            print(f'{d.time:5.2f}s  {phase}', flush=True)
        if renderer is not None and (len(frames) % 2 == 0 or last_phase != phase):
            renderer.update_scene(d, camera=camera, scene_option=option)
            image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            if last_phase != phase:
                cv2.imwrite(str(output / f'{phase}.png'), image)
            if len(frames) % 2 == 0:
                cv2.putText(image, phase.replace('_', ' '), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, .8, (230, 240, 240), 2)
                writer.write(image)
        last_phase = phase
        arm.step(command)
        frames.append(d.qpos.copy()); phases.append(phase); actions.append(command.copy())
        objects.append(d.body('real_screwdriver').xpos.copy()); contacts.append(arm.finger_contacts()); times.append(d.time)
        max_height = max(max_height, float(d.body('real_screwdriver').xpos[2]))
        grasp_seen |= arm.finger_contacts() == 2
        max_speed = max(max_speed, float(np.abs(d.qvel[arm.dadr]).max()))
        contact = arm.forbidden_contact()
        if contact:
            raise RuntimeError(f'Forbidden contact: {contact}')
        if phase in CARRY_PHASES and arm.finger_contacts() != 2:
            raise RuntimeError(f'Lost two-finger grasp during {phase}')
        if viewer is not None:
            update_viewer_spawn_region(viewer, region)
            viewer.sync()
        if realtime:
            time.sleep(max(0., 1 / FPS - (time.monotonic() - start)))

    def move_joints(target, duration, phase):
        nonlocal action
        initial = action[:6].copy()
        for t in np.linspace(0., 1., round(duration * FPS) + 1)[1:]:
            blend = t * t * (3 - 2 * t)
            action = np.r_[initial + blend * (target - initial), action[6]]
            tick(action, phase)
        for _ in range(15):
            tick(action, phase)

    def move(position, duration, phase, rotation=GRASP_ROTATION):
        nonlocal action
        start = d.site('tcp_site').xpos.copy()
        start_rotation = Rotation.from_matrix(d.site('tcp_site').xmat.reshape(3, 3))
        delta_rotation = (Rotation.from_matrix(rotation) * start_rotation.inv()).as_rotvec()
        for t in np.linspace(0., 1., round(duration * FPS) + 1)[1:]:
            blend = t * t * (3 - 2 * t)
            target_rotation = (Rotation.from_rotvec(blend * delta_rotation) * start_rotation).as_matrix()
            q = arm.solve_ik(start + blend * (np.array(position) - start), target_rotation, action[:6])
            if np.max(np.abs(q - action[:6])) * FPS > 2.:
                raise RuntimeError(f'Joint target exceeds 2 rad/s in {phase}')
            action = np.r_[q, action[6]]
            tick(action, phase)
        for _ in range(15):
            tick(action, phase)
        error = float(np.linalg.norm(d.site('tcp_site').xpos - position))
        if error > .008:
            raise RuntimeError(f'TCP tracking error in {phase}: {error:.4f} m')

    def grip(width, duration, phase):
        nonlocal action
        start = action[6]
        for t in np.linspace(0., 1., round(duration * FPS) + 1)[1:]:
            action = np.r_[action[:6], start + t * (width - start)]
            tick(action, phase)

    try:
        for _ in range(FPS):
            tick(action, 'z_home')
        start_height = float(d.body('real_screwdriver').xpos[2])
        obj = d.body('real_screwdriver').xpos.copy()
        pregrasp = np.array([obj[0], obj[1], .885])
        seed = np.array([2.49, 2.12, -.71, -1.85, 3.47, 1.81])
        q = arm.solve_ik(pregrasp, GRASP_ROTATION, seed)
        move_joints(q, 3., 'approach')
        grip(.060, .5, 'preshape')
        move(obj + [0, 0, .002], 2., 'descend')
        grip(0., 1.2, 'close')
        for _ in range(25):
            tick(action, 'grasp_hold')
        if arm.finger_contacts() != 2:
            raise RuntimeError('No two-finger grasp')
        move([obj[0], obj[1], CARRY_HEIGHT], 2., 'lift', PLACE_ROTATION)
        if d.body('real_screwdriver').xpos[2] < start_height + .05:
            raise RuntimeError('Failed 5 cm lift check')
        move([.40, 0., CARRY_HEIGHT], 2., 'transfer', PLACE_ROTATION)
        move([.421, .185, CARRY_HEIGHT], 3., 'above_bin', PLACE_ROTATION)
        move([.419, .185, .872], 1., 'lower', PLACE_ROTATION)
        if arm.finger_contacts() != 2:
            raise RuntimeError('Lost grasp before commanded release')
        grip(.101, 1., 'release')
        move([.38, .15, .90], 1.5, 'retract', PLACE_ROTATION)
        move([.40, 0., .885], 1.5, 'clear_bin')
        move_joints(arm.home, 3., 'return_z')
        for _ in range(3 * FPS):
            tick(action, 'verify')
            metrics = arm.placement_metrics()
            valid = (metrics['inside_bin'] and metrics['height_ok'] and metrics['object_speed_m_s'] < .03
                     and metrics['finger_contacts'] == 0 and metrics['gripper_width_m'] > .08)
            retained = retained + 1 if valid else 0
            if retained >= FPS:
                break
        success = retained >= FPS
        reason = 'success' if success else 'Final release/retention check failed'
    except RuntimeError as exc:
        success, reason = False, str(exc)
    finally:
        if renderer is not None:
            renderer.update_scene(d, camera=camera, scene_option=option)
            cv2.imwrite(str(output / 'final.png'), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            renderer.close()
            writer.release()
    result = {'success': bool(success), 'reason': reason, 'max_lift_m': max_height - (start_height or max_height),
              'scene': str(SCENE.relative_to(ROOT)),
              'obstacle_present': mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, 'dynamic_pillar') >= 0,
              'two_finger_contact_seen': grasp_seen, 'retention_frames': retained, 'required_retention_frames': FPS,
              'continuous_two_finger_transport': bool(any(p in CARRY_PHASES for p in phases) and
                  all(c == 2 for p, c in zip(phases, contacts) if p in CARRY_PHASES)),
              'physics_time_s': d.time, 'max_measured_joint_speed_rad_s': max_speed,
              'start_joints_rad': arm.home.tolist(), 'final_joints_rad': d.qpos[arm.qadr].tolist(),
              'mujoco_version': mujoco.__version__, 'object_attached': False, **arm.placement_metrics()}
    np.savez_compressed(output / 'trace.npz', sim_qpos=np.array(frames), actions=np.array(actions),
                        object_xyz=np.array(objects), finger_contacts=np.array(contacts), phase=np.array(phases), timestamps=np.array(times))
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--no-video', action='store_true', help='Skip rendering for physics-only checks')
    parser.add_argument('--show-spawn-region', action='store_true', help='Show the position rectangle in the live Viewer only')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.headless:
        os.environ.setdefault('MUJOCO_GL', 'egl')
    arm = AGVDemo()
    arm.reset()  # Show the Z pose from the first Viewer frame.
    output = args.output or ROOT / 'outputs/agv_dummyx' / datetime.now().strftime('demo_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    if args.headless:
        context = nullcontext(None)
    else:
        import mujoco.viewer
        context = mujoco.viewer.launch_passive(arm.model, arm.data)
    with context as viewer:
        success = run(arm, output, viewer, video=not args.no_video, realtime=not args.headless,
                      show_spawn_region=args.show_spawn_region)
    print(f'Output: {output}')
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
