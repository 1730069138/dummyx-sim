"""Load the copied AGV base with the new arm for kinematic MuJoCo inspection.

Default: Z pose, open gripper, fixed AGV. No physics stepping or
collection. --pose zero shows the source all-zero contact-inspection pose.
"""
import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/agv_dummyx/scene.xml"
OBSTACLE_MODEL_PATH = MODEL_PATH.parent / "scene_obstacle.xml"
POSE_DIR = ROOT / "models/arm_description/config"


def load_model(pose="z", obstacle=False):
    import mujoco
    scene_path = OBSTACLE_MODEL_PATH if obstacle else MODEL_PATH
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    filename = "straight7_pose.json" if pose == "straight7" else "initial_pose.json"
    pose_path = MODEL_PATH.parent / "z_pose.json" if pose == "z" else POSE_DIR / filename
    positions = json.loads(pose_path.read_text())["joint_positions"]
    for name, value in positions.items():
        data.joint(name).qpos[:] = value
    # Match servo targets to the inspection pose if the model is later stepped.
    for index in range(model.nu):
        joint_id = model.actuator_trnid[index, 0]
        adr = model.jnt_qposadr[joint_id]
        data.ctrl[index] = data.qpos[adr] - model.qpos0[adr]
    mujoco.mj_forward(model, data)
    return model, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", choices=("z", "straight7", "zero"), default="z")
    parser.add_argument("--headless", action="store_true", help="Render one PNG and exit")
    parser.add_argument("--obstacle", action="store_true", help="Load the optional inference-test obstacle")
    parser.add_argument("--hide-spawn-region", action="store_true", help="Hide the viewer-only position rectangle")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    model, data = load_model(args.pose, obstacle=args.obstacle)
    from agv_spawn_region import load_spawn_region, draw_spawn_region, update_viewer_spawn_region
    region = None if args.hide_spawn_region else load_spawn_region(model, data)
    print(f"Model: {OBSTACLE_MODEL_PATH if args.obstacle else MODEL_PATH}")
    print(f"Obstacle: {'enabled (inference-test scene)' if args.obstacle else 'absent (default scene)'}")
    print(f"Pose: {args.pose}; AGV fixed; kinematic preview (physics paused)")
    print(f"AGV origin: {data.body('base_link').xpos}")
    print(f"Arm mounting origin: {data.body('link0').xpos}")
    print(f"TCP: {data.site('tcp_site').xpos}")
    if region is not None:
        print(f"Debug spawn rectangle: X={region['x_range_m']}, Y={region['y_range_m']} m (not in camera observations)")

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.45, 0., 0.85]
    camera.distance = 3.5
    camera.azimuth = 125
    camera.elevation = -20
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0  # hide collision/debug meshes

    if args.headless:
        import cv2
        args.output = args.output or ROOT / "outputs/agv_dummyx" / ("preview_obstacle.png" if args.obstacle else "preview.png")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with mujoco.Renderer(model, height=900, width=1200) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=option)
            if region is not None:
                draw_spawn_region(renderer.scene, region)
            image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(args.output), image):
                raise OSError(f"Could not write {args.output}")
        print(f"Preview saved: {args.output}")
        return

    import mujoco.viewer
    print("Drag the mouse to rotate/pan; scroll to zoom. Close the window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.azimuth = camera.azimuth
            viewer.cam.elevation = camera.elevation
            viewer.opt.geomgroup[:] = option.geomgroup
        while viewer.is_running():
            # Deliberately leave physics paused so inspection cannot change pose.
            update_viewer_spawn_region(viewer, region)
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
