"""Run a plain pi0 policy on the obstacle-free AGV screwdriver scene."""

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/collect"))
sys.path.insert(0, str(ROOT / "scripts/tools"))

from collect_hardstop_screwdriver import FPS, HardstopArm, INSTRUCTION, sample_pose
from agv_spawn_region import load_spawn_region


def run_episode(arm, policy, fixed_renderer, wrist_renderer, render_option,
                rng, fixed_eval, max_steps, chunk_steps, viewer=None, realtime=False,
                task_prompt=INSTRUCTION):
    arm.reset()
    region = load_spawn_region(arm.model, arm.data)
    xy, yaw = sample_pose(rng, region, fixed_eval)
    home = arm.reset(xy, yaw)
    for _ in range(FPS):
        arm.step(home)

    states, actions, object_xyz = [], [], []
    grasp_seen = False
    retained_frames = 0
    max_height = float(arm.data.body("real_screwdriver").xpos[2])
    initial_height = max_height
    reason = "Maximum control steps reached"

    try:
        while len(actions) < max_steps:
            if viewer is not None and not viewer.is_running():
                reason = "Viewer closed"
                break
            fixed_renderer.update_scene(arm.data, camera="overview", scene_option=render_option)
            fixed = fixed_renderer.render().copy()
            wrist_renderer.update_scene(arm.data, camera="wrist_cam", scene_option=render_option)
            wrist = wrist_renderer.render().copy()
            result = policy.infer({"observation/image": fixed,
                                   "observation/wrist_image": wrist,
                                   "observation/state": arm.state().astype(np.float32),
                                   "prompt": task_prompt})
            chunk = np.asarray(result["actions"])
            if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) == 0 or not np.isfinite(chunk).all():
                raise ValueError(f"Expected finite policy actions with shape (N, 7), got {chunk.shape}")

            for action in chunk[:min(chunk_steps, max_steps - len(actions))]:
                started = time.monotonic()
                if viewer is not None and not viewer.is_running():
                    reason = "Viewer closed"
                    break
                action = action.copy()
                action[:6] = np.clip(action[:6], arm.limits[:, 0], arm.limits[:, 1])
                action[6] = np.clip(action[6], 0., .101)
                state = arm.state().copy()
                arm.step(action)
                states.append(state)
                actions.append(action.copy())
                object_xyz.append(arm.data.body("real_screwdriver").xpos.copy())
                grasp_seen |= arm.finger_contacts() == 2
                max_height = max(max_height, float(object_xyz[-1][2]))
                metrics = arm.placement_metrics()
                valid = (metrics["inside_bin"] and metrics["height_ok"] and metrics["bin_contact"]
                         and metrics["object_speed_m_s"] < .03 and metrics["finger_contacts"] == 0
                         and metrics["gripper_width_m"] > .08)
                retained_frames = retained_frames + 1 if valid else 0
                if viewer is not None:
                    viewer.sync()
                if realtime:
                    time.sleep(max(0., 1 / FPS - (time.monotonic() - started)))
                if retained_frames >= FPS and grasp_seen and max_height >= initial_height + .05:
                    reason = "success"
                    break
            if reason in ("success", "Viewer closed"):
                break
    except (RuntimeError, ValueError, KeyError) as exc:
        reason = str(exc)

    return ({"success": reason == "success", "reason": reason,
             "task_prompt": task_prompt,
             "steps": len(actions), "initial_xy": xy.tolist(), "initial_yaw": float(yaw),
             "max_lift_m": max_height - initial_height, "two_finger_contact_seen": grasp_seen,
             "retention_frames": retained_frames, **arm.placement_metrics()},
            np.asarray(states, dtype=np.float32).reshape(-1, 7),
            np.asarray(actions, dtype=np.float32).reshape(-1, 7),
            np.asarray(object_xyz, dtype=np.float32).reshape(-1, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--chunk_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_prompt", default=INSTRUCTION,
                        help="Task instruction sent to pi0; defaults to the original wording")
    parser.add_argument("--fixed_eval", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/agv_dummyx/pi0_inference")
    args = parser.parse_args()
    if min(args.num_episodes, args.max_steps, args.chunk_steps) < 1:
        parser.error("num_episodes, max_steps and chunk_steps must be positive")
    if args.headless:
        os.environ["MUJOCO_GL"] = "egl"

    from openpi_client import websocket_client_policy
    arm = HardstopArm()
    if arm.mj.mj_name2id(arm.model, arm.mj.mjtObj.mjOBJ_BODY, "dynamic_pillar") >= 0:
        raise RuntimeError("Plain pi0 evaluation requires the obstacle-free collection scene")
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    fixed_renderer = arm.mj.Renderer(arm.model, height=360, width=640)
    wrist_renderer = arm.mj.Renderer(arm.model, height=256, width=256)
    render_option = arm.mj.MjvOption()
    render_option.geomgroup[3:] = 0
    output = args.output / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    context = nullcontext(None)
    if not args.headless:
        import mujoco.viewer
        context = mujoco.viewer.launch_passive(arm.model, arm.data)

    try:
        with context as viewer:
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.lookat[:] = [.48, 0., .98]
                    viewer.cam.distance = 2.7
                    viewer.cam.azimuth = 145
                    viewer.cam.elevation = -27
                    viewer.opt.geomgroup[3:] = 0
            for episode in range(args.num_episodes):
                result, states, actions, objects = run_episode(
                    arm, policy, fixed_renderer, wrist_renderer, render_option, rng,
                    args.fixed_eval, args.max_steps, args.chunk_steps, viewer, args.realtime,
                    args.task_prompt)
                folder = output / f"ep_{episode:04d}"
                folder.mkdir()
                (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                np.savez_compressed(folder / "trace.npz", states=states, actions=actions,
                                    object_xyz=objects)
                print(f"Episode {episode + 1}/{args.num_episodes}: {result['reason']} "
                      f"({result['steps']} steps)", flush=True)
                if result["reason"] == "Viewer closed":
                    break
    finally:
        fixed_renderer.close()
        wrist_renderer.close()
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
