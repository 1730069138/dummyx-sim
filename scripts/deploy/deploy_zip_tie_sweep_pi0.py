"""Run a plain pi0 policy on the obstacle-free short-table zip-tie sweep scene."""

import argparse
import csv
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

from collect_zip_tie_sweep import CleanupArm, FPS, INSTRUCTIONS, SCENE, TIE_NAMES, model_manifest


HOME_TOLERANCE_RAD = 0.1


def run_episode(arm, policy, fixed_renderer, wrist_renderer, render_option,
                rng, fixed_eval, max_steps, chunk_steps, viewer=None, realtime=False,
                task_prompt=INSTRUCTIONS[0], timings=None):
    home = arm.reset(rng, fixed_eval)
    for _ in range(FPS):
        arm.step(home)

    state_spec = arm.mj.mjtState.mjSTATE_INTEGRATION
    def snapshot():
        value = np.empty(arm.mj.mj_stateSize(arm.model, state_spec))
        arm.mj.mj_getState(arm.model, arm.data, value, state_spec)
        return value

    initial_tie_xyz = [arm.data.body(name).xpos.tolist() for name in TIE_NAMES]
    initial_sponge_xyz = arm.data.body("cleanup_sponge").xpos.tolist()
    states, actions, tie_xyz, sponge_xyz = [], [], [], []
    policy_actions, request_indices = [], []
    request_index = 0
    simulation_states = [snapshot()]
    measured_states = [arm.state().copy()]
    grasp_seen = False
    robot_table_seen = False
    camera_min_height_after_grasp_m = float("inf")
    stable_success_frames = 0
    reason = "timeout"

    try:
        while len(actions) < max_steps:
            if viewer is not None and not viewer.is_running():
                reason = "Viewer closed"
                break
            fixed_renderer.update_scene(arm.data, camera="overview", scene_option=render_option)
            fixed = fixed_renderer.render().copy()
            wrist_renderer.update_scene(arm.data, camera="wrist_cam", scene_option=render_option)
            wrist = wrist_renderer.render().copy()
            inference_started = time.perf_counter() if timings is not None else None
            result = policy.infer({"observation/image": fixed,
                                   "observation/wrist_image": wrist,
                                   "observation/state": arm.state().astype(np.float32),
                                   "prompt": task_prompt})
            if timings is not None:
                timings["policy_inference_time_ms"].append((time.perf_counter() - inference_started) * 1000)
            chunk = np.asarray(result["actions"])
            if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) == 0 or not np.isfinite(chunk).all():
                raise ValueError(f"Expected finite policy actions with shape (N, 7), got {chunk.shape}")

            for action in chunk[:min(chunk_steps, max_steps - len(actions))]:
                started = time.monotonic()
                if viewer is not None and not viewer.is_running():
                    reason = "Viewer closed"
                    break
                action = action.copy()
                raw_action = action.copy()
                action[:6] = np.clip(action[:6], arm.limits[:, 0], arm.limits[:, 1])
                action[6] = np.clip(action[6], 0., .101)
                state = arm.state().copy()
                arm.step(action)
                states.append(state)
                actions.append(action.copy())
                policy_actions.append(raw_action)
                request_indices.append(request_index)
                simulation_states.append(snapshot())
                measured_states.append(arm.state().copy())
                tie_xyz.append([arm.data.body(name).xpos.copy() for name in TIE_NAMES])
                sponge_xyz.append(arm.data.body("cleanup_sponge").xpos.copy())
                grasp_seen |= arm.finger_contacts() == 2
                robot_table_seen |= arm.last_contact["robot_table_contact"]
                metrics = arm.result_metrics()
                if grasp_seen and not metrics["sponge_returned"]:
                    camera_z = arm.data.cam_xpos[arm.model.camera("wrist_cam").id, 2]
                    finger_z = max(arm.data.body("link10").xpos[2],
                                   arm.data.body("link12").xpos[2])
                    camera_min_height_after_grasp_m = min(
                        camera_min_height_after_grasp_m, float(camera_z - finger_z))
                valid = (grasp_seen and metrics["collected_count"] == len(TIE_NAMES)
                         and metrics["sponge_returned"]
                         and metrics["home_joint_max_error_rad"] <= HOME_TOLERANCE_RAD
                         and abs(arm.state()[6] - .101) < .005
                         and camera_min_height_after_grasp_m > .005
                         and not robot_table_seen)
                stable_success_frames = stable_success_frames + 1 if valid else 0
                if viewer is not None:
                    viewer.sync()
                if realtime:
                    time.sleep(max(0., 1 / FPS - (time.monotonic() - started)))
                if robot_table_seen:
                    reason = "Robot arm touched the tabletop"
                    break
                if stable_success_frames >= FPS // 2:
                    reason = "success"
                    break
            if reason in ("success", "Viewer closed", "Robot arm touched the tabletop"):
                break
            request_index += 1
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        reason = str(exc)

    metrics = arm.result_metrics()
    return ({"success": reason == "success", "reason": reason,
             "task_prompt": task_prompt, "steps": len(actions),
             "initial_tie_xyz": initial_tie_xyz,
             "initial_sponge_xyz": initial_sponge_xyz,
             "home_joints_rad": home[:6].tolist(),
             "home_max_error_rad": metrics["home_joint_max_error_rad"],
             "returned_home": metrics["home_joint_max_error_rad"] <= HOME_TOLERANCE_RAD,
             "two_finger_contact_seen": grasp_seen,
             "camera_min_height_after_grasp_m": camera_min_height_after_grasp_m
             if np.isfinite(camera_min_height_after_grasp_m) else None,
             "robot_table_contact": robot_table_seen,
             "stable_success_frames": stable_success_frames, **metrics},
            np.asarray(states, dtype=np.float32).reshape(-1, 7),
            np.asarray(actions, dtype=np.float32).reshape(-1, 7),
            np.asarray(tie_xyz, dtype=np.float32).reshape(-1, len(TIE_NAMES), 3),
            np.asarray(sponge_xyz, dtype=np.float32).reshape(-1, 3),
            np.asarray(simulation_states), np.asarray(measured_states),
            np.asarray(policy_actions, dtype=np.float32).reshape(-1, 7),
            np.asarray(request_indices, dtype=np.int64))


def write_summary(output, results, requested):
    fields = ["episode", "success", "reason", "steps", "initial_tie_xyz",
              "initial_sponge_xyz", "two_finger_contact_seen", "collected_count",
              "sponge_returned", "home_max_error_rad", "returned_home",
              "camera_min_height_after_grasp_m", "robot_table_contact", "stable_success_frames"]
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    completed = [r for r in results if r["reason"] != "Viewer closed"]
    successes = sum(r["success"] for r in completed)
    n = len(completed)
    summary = {"requested_episodes": requested, "recorded_episodes": len(results),
               "evaluated_episodes": n, "successes": successes,
               "success_rate": successes / n if n else None,
               "robot_table_contact_episodes": sum(r.get("robot_table_contact", False) for r in completed),
               "timeouts": sum(r["reason"] == "timeout" for r in completed),
               "errors": sum(r["reason"] not in ("success", "timeout", "Robot arm touched the tabletop") for r in completed),
               "interrupted_episodes": len(results) - n,
               "denominator": "All recorded episodes except manual viewer closure; runtime errors and robot-table contact count as failures"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Batch success: {successes}/{n}" + (f" ({successes / n:.1%})" if n else ""), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=3500)
    parser.add_argument("--chunk_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_prompt", default=INSTRUCTIONS[0])
    parser.add_argument("--fixed_eval", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--record_experiment", action="store_true",
                        help="Export experiment CSV metrics and PNG/PDF plots")
    parser.add_argument("--export_video", action="store_true",
                        help="Export full recorded-state overview and front-view MP4s")
    parser.add_argument("--defer_plots", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/agv_dummyx/zip_tie_sweep_pi0_inference")
    args = parser.parse_args()
    if min(args.num_episodes, args.max_steps, args.chunk_steps) < 1:
        parser.error("num_episodes, max_steps and chunk_steps must be positive")
    if args.max_steps > 5000:
        parser.error("max_steps must not exceed 5000")
    if args.headless:
        os.environ["MUJOCO_GL"] = "egl"

    from openpi_client import websocket_client_policy
    arm = CleanupArm()
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    fixed_renderer = arm.mj.Renderer(arm.model, height=360, width=640)
    wrist_renderer = arm.mj.Renderer(arm.model, height=256, width=256)
    render_option = arm.mj.MjvOption()
    render_option.geomgroup[3:] = 0
    output = args.output / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    (output / "run_config.json").write_text(json.dumps(
        {**vars(args), "output": str(output), "task": "zip_tie_sweep", "fps": FPS,
         "scene": str(SCENE.relative_to(ROOT)), "evaluation_mode": "no_obstacle",
         "model_manifest": model_manifest(),
         "simulation_state_spec": int(arm.mj.mjtState.mjSTATE_INTEGRATION),
         "mujoco_version": arm.mj.__version__,
         "state_columns": [f"joint{i}_rad" for i in range(1, 7)] + ["gripper_width_m"],
         "timing_definitions": {
             "policy_inference_time_ms": "Client policy.infer round trip including encoding, network, server inference and decoding; per completed request"},
         "success_criteria": {"requires_return_home": True, "home_tolerance_rad": HOME_TOLERANCE_RAD,
                              "simultaneous_hold_frames": FPS // 2,
                              "requires_two_finger_contact": True,
                              "collected_count": len(TIE_NAMES),
                              "ties_collected": "Whole outline off tabletop, using existing 1 mm edge tolerance",
                              "sponge_returned": "CleanupArm.result_metrics sponge_returned",
                              "gripper_target_m": .101, "gripper_tolerance_m": .005,
                              "camera_min_height_after_grasp_m": .005,
                              "requires_no_robot_table_contact": True},
         "trace_alignment": "states[t] before actions[t]; measured_states and simulation_states contain initial frame plus every post-action frame"},
        indent=2) + "\n")
    results = []
    if args.record_experiment:
        from experiment_recording import export_episode, export_run_config, export_summary
        export_run_config(output, "zip_tie_sweep", vars(args), FPS, policy.get_server_metadata())
        experiment_metrics = []
    rng = np.random.default_rng(args.seed)
    context = nullcontext(None)
    if not args.headless:
        import mujoco.viewer
        context = mujoco.viewer.launch_passive(arm.model, arm.data)

    try:
        with context as viewer:
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.lookat[:] = [.55, .10, .90]
                    viewer.cam.distance = 2.4
                    viewer.cam.azimuth = 145
                    viewer.cam.elevation = -35
                    viewer.opt.geomgroup[3:] = 0
            for episode in range(args.num_episodes):
                timings = {"policy_inference_time_ms": []} if args.record_experiment else None
                result, states, actions, ties, sponge, simulation_states, measured_states, raw_actions, request_indices = run_episode(
                    arm, policy, fixed_renderer, wrist_renderer, render_option, rng,
                    args.fixed_eval, args.max_steps, args.chunk_steps, viewer,
                    args.realtime, args.task_prompt, timings)
                result["episode"] = episode
                result["evaluation_mode"] = "no_obstacle"
                if timings is not None:
                    for name, values in timings.items():
                        finite = np.asarray(values)[np.isfinite(values)]
                        prefix = name.removesuffix("_ms")
                        result[prefix + "_samples"] = len(finite)
                        result[prefix + "_mean_ms"] = float(np.mean(finite)) if len(finite) else None
                        result[prefix + "_p95_ms"] = float(np.percentile(finite, 95)) if len(finite) else None
                folder = output / f"ep_{episode:04d}"
                folder.mkdir()
                (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                np.savez_compressed(folder / "trace.npz", states=states, actions=actions,
                                    tie_xyz=ties, sponge_xyz=sponge,
                                    simulation_states=simulation_states, measured_states=measured_states,
                                    policy_actions=raw_actions, request_indices=request_indices,
                                    simulation_state_spec=int(arm.mj.mjtState.mjSTATE_INTEGRATION), fps=FPS,
                                    **({name: np.asarray(values) for name, values in timings.items()} if timings is not None else {}))
                results.append(result)
                if not args.defer_plots:
                    write_summary(output, results, args.num_episodes)
                if args.record_experiment:
                    objects = {"sponge": sponge,
                               **{name: ties[:, i] for i, name in enumerate(TIE_NAMES)}}
                    experiment_metrics.append(export_episode(
                        output, episode, "zip_tie_sweep", result, actions,
                        measured_states[1:len(actions) + 1], objects, FPS,
                        plots=not args.defer_plots))
                    if not args.defer_plots:
                        export_summary(output, experiment_metrics, args.num_episodes)
                if args.export_video:
                    from replay_agv_dummyx_pi0 import export_video
                    export_video(folder, output / "experiment" / f"ep_{episode:04d}" / "inference_full.mp4")
                print(f"Episode {episode + 1}/{args.num_episodes}: {result['reason']} "
                      f"({result['steps']} steps)", flush=True)
                if result["reason"] == "Viewer closed":
                    break
    finally:
        fixed_renderer.close()
        wrist_renderer.close()
    if args.defer_plots:
        (output / "completed.json").write_text(json.dumps({"recorded_episodes": len(results)}) + "\n")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
