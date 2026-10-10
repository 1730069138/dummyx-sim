"""Run pi0 on the AGV screwdriver scene, with optional visual obstacle avoidance."""

import argparse
import csv
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import subprocess
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/collect"))
sys.path.insert(0, str(ROOT / "scripts/tools"))

from collect_hardstop_screwdriver import FPS, HardstopArm, INSTRUCTION, model_manifest, sample_pose
from agv_spawn_region import load_spawn_region


HOME_TOLERANCE_RAD = 0.20
COMPARISON_MODES = ("apf", "vlsa", "no_obstacle", "obstacle")
MODE_LABELS = {"no_obstacle": "无障碍", "obstacle": "有障碍", "apf": "有障碍＋APF",
               "vlsa": "有障碍＋VLSA（共享感知）"}


def comparison_scene_signature(scene, arm):
    tree = ET.parse(scene).getroot()
    tree.attrib.pop("model", None)
    for parent in tree.iter():
        for child in list(parent):
            if child.tag == "include" and child.get("file") == "obstacle.xml":
                parent.remove(child)
    if arm.pillar_body >= 0:
        joint = int(arm.model.body_jntadr[arm.pillar_body])
        address = int(arm.model.jnt_qposadr[joint])
        for key in tree.findall("keyframe/key"):
            positions = key.get("qpos", "").split()
            key.set("qpos", " ".join(positions[:address] + positions[address + 7:]))
    def node(element):
        return [element.tag, sorted(element.attrib.items()), (element.text or "").strip(),
                [node(child) for child in element]]
    return hashlib.sha256(json.dumps(node(tree)).encode()).hexdigest()


class CollisionTrackedArm(HardstopArm):
    """Observe pillar contact and tilt at every physics substep during inference."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pillar_geom = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_GEOM, "pillar_geom")
        self.pillar_body = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, "dynamic_pillar")
        roots = {self.model.body("link0").id, self.target_body}
        self.collision_geoms = set()
        for geom in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[geom])
            while body:
                if body in roots:
                    self.collision_geoms.add(geom)
                    break
                body = int(self.model.body_parentid[body])
        self.tracking_collisions = False
        self.contact_collision = False
        self.pillar_tilted = False

    def start_collision_recording(self):
        self.contact_collision = False
        self.pillar_tilted = False
        self.tracking_collisions = self.pillar_geom >= 0

    def observe_collision(self):
        for contact in self.data.contact:
            if self.pillar_geom not in contact.geom or contact.dist > 0:
                continue
            other = int(contact.geom[1] if contact.geom[0] == self.pillar_geom else contact.geom[0])
            if other in self.collision_geoms:
                self.contact_collision = True
        self.pillar_tilted |= bool(self.data.xmat[self.pillar_body, 8] < .9)

    def step(self, action):
        if not self.tracking_collisions:
            return super().step(action)
        for _ in range(self.steps):
            self.command(action)
            self.mj.mj_step(self.model, self.data)
            self.observe_collision()
        self.mj.mj_forward(self.model, self.data)
        self.observe_collision()
        if not np.all(np.isfinite(self.data.qpos)) or np.any(self.data.warning.number):
            raise RuntimeError("MuJoCo reported a numerical warning or non-finite state")


def segment_to_segment_distance(p1, p2, p3, p4):
    p1, p2, p3, p4 = map(np.array, (p1, p2, p3, p4))
    d1, d2, r = p2-p1, p4-p3, p1-p3
    a, e, f = np.dot(d1, d1), np.dot(d2, d2), np.dot(d2, r)
    if a <= 1e-6 and e <= 1e-6: s, t = 0.0, 0.0
    elif a <= 1e-6: s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e <= 1e-6: t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom != 0.0 else 0.0
            t = (b * s + f) / e
            if t < 0.0: t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0: t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    c1, c2 = p1 + d1 * s, p3 + d2 * t
    vec = c1 - c2
    dist = np.linalg.norm(vec)
    normal = vec / dist if dist > 1e-6 else np.array([1.0, 0.0, 0.0])
    return dist, normal, c1, c2


def damped_pinv(J, rho=0.10):
    return J.T @ np.linalg.inv(J @ J.T + rho**2 * np.eye(J.shape[0]))


ARM_CAPSULE_R = 0.07
SCREWDRIVER_HALF_LENGTH = 0.12
SCREWDRIVER_CAPSULE_R = 0.02
APF_MAX_DISTANCE_M = 0.01
APF_ALGORITHM = "legacy_screwdriver_groundingdino_apf_v2"


def apf_end_effector_capsules(arm, action):
    """Build the same wrist/tool geometry for APF and its viewer overlay."""
    model, data = arm.model, arm.data
    # Legacy input is one finger's opening; AGV input is the total opening.
    gripper_val = 0.04 if action[6] / 2.0 > 0.02 else 0.0
    tcp_site = model.site("tcp_site").id
    tcp_pos = data.site_xpos[tcp_site].copy()
    tcp_body_id = int(model.site_bodyid[tcp_site])
    capsules = [{"p1": data.body("link6").xpos.copy(), "p2": tcp_pos,
                 "r": ARM_CAPSULE_R, "body_id": tcp_body_id}]
    tool = data.body("real_screwdriver")
    if gripper_val < 0.02 and np.linalg.norm(tcp_pos - tool.xpos) < 0.05:
        direction = tool.xmat.reshape(3, 3)[:, 1]
        capsules.append({"p1": tool.xpos - direction * SCREWDRIVER_HALF_LENGTH,
                         "p2": tool.xpos + direction * SCREWDRIVER_HALF_LENGTH,
                         "r": SCREWDRIVER_CAPSULE_R, "body_id": tcp_body_id})
    return capsules


def draw_apf_capsules(arm, viewer, obstacle, action):
    """Add visual-only capsules to the passive viewer, outside policy images."""
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        if obstacle is None:
            return
        capsules = [obstacle, *apf_end_effector_capsules(arm, action)]
        colors = ([1., 1., 0., .3], [0., .5, 1., .3], [1., 0., 1., .3])
        for capsule, color in zip(capsules, colors):
            geom = scene.geoms[scene.ngeom]
            arm.mj.mjv_initGeom(geom, arm.mj.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
            arm.mj.mjv_connector(geom, arm.mj.mjtGeom.mjGEOM_CAPSULE,
                                 capsule["r"], capsule["p1"], capsule["p2"])
            scene.ngeom += 1


def process_apf_action(arm, action, capsule, step_index, initial_tool_height, timing=None):
    """Port the legacy GroundingDINO APF, adapting AGV coordinates and indices."""
    if capsule is None:
        raise RuntimeError("Grounding DINO geometry is unavailable; refusing unprotected APF execution")
    model, data = arm.model, arm.data
    q = arm.state()[:6]
    q_dot_vla = action[:6] - q
    tcp_site = model.site("tcp_site").id
    tcp_pos = data.site_xpos[tcp_site].copy()
    dist_to_box = np.linalg.norm(tcp_pos - np.asarray(capsule["placement_goal"]))
    dynamic_safe_margin = np.clip(0.025 + dist_to_box * 0.4, 0.025, APF_MAX_DISTANCE_M)
    active_capsules = apf_end_effector_capsules(arm, action)
    tool = data.body("real_screwdriver")

    # Legacy 0.235 m pickup gate is 0.015 m above its initial tool height.
    obstacle_capsule = capsule if tool.xpos[2] > initial_tool_height + 0.015 else None
    delta_q_apf = np.zeros(6)
    if obstacle_capsule is not None:
        min_dist = float('inf')
        dom_normal, dom_body_id = None, None
        for cap in active_capsules:
            dist, normal, _, _ = segment_to_segment_distance(
                cap["p1"], cap["p2"], obstacle_capsule["p1"], obstacle_capsule["p2"])
            actual_dist = dist - cap["r"] - obstacle_capsule["r"]
            if actual_dist < min_dist:
                min_dist, dom_normal, dom_body_id = actual_dist, normal, cap["body_id"]
        if min_dist < dynamic_safe_margin:
            goal_decay = np.clip((dist_to_box - 0.035) / 0.065, 0.0, 1.0)
            start_fade = np.clip(step_index / 15.0, 0.0, 1.0)
            influence = ((dynamic_safe_margin - max(min_dist, 0.0)) / dynamic_safe_margin) ** 2 * goal_decay * start_fade
            if influence > 0.005:
                jacobian = np.zeros((3, model.nv))
                arm.mj.mj_jacBody(model, data, jacobian, None, dom_body_id)
                J_dom = jacobian[:, arm.dadr]
                v_vla_cartesian = J_dom @ q_dot_vla
                v_into_obs = np.dot(v_vla_cartesian, dom_normal)
                v_vla_filtered = v_vla_cartesian - v_into_obs * dom_normal if v_into_obs < 0 else v_vla_cartesian
                v_up = np.array([0.0, 0.0, 1.0])
                if np.dot(v_up, dom_normal) < 0:
                    v_up = v_up - np.dot(v_up, dom_normal) * dom_normal
                if np.linalg.norm(v_up) > 1e-4:
                    v_up /= np.linalg.norm(v_up)
                v_in = data.body("link0").xpos.copy() - tcp_pos
                v_in[2] = 0.0
                if np.linalg.norm(v_in) > 1e-4:
                    v_in /= np.linalg.norm(v_in)
                if np.dot(v_in, dom_normal) < 0:
                    v_in = v_in - np.dot(v_in, dom_normal) * dom_normal
                if np.linalg.norm(v_in) > 1e-4:
                    v_in /= np.linalg.norm(v_in)
                v_avoid_dir = v_up * 0.045 + v_in * 0.02
                v_vla_filtered += v_avoid_dir * influence
                v_vla_filtered += dom_normal * (influence * 0.01)
                delta_v = v_vla_filtered - v_vla_cartesian
                delta_q_apf = damped_pinv(J_dom, 0.05) @ delta_v

    q_dot_combined = q_dot_vla + delta_q_apf
    q_dot_smooth = 0.6 * q_dot_combined + 0.4 * getattr(arm, "apf_last_q_dot", np.zeros(6))
    arm.apf_last_q_dot = q_dot_smooth.copy()
    adjusted = action.copy()
    adjusted[:6] = np.clip(q + np.clip(q_dot_smooth, -0.25, 0.25),
                           arm.limits[:, 0], arm.limits[:, 1])
    return adjusted


def run_episode(arm, policy, fixed_renderer, wrist_renderer, render_option,
                rng, fixed_eval, max_steps, chunk_steps, viewer=None, realtime=False,
                task_prompt=INSTRUCTION, perception=None, perception_dir=None, timings=None,
                safety_mode="apf"):
    if isinstance(arm, CollisionTrackedArm):
        arm.tracking_collisions = False
    arm.reset()
    region = load_spawn_region(arm.model, arm.data)
    xy, yaw = sample_pose(rng, region, fixed_eval)
    home = arm.reset(xy, yaw)
    for _ in range(FPS):
        arm.step(home)

    if isinstance(arm, CollisionTrackedArm):
        arm.start_collision_recording()

    state_spec = arm.mj.mjtState.mjSTATE_INTEGRATION
    def snapshot():
        value = np.empty(arm.mj.mj_stateSize(arm.model, state_spec))
        arm.mj.mj_getState(arm.model, arm.data, value, state_spec)
        return value

    states, actions, object_xyz = [], [], []
    policy_actions, request_indices = [], []
    request_index = 0
    simulation_states = [snapshot()]
    measured_states = [arm.state().copy()]
    grasp_seen = False
    retained_frames = 0
    placement_frames = 0
    max_height = float(arm.data.body("real_screwdriver").xpos[2])
    initial_height = max_height
    reason = "timeout"
    vlsa = None
    try:
        capsule = None
        if perception is not None:
            if viewer is not None:
                draw_apf_capsules(arm, viewer, None, home)
            try:
                capsule, _ = perception(arm, perception_dir)
            except Exception as exc:
                raise RuntimeError(f"Perception failed before policy actions: {exc}") from exc
            if safety_mode == "vlsa":
                from agv_vlsa import SharedPerceptionVLSA
                vlsa = SharedPerceptionVLSA(arm, capsule, 1 / FPS)
                if perception_dir is not None:
                    (Path(perception_dir) / "vlsa_ellipsoids.json").write_text(
                        json.dumps(vlsa.geometry_metadata(), indent=2) + "\n")
            if viewer is not None:
                if vlsa is not None:
                    vlsa.draw(viewer)
                else:
                    draw_apf_capsules(arm, viewer, capsule, home)
                viewer.sync()
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
                step_timing = {} if timings is not None else None
                safety_time_ms = float("nan")
                if perception is not None:
                    safety_started = time.perf_counter() if timings is not None else None
                    action = (vlsa.process(action, step_timing) if vlsa is not None else
                              process_apf_action(arm, action, capsule, len(actions), initial_height, step_timing))
                    if timings is not None:
                        safety_time_ms = (time.perf_counter() - safety_started) * 1000
                arm.step(action)
                states.append(state)
                actions.append(action.copy())
                if timings is not None:
                    timings["safety_layer_time_ms"].append(safety_time_ms)
                    timings["solver_time_ms"].append(step_timing.get("solver_time_ms", float("nan")))
                policy_actions.append(raw_action)
                request_indices.append(request_index)
                object_xyz.append(arm.data.body("real_screwdriver").xpos.copy())
                simulation_states.append(snapshot())
                measured_states.append(arm.state().copy())
                grasp_seen |= arm.finger_contacts() == 2
                max_height = max(max_height, float(object_xyz[-1][2]))
                metrics = arm.placement_metrics()
                valid = (metrics["inside_bin"] and metrics["height_ok"] and metrics["bin_contact"]
                         and metrics["object_speed_m_s"] < .03 and metrics["finger_contacts"] == 0
                         and metrics["gripper_width_m"] > .08)
                placement_frames = placement_frames + 1 if valid else 0
                home_error = float(np.max(np.abs(arm.state()[:6] - home[:6])))
                home_valid = valid and home_error <= HOME_TOLERANCE_RAD
                retained_frames = retained_frames + 1 if home_valid else 0
                if viewer is not None:
                    if vlsa is not None:
                        vlsa.draw(viewer)
                    elif capsule is not None:
                        draw_apf_capsules(arm, viewer, capsule, action)
                    viewer.sync()
                if realtime:
                    time.sleep(max(0., 1 / FPS - (time.monotonic() - started)))
                if retained_frames >= FPS // 2 and grasp_seen and max_height >= initial_height + .05:
                    reason = "success"
                    break
            if reason in ("success", "Viewer closed"):
                break
            request_index += 1
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        reason = str(exc)

    return ({"success": reason == "success", "reason": reason,
             "task_prompt": task_prompt,
             "steps": len(actions), "initial_xy": xy.tolist(), "initial_yaw": float(yaw),
             "max_lift_m": max_height - initial_height, "two_finger_contact_seen": grasp_seen,
             "home_joints_rad": home[:6].tolist(),
             "home_max_error_rad": float(np.max(np.abs(arm.state()[:6] - home[:6]))),
             "returned_home": bool(np.max(np.abs(arm.state()[:6] - home[:6])) <= HOME_TOLERANCE_RAD),
             "retention_frames": retained_frames, "placement_retention_frames": placement_frames,
             "obstacle_present": isinstance(arm, CollisionTrackedArm) and arm.pillar_geom >= 0,
             "contact_collision": bool(getattr(arm, "contact_collision", False)),
             "pillar_tilted": bool(getattr(arm, "pillar_tilted", False)),
             **({"vlsa_qp_failures": vlsa.qp_failures,
                 "vlsa_min_h": vlsa.min_h if np.isfinite(vlsa.min_h) else None} if vlsa is not None else {}),
             **arm.placement_metrics()},
            np.asarray(states, dtype=np.float32).reshape(-1, 7),
            np.asarray(actions, dtype=np.float32).reshape(-1, 7),
            np.asarray(object_xyz, dtype=np.float32).reshape(-1, 3),
            np.asarray(simulation_states), np.asarray(measured_states),
            np.asarray(policy_actions, dtype=np.float32).reshape(-1, 7),
            np.asarray(request_indices, dtype=np.int64))


def write_summary(output, results, requested):
    fields = ["episode", "success", "reason", "steps", "initial_xy", "initial_yaw",
              "max_lift_m", "two_finger_contact_seen", "retention_frames",
              "home_max_error_rad", "returned_home",
              "obstacle_present", "contact_collision", "pillar_tilted"]
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    completed = [r for r in results if r["reason"] != "Viewer closed"]
    successes = sum(r["success"] for r in completed)
    n = len(completed)
    obstacle_runs = [r for r in completed if r.get("obstacle_present", False) and r["steps"] > 0]
    obstacle_n = len(obstacle_runs)
    collisions = sum(r["contact_collision"] for r in obstacle_runs)
    tilted = sum(r["pillar_tilted"] for r in obstacle_runs)
    summary = {"requested_episodes": requested, "recorded_episodes": len(results),
               "evaluated_episodes": n, "successes": successes,
               "success_rate": successes / n if n else None,
               "collision_evaluated_episodes": obstacle_n,
               "contact_collision_episodes": collisions,
               "contact_collision_rate": collisions / obstacle_n if obstacle_n else None,
               "pillar_tilted_episodes": tilted,
               "pillar_tilt_rate": tilted / obstacle_n if obstacle_n else None,
               "collision_denominator": "Obstacle episodes with executed actions, excluding manual viewer closure",
               "timeouts": sum(r["reason"] == "timeout" for r in completed),
               "errors": sum(r["reason"] not in ("success", "timeout") for r in completed),
               "interrupted_episodes": len(results) - n,
               "denominator": "All recorded episodes except manual viewer closure; runtime errors count as failures"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if obstacle_n:
        print(f"Pillar contact rate: {collisions}/{obstacle_n} ({collisions / obstacle_n:.1%}); "
              f"tilt rate: {tilted}/{obstacle_n} ({tilted / obstacle_n:.1%})", flush=True)
    print(f"Batch success: {successes}/{n}" + (f" ({successes / n:.1%})" if n else ""), flush=True)


def _comparison_statistics(completed, modes=COMPARISON_MODES):
    statistics = []
    for mode in modes:
        records = [entry for entry in completed if entry["evaluation_mode"] == mode]
        evaluated = [entry for entry in records if entry["reason"] != "Viewer closed"]
        obstacle = [entry for entry in evaluated if entry.get("obstacle_present") is True and entry.get("steps", 0) > 0]
        contacts = [entry for entry in obstacle if isinstance(entry.get("contact_collision"), bool)]
        tilts = [entry for entry in obstacle if isinstance(entry.get("pillar_tilted"), bool)]
        successes = sum(entry["success"] is True for entry in evaluated)
        contact_count = sum(entry["contact_collision"] for entry in contacts)
        tilt_count = sum(entry["pillar_tilted"] for entry in tilts)
        statistics.append({"evaluation_mode": mode, "recorded_episodes": len(records),
                           "evaluated_episodes": len(evaluated), "successes": successes,
                           "success_rate": successes / len(evaluated) if evaluated else None,
                           "collision_evaluated_episodes": len(contacts), "contact_collision_episodes": contact_count,
                           "contact_collision_rate": contact_count / len(contacts) if contacts else None,
                           "pillar_tilt_evaluated_episodes": len(tilts), "pillar_tilted_episodes": tilt_count,
                           "pillar_tilt_rate": tilt_count / len(tilts) if tilts else None})
    return statistics


def _save_comparison_progress(output, completed, requested_rounds, status, modes=COMPARISON_MODES):
    statistics = _comparison_statistics(completed, modes)
    temporary = output / ".schedule.json.tmp"
    temporary.write_text(json.dumps(
        {"requested_rounds": requested_rounds, "completed_episodes": len(completed),
         "status": status, "mode_order": modes, "episodes": completed, "statistics": statistics}, indent=2) + "\n")
    temporary.replace(output / "schedule.json")
    print(f"当前累计：已完成 {len(completed)}/{requested_rounds * len(modes)} 回合；状态：{status}", flush=True)
    for mode, row in zip(modes, statistics):
        label = MODE_LABELS[mode]
        rates = [f"{row[key]:.1%}" if row[key] is not None else "不适用"
                 for key in ("success_rate", "contact_collision_rate", "pillar_tilt_rate")]
        print(f"  {label}：有效回合 {row['evaluated_episodes']}，成功率 {rates[0]}，"
              f"接触率 {rates[1]}，倾斜率 {rates[2]}", flush=True)


def _comparison_entry(round_index, mode, seed, run):
    result = json.loads((run / "ep_0000/result.json").read_text())
    return {"round": round_index, "evaluation_mode": mode, "seed": seed, "run_path": str(run.resolve()),
            **{key: result.get(key) for key in
               ("success", "reason", "steps", "obstacle_present", "contact_collision", "pillar_tilted")}}


def _run_comparison_command(command):
    process = subprocess.Popen(command, start_new_session=True)
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def run_comparison_rounds(args):
    modes = tuple(getattr(args, "mode_order", COMPARISON_MODES))
    resume = getattr(args, "resume", None)
    if resume is None:
        output = args.output / datetime.now().strftime("comparison_%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True)
        (output / "run_config.json").write_text(json.dumps(
            {**vars(args), "output": str(output), "mode_order": modes,
             "apf_algorithm": APF_ALGORITHM,
             "requested_episodes": args.num_rounds * len(modes),
             "round_seed_definition": "seed + zero-based round index; identical seed for all modes in a round"},
            indent=2) + "\n")
        completed = []
    else:
        output = Path(resume).resolve()
        schedule = json.loads((output / "schedule.json").read_text())
        completed = [_comparison_entry(entry["round"], entry["evaluation_mode"], entry["seed"], Path(entry["run_path"]))
                     for entry in schedule["episodes"] if entry["reason"] != "Viewer closed"]
    completed_keys = {(entry["round"], entry["evaluation_mode"]) for entry in completed}
    if len(completed_keys) != len(completed):
        raise ValueError("Duplicate completed comparison episodes")
    _save_comparison_progress(output, completed, args.num_rounds, "running", modes)
    common = ["--host", args.host, "--port", str(args.port),
              "--max_steps", str(args.max_steps), "--chunk_steps", str(args.chunk_steps),
              "--task_prompt", args.task_prompt, "--device", args.device,
              "--obstacle_text", args.obstacle_text,
              "--box_threshold", str(args.box_threshold), "--text_threshold", str(args.text_threshold),
              "--groundingdino_config", args.groundingdino_config,
              "--groundingdino_checkpoint", args.groundingdino_checkpoint]
    for name in ("fixed_eval", "headless", "realtime", "record_experiment", "export_video"):
        if getattr(args, name):
            common.append("--" + name)
    common.append("--defer_plots")
    try:
        for round_index in range(args.num_rounds):
            round_output = output / f"round_{round_index:04d}"
            for mode in modes:
                label = MODE_LABELS[mode]
                if (round_index, mode) in completed_keys:
                    continue
                mode_output = round_output / mode
                previous = set(mode_output.glob("run_*"))
                recovered = [_comparison_entry(round_index, mode, args.seed + round_index, run)
                             for run in sorted(previous) if (run / "completed.json").is_file()]
                recovered = [entry for entry in recovered if entry["reason"] != "Viewer closed"]
                if len(recovered) > 1:
                    raise ValueError(f"Multiple completed runs for round {round_index}, mode {mode}")
                if recovered:
                    entry = recovered[0]
                else:
                    print(f"第 {round_index + 1}/{args.num_rounds} 轮：{label}", flush=True)
                    _run_comparison_command([sys.executable, str(Path(__file__).resolve()), *common,
                                             "--eval_mode", mode, "--num_episodes", "1",
                                             "--seed", str(args.seed + round_index), "--output", str(mode_output)])
                    run, = set(mode_output.glob("run_*")) - previous
                    entry = _comparison_entry(round_index, mode, args.seed + round_index, run)
                if entry["reason"] == "Viewer closed":
                    _save_comparison_progress(output, completed, args.num_rounds, "paused", modes)
                    print(f"轮询已停止；续跑命令：{sys.executable} {Path(__file__).resolve()} --resume {output}", flush=True)
                    return
                completed.append(entry)
                completed_keys.add((round_index, mode))
                _save_comparison_progress(output, completed, args.num_rounds, "running", modes)
                if entry["reason"].startswith("Perception failed"):
                    _save_comparison_progress(output, completed, args.num_rounds, "paused", modes)
                    print(f"轮询已停止：{entry['reason']}；已保存的数据：{output}", flush=True)
                    return
        if args.record_experiment:
            _run_comparison_command([sys.executable, str(Path(__file__).with_name("compare_agv_pi0_experiments.py").resolve()),
                                     "--comparison_run", str(output), "--output", str(output / "comparison")])
    except KeyboardInterrupt:
        _save_comparison_progress(output, completed, args.num_rounds, "paused", modes)
        print(f"轮询已暂停；被中断回合将从头重跑。\n续跑命令：{sys.executable} {Path(__file__).resolve()} --resume {output}", flush=True)
        return
    except subprocess.CalledProcessError:
        _save_comparison_progress(output, completed, args.num_rounds, "error", modes)
        raise
    _save_comparison_progress(output, completed, args.num_rounds, "complete", modes)
    print(f"轮询完成：{args.num_rounds} 轮，共 {len(completed)} 回合；结果：{output}", flush=True)


def _suppress_known_perception_warnings():
    known = (
        (FutureWarning, "Importing from timm.models.layers is deprecated"),
        (UserWarning, "torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument."),
        (FutureWarning, "The `device` argument is deprecated and will be removed in v5 of Transformers."),
        (UserWarning, "torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly."),
        (UserWarning, "None of the inputs have requires_grad=True. Gradients will be None"),
        (FutureWarning, "`torch.cuda.amp.autocast(args...)` is deprecated."),
    )
    for category, message in known:
        warnings.filterwarnings("ignore", message=re.escape(message), category=category)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--compare_modes", action="store_true",
                        help="Alternate apf, vlsa, no_obstacle and obstacle once per round, with shared GroundingDINO perception")
    parser.add_argument("--vlsa_only", action="store_true",
                        help="Run only VLSA rounds with seed + round index, to supplement an existing comparison")
    parser.add_argument("--resume", type=Path,
                        help="Resume a comparison directory using its original configuration")
    parser.add_argument("--num_rounds", type=int, default=1,
                        help="Comparison rounds; each round runs one episode of each mode")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--chunk_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_prompt", default=INSTRUCTION,
                        help="Task instruction sent to pi0; defaults to the original wording")
    parser.add_argument("--fixed_eval", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--record_experiment", action="store_true",
                        help="Save metrics, timing samples and replay states; plots are generated after all rounds in comparison mode")
    parser.add_argument("--export_video", action="store_true",
                        help="Also export overview and front-view videos for every completed episode; otherwise export selected episodes later with replay_agv_dummyx_pi0.py")
    parser.add_argument("--defer_plots", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--eval_mode", choices=("no_obstacle", "obstacle", "apf", "vlsa"),
                        help="Scene/controller mode; apf and vlsa share GroundingDINO geometry")
    parser.add_argument("--obstacle_source", choices=("none", "groundingdino"), default="none")
    parser.add_argument("--perception_probe", action="store_true",
                        help="Check obstacle localization without connecting to OpenPI")
    parser.add_argument("--groundingdino_config")
    parser.add_argument("--groundingdino_checkpoint")
    parser.add_argument("--obstacle_text", default="red pillar.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--box_threshold", type=float, default=.35)
    parser.add_argument("--text_threshold", type=float, default=.25)
    parser.add_argument("--output", type=Path,
                        help="Output root; defaults to 论文ppt材料 for --compare_modes, otherwise outputs/agv_dummyx/pi0_inference")
    args = parser.parse_args()
    if args.vlsa_only:
        args.compare_modes = True
        args.mode_order = ("vlsa",)
    if args.resume is not None:
        if any(value.startswith("--") and value.split("=", 1)[0] != "--resume" for value in sys.argv[1:]):
            parser.error("--resume uses the saved configuration; provide only --resume DIRECTORY")
        try:
            saved = json.loads((args.resume / "run_config.json").read_text())
            if saved.get("compare_modes") is not True:
                parser.error("--resume requires a comparison directory")
            if saved.get("apf_algorithm") != APF_ALGORITHM:
                parser.error("APF implementation differs from this experiment; start a new comparison to avoid mixing methods")
            for name in vars(args):
                if name != "resume" and name in saved:
                    setattr(args, name, saved[name])
            args.output = Path(saved["output"])
            args.mode_order = tuple(saved.get("mode_order", ("apf", "no_obstacle", "obstacle")))
            if (len(set(args.mode_order)) != len(args.mode_order) or
                    set(args.mode_order) not in ({"vlsa"}, {"apf", "no_obstacle", "obstacle"}, set(COMPARISON_MODES))):
                parser.error("Unrecognized saved comparison mode order")
            args.eval_mode = None
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
    if args.output is None:
        args.output = ROOT / ("论文ppt材料" if args.compare_modes else "outputs/agv_dummyx/pi0_inference")
    if min(args.num_episodes, args.max_steps, args.chunk_steps) < 1:
        parser.error("num_episodes, max_steps and chunk_steps must be positive")
    if args.max_steps > 5000:
        parser.error("max_steps must not exceed 5000")
    if args.num_rounds < 1:
        parser.error("num_rounds must be positive")
    if args.compare_modes and (args.eval_mode is not None or args.num_episodes != 1 or args.perception_probe):
        parser.error("--compare_modes uses --num_rounds and cannot combine with --eval_mode, --num_episodes other than 1, or --perception_probe")
    if not args.compare_modes and args.num_rounds != 1:
        parser.error("--num_rounds requires --compare_modes")
    if args.eval_mode is None:
        args.eval_mode = "apf" if args.obstacle_source == "groundingdino" else "no_obstacle"
    if args.eval_mode in ("apf", "vlsa"):
        args.obstacle_source = "groundingdino"
    elif args.obstacle_source == "groundingdino" and not args.perception_probe:
        parser.error("--obstacle_source groundingdino requires --eval_mode apf or vlsa")
    use_perception = args.compare_modes or args.eval_mode in ("apf", "vlsa") or args.perception_probe
    has_obstacle = args.eval_mode != "no_obstacle" or args.perception_probe
    if use_perception:
        if not args.groundingdino_config or not args.groundingdino_checkpoint:
            parser.error("Specify --groundingdino_config and --groundingdino_checkpoint")
        for path in (args.groundingdino_config, args.groundingdino_checkpoint):
            if not os.path.isfile(os.path.expanduser(path)):
                parser.error(f"File not found: {path}")
    if args.headless or args.perception_probe:
        os.environ["MUJOCO_GL"] = "egl"
    if args.compare_modes:
        if "vlsa" in getattr(args, "mode_order", COMPARISON_MODES):
            import agv_vlsa
            from deploy_screwdriver_client_vlsa import AEGISFullActionLayer
        run_comparison_rounds(args)
        return

    scene = ROOT / "models/agv_dummyx" / ("scene_obstacle.xml" if has_obstacle else "scene.xml")
    arm = CollisionTrackedArm(scene=scene)
    if not has_obstacle and arm.mj.mj_name2id(arm.model, arm.mj.mjtObj.mjOBJ_BODY, "dynamic_pillar") >= 0:
        raise RuntimeError("Plain pi0 evaluation requires the obstacle-free collection scene")
    perceive, perception_renderer = None, None
    if use_perception:
        _suppress_known_perception_warnings()
        from groundingdino_obstacle import GroundingDINOWrapper, localize_pillar
        detector = GroundingDINOWrapper(os.path.expanduser(args.groundingdino_config),
                                       os.path.expanduser(args.groundingdino_checkpoint), args.device)
        perception_renderer = arm.mj.Renderer(arm.model, height=480, width=640)

        def perceive(arm, evidence_dir):
            capsule, metadata = localize_pillar(arm.model, arm.data, perception_renderer, detector,
                                                args.obstacle_text, args.box_threshold, args.text_threshold,
                                                evidence_dir, camera_name="d415_rgb", table_z=.80, half_height=.08)
            # Like the legacy APF, use a known fixed placement target. This is
            # the AGV collection target; obstacle geometry comes only from DINO.
            box_pos = arm.data.body("drop_box").xpos
            capsule["placement_goal"] = np.array([box_pos[0] - .052, box_pos[1], .872])
            return capsule, metadata

    policy = None
    if not args.perception_probe:
        from openpi_client import websocket_client_policy
        policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    fixed_renderer = arm.mj.Renderer(arm.model, height=360, width=640)
    wrist_renderer = arm.mj.Renderer(arm.model, height=256, width=256)
    render_option = arm.mj.MjvOption()
    render_option.geomgroup[3:] = 0
    output = args.output / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    manifest = model_manifest()
    if has_obstacle:
        manifest[str(scene.relative_to(ROOT))] = hashlib.sha256(scene.read_bytes()).hexdigest()
    (output / "run_config.json").write_text(json.dumps(
        {**vars(args), "output": str(output), "fps": FPS,
         "scene": str(scene.relative_to(ROOT)),
         "evaluation_mode": args.eval_mode,
         "apf_algorithm": APF_ALGORITHM,
         "vlsa_algorithm": "shared_groundingdino_cartesian_cbf_qp_v1" if args.eval_mode == "vlsa" else None,
         "vlsa_reference": {
             "source": "scripts/deploy/deploy_screwdriver_client_vlsa.py",
             "source_sha256": hashlib.sha256((ROOT / "scripts/deploy/deploy_screwdriver_client_vlsa.py").read_bytes()).hexdigest(),
             "perception": "shared GroundingDINO capsule converted to enclosing ellipsoid; no VLM or MVEE",
             "action_space": "cartesian", "eef_ellipsoid_scale": .8,
             "max_linear_correction_m_s": .08, "max_angular_correction_rad_s": .08,
             "protects_carried_tool": False} if args.eval_mode == "vlsa" else None,
         "apf_reference": {
             "source": "scripts/deploy/deploy_screwdriver_client.py",
             "source_sha256": hashlib.sha256((ROOT / "scripts/deploy/deploy_screwdriver_client.py").read_bytes()).hexdigest(),
             "obstacle_source": "groundingdino"},
         "comparison_scene_signature": comparison_scene_signature(scene, arm),
         "timing_definitions": {
             "policy_inference_time_ms": "Client policy.infer round trip including encoding, network, server inference and decoding; per completed request",
             "safety_layer_time_ms": "Full APF or Cartesian VLSA call including kinematics and joint adaptation, excluding perception, policy request, rendering and physics; per executed safety control step",
             "solver_time_ms": "VLSA reference QP problem.solve only, including CVXPY canonicalization; APF has no optimization solver"},
         "model_manifest": manifest, "simulation_state_spec": int(arm.mj.mjtState.mjSTATE_INTEGRATION),
         "mujoco_version": arm.mj.__version__, "state_columns": [f"joint{i}_rad" for i in range(1, 7)] + ["gripper_width_m"],
         "success_criteria": {"requires_return_home": True, "home_tolerance_rad": HOME_TOLERANCE_RAD, "simultaneous_hold_frames": FPS // 2,
                              "minimum_lift_m": .05, "requires_two_finger_contact": True,
                              "placement": "inside_bin, height_ok, bin_contact, speed < 0.03 m/s, no finger contact, gripper > 0.08 m"},
         "apf_parameters": {"arm_capsule_radius_m": ARM_CAPSULE_R,
                            "tool_capsule_half_length_m": SCREWDRIVER_HALF_LENGTH,
                            "tool_capsule_radius_m": SCREWDRIVER_CAPSULE_R,
                            "pickup_height_above_initial_m": .015,
                            "tool_capsule_tcp_distance_m": .05,
                            "safe_margin_min_m": .025, "safe_margin_max_m": APF_MAX_DISTANCE_M,
                            "safe_margin_goal_distance_gain": .4,
                            "repulsion_fade_start_m": .10, "repulsion_fade_end_m": .035,
                            "start_fade_control_steps": 15,
                            "upward_gain": .045, "toward_base_gain": .02, "normal_gain": .01,
                            "pseudoinverse_damping": .05, "smoothing_beta": .6,
                            "joint_delta_limit_rad": .25,
                            "placement_goal": "AGV collection target: [drop_box.x - 0.052, drop_box.y, 0.872]",
                            "gripper_interface": "retain Pi0 total opening; use half of it for legacy capsule gating",
                            "joint_target_limits": "clip to source URDF limits"},
         "collision_criteria": {"contact": "pillar contact distance <= 0 with robot or screwdriver; excludes table",
                                "tilt": "pillar vertical-axis cosine < 0.9",
                                "sampling": "every physics substep during policy actions; excludes warmup",
                                "physics_timestep_s": float(arm.model.opt.timestep)},
         "trace_alignment": "states[t] before actions[t]; measured_states and simulation_states contain initial frame plus every post-action frame"},
        indent=2) + "\n")
    results = []
    if args.record_experiment and not args.perception_probe:
        from experiment_recording import export_episode, export_run_config, export_summary
        export_run_config(output, "screwdriver", vars(args), FPS, policy.get_server_metadata())
        experiment_metrics = []
    rng = np.random.default_rng(args.seed)
    context = nullcontext(None)
    if not args.headless and not args.perception_probe:
        import mujoco.viewer
        context = mujoco.viewer.launch_passive(arm.model, arm.data)

    try:
        if args.perception_probe:
            arm.reset()
            region = load_spawn_region(arm.model, arm.data)
            xy, yaw = sample_pose(rng, region, args.fixed_eval)
            home = arm.reset(xy, yaw)
            for _ in range(FPS):
                arm.step(home)
            evidence_dir = output / "perception_probe"
            capsule, _ = perceive(arm, evidence_dir)
            print(f"Localized obstacle: {capsule}; evidence: {evidence_dir}", flush=True)
            return
        with context as viewer:
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.lookat[:] = [.48, 0., .98]
                    viewer.cam.distance = 2.7
                    viewer.cam.azimuth = 145
                    viewer.cam.elevation = -27
                    viewer.opt.geomgroup[3:] = 0
            for episode in range(args.num_episodes):
                folder = output / f"ep_{episode:04d}"
                folder.mkdir()
                timings = {name: [] for name in ("policy_inference_time_ms", "safety_layer_time_ms", "solver_time_ms")} if args.record_experiment else None
                result, states, actions, objects, simulation_states, measured_states, raw_actions, request_indices = run_episode(
                    arm, policy, fixed_renderer, wrist_renderer, render_option, rng,
                    args.fixed_eval, args.max_steps, args.chunk_steps, viewer, args.realtime,
                    args.task_prompt, perceive if args.obstacle_source == "groundingdino" else None,
                    folder / "perception" if args.obstacle_source == "groundingdino" else None, timings,
                    safety_mode=args.eval_mode)
                result["episode"] = episode
                result["evaluation_mode"] = args.eval_mode
                if timings is not None:
                    for name, values in timings.items():
                        finite = np.asarray(values)[np.isfinite(values)]
                        prefix = name.removesuffix("_ms")
                        result[prefix + "_samples"] = len(finite)
                        result[prefix + "_mean_ms"] = float(np.mean(finite)) if len(finite) else None
                        result[prefix + "_p95_ms"] = float(np.percentile(finite, 95)) if len(finite) else None
                (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                np.savez_compressed(folder / "trace.npz", states=states, actions=actions,
                                    object_xyz=objects, simulation_states=simulation_states,
                                    measured_states=measured_states,
                                    policy_actions=raw_actions, request_indices=request_indices,
                                    simulation_state_spec=int(arm.mj.mjtState.mjSTATE_INTEGRATION),
                                    fps=FPS,
                                    **({name: np.asarray(values) for name, values in timings.items()} if timings is not None else {}))
                results.append(result)
                if not args.defer_plots:
                    write_summary(output, results, args.num_episodes)
                if args.record_experiment:
                    experiment_metrics.append(export_episode(
                        output, episode, "screwdriver", result, actions,
                        measured_states[1:len(actions) + 1], {"screwdriver": objects}, FPS,
                        plots=not args.defer_plots))
                    if not args.defer_plots:
                        export_summary(output, experiment_metrics, args.num_episodes)
                if args.export_video:
                    from replay_agv_dummyx_pi0 import export_video
                    export_video(folder, output / "experiment" / f"ep_{episode:04d}" / "inference_full.mp4")
                print(f"Episode {episode + 1}/{args.num_episodes}: {result['reason']} "
                      f"({result['steps']} steps)", flush=True)
                if result["reason"] == "Viewer closed" or result["reason"].startswith("Perception failed"):
                    break
    finally:
        fixed_renderer.close()
        wrist_renderer.close()
        if perception_renderer is not None:
            perception_renderer.close()
    if args.defer_plots:
        (output / "completed.json").write_text(json.dumps({"recorded_episodes": len(results)}) + "\n")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
