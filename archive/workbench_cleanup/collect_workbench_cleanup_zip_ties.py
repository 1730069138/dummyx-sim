"""Collect single-arm sponge sweeping demonstrations in the AGV workbench scene.

The expert grasps a free sponge, turns and sets it down, then regrips it with
the wrist camera facing forward. It sweeps three whole cable ties into a suspended
tray, returns the sponge to the pickup area, and moves the arm to Z home.
The cup is absent from collection; ``scene_cup.xml`` is reserved for later
obstacle evaluation. No
contact-solver force is used to choose an action. Contact loads are logged only
for offline evaluation.

State/action: joint1..joint6 [rad], total gripper opening [m]. Each action is
an absolute position-servo target held for the following 20 ms.
"""

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "archive/workbench_cleanup/workbench_cleanup_zip_ties/scene.xml"
ARM_SOURCE = ROOT / "models/arm_description"
FPS = 50
STATE_NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper_width"]
TIE_NAMES = [f"cleanup_tie_{i}" for i in range(3)]
INSTRUCTION = "Pick up and turn the sponge, set it down, rotate the wrist 180 degrees, regrasp with the camera above the gripper, sweep the whole cable ties into the tray, return the sponge, and move to Z home."
DOWN_ROTATION = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
ARC_RADIUS = .460
ARC_END_DEG = 75.
# Turn the pad's long edge nearly across the arc tangent while sweeping.
TOOL_YAW_OFFSET_DEG = 75.
MAX_CORRECTIVE_SWEEPS = 4
# Whole ties lie across world X and stay below world Y=0.30 m.
TIE_SPAWN_XY = (((.565, .585), (.025, .035)),
                ((.550, .570), (.195, .205)),
                ((.565, .585), (.270, .290)))
SPONGE_RETURN_XY = np.array([.415, -.160])
SPONGE_HANDOFF_XY = np.array([.450, -.050])
HANDOFF_YAW_DEG = 60.
WRIST_FLIP = Rotation.from_euler("z", 180, degrees=True).as_matrix()


def model_manifest():
    files = {ROOT / "models/agv_dummyx/z_pose.json",
             ARM_SOURCE / "scripts/gripper_kinematics.py"}
    pending = [SCENE]
    while pending:
        path = pending.pop().resolve()
        if path in files:
            continue
        files.add(path)
        if path.suffix == ".xml":
            for element in ET.parse(path).iter():
                if element.get("file"):
                    pending.append(path.parent / element.get("file"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def load_coupling():
    path = ARM_SOURCE / "scripts/gripper_kinematics.py"
    spec = importlib.util.spec_from_file_location("cleanup_gripper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CleanupArm:
    def __init__(self):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.joints = [self.model.joint(name) for name in STATE_NAMES[:6]]
        self.qadr = np.array([joint.qposadr[0] for joint in self.joints])
        self.limits = np.array([joint.range for joint in self.joints])
        self.servo = {i: self.model.actuator(f"servo_joint{i}").id
                      for i in (1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13)}
        self.coupling = load_coupling()
        self.crank_samples = np.linspace(0, self.coupling.LOWER, 4097)
        self.width_samples = np.array([-2 * self.coupling.coupled_positions(q)["joint10"]
                                       for q in self.crank_samples])
        self.steps = round(1 / FPS / self.model.opt.timestep)
        if not np.isclose(self.steps * self.model.opt.timestep, 1 / FPS):
            raise ValueError("Physics timestep must divide the recording interval")
        self.sponge_body = self.model.body("cleanup_sponge").id
        self.finger_bodies = {self.model.body("link10").id, self.model.body("link12").id}
        self.table_geom = self.model.geom("workbench_top").id
        robot_bodies = {self.model.body("base_link").id}
        for body_id in range(1, self.model.nbody):
            if int(self.model.body_parentid[body_id]) in robot_bodies:
                robot_bodies.add(body_id)
        self.robot_geoms = set(np.flatnonzero(np.isin(self.model.geom_bodyid,
                                                     list(robot_bodies))))
        self.pad_geom = self.model.geom("cleanup_sponge_pad").id
        self.tie_bodies = {self.model.body(name).id for name in TIE_NAMES}
        if self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, "cleanup_cup") >= 0:
            raise ValueError("Training scene must not contain the evaluation cup")
        self.home = None
        self.last_contact = {}
        self.total_pad_impulse = 0.
        self.peak_pad_force = 0.
        self.total_debris_table_impulse = 0.
        self.total_table_impulse = 0.
        self.peak_table_force = 0.

    def state(self):
        d = self.data
        width = -float(d.joint("joint10").qpos[0] + d.joint("joint12").qpos[0])
        return np.r_[d.qpos[self.qadr], width]

    def reset(self, rng, fixed_eval):
        self.mj.mj_resetData(self.model, self.data)
        initial = json.loads((ROOT / "models/agv_dummyx/z_pose.json").read_text())["joint_positions"]
        for name, value in initial.items():
            self.data.joint(name).qpos[:] = value
        if not fixed_eval:
            sponge = self.data.joint("fj_sponge").qpos
            sponge[:2] += rng.uniform([-.012, -.015], [.012, .015])
            for name, ((x_min, x_max), (y_min, y_max)) in zip(TIE_NAMES, TIE_SPAWN_XY):
                qpos = self.data.joint("fj_" + name.removeprefix("cleanup_")).qpos
                qpos[0] = rng.uniform(x_min, x_max)
                qpos[1] = rng.uniform(y_min, y_max)
        self.mj.mj_forward(self.model, self.data)
        self.home = self.data.qpos[self.qadr].copy()
        self.total_pad_impulse = self.peak_pad_force = self.total_debris_table_impulse = 0.
        self.total_table_impulse = self.peak_table_force = 0.
        self.last_contact = {"pad_table_normal_n": 0., "pad_table_impulse_ns": 0.,
                             "debris_table_impulse_ns": 0., "table_normal_n": 0.,
                             "table_impulse_ns": 0., "robot_table_contact": False}
        return np.r_[self.home, .101]

    def solve_ik(self, position, rotation, seed):
        d = self.ik_data
        d.qpos[:] = self.data.qpos

        def residual(q):
            d.qpos[self.qadr] = q
            self.mj.mj_forward(self.model, d)
            tcp = d.site("tcp_site")
            angular = Rotation.from_matrix(rotation @ tcp.xmat.reshape(3, 3).T).as_rotvec()
            return np.r_[5 * (tcp.xpos - position), angular]

        lower, upper = self.limits[:, 0] + .01, self.limits[:, 1] - .01
        result = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper),
                               max_nfev=60, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        error = residual(result.x)
        if np.linalg.norm(error[:3]) / 5 > .003 or np.linalg.norm(error[3:]) > .035:
            raise RuntimeError(f"IK unreachable at {np.round(position, 4).tolist()}")
        return result.x

    def command(self, action):
        if np.shape(action) != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("Expected seven finite action values")
        if np.any(action[:6] < self.limits[:, 0]) or np.any(action[:6] > self.limits[:, 1]):
            raise ValueError("Arm target outside joint limits")
        width = float(action[6])
        if not -1e-8 <= width <= .101 + 1e-8:
            raise ValueError("Gripper opening outside [0, 0.101] m")
        width = np.clip(width, 0., .101)
        for i in range(1, 7):
            self.data.ctrl[self.servo[i]] = action[i - 1]
        for i in (10, 12):
            self.data.ctrl[self.servo[i]] = -width / 2
        measured_width = np.clip(self.state()[6], 0, .101)
        crank = np.interp(measured_width, self.width_samples, self.crank_samples)
        self.data.ctrl[self.servo[7]] = crank
        coupled = self.coupling.coupled_positions(float(crank))
        for i in (11, 13):
            self.data.ctrl[self.servo[i]] = coupled[f"joint{i}"]

    def finger_contacts(self):
        fingers = set()
        for contact in self.data.contact:
            b1, b2 = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if b1 == self.sponge_body and b2 in self.finger_bodies:
                fingers.add(int(b2))
            if b2 == self.sponge_body and b1 in self.finger_bodies:
                fingers.add(int(b1))
        return len(fingers)

    def step(self, action):
        pad_impulse = debris_impulse = peak_pad = 0.
        table_impulse = peak_table = 0.
        robot_table = False
        wrench = np.zeros(6)
        for _ in range(self.steps):
            self.command(action)
            self.mj.mj_step(self.model, self.data)
            dt = self.model.opt.timestep
            pad_force = table_force = 0.
            for contact_index, contact in enumerate(self.data.contact):
                pair = {int(contact.geom1), int(contact.geom2)}
                if self.table_geom not in pair:
                    continue
                self.mj.mj_contactForce(self.model, self.data, contact_index, wrench)
                normal_force = max(float(wrench[0]), 0.)
                impulse = normal_force * dt
                table_force += normal_force
                table_impulse += impulse
                if self.pad_geom in pair:
                    pad_impulse += impulse
                    pad_force += normal_force
                elif int(self.model.geom_bodyid[contact.geom1]) in self.tie_bodies or \
                        int(self.model.geom_bodyid[contact.geom2]) in self.tie_bodies:
                    debris_impulse += impulse
                else:
                    other = pair - {self.table_geom}
                    if other & self.robot_geoms and contact.dist < -.001:
                        robot_table = True
            peak_pad = max(peak_pad, pad_force)
            peak_table = max(peak_table, table_force)
        self.mj.mj_forward(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)) or np.any(self.data.warning.number):
            raise RuntimeError("MuJoCo numerical warning or non-finite state")
        self.total_pad_impulse += pad_impulse
        self.total_debris_table_impulse += debris_impulse
        self.peak_pad_force = max(self.peak_pad_force, peak_pad)
        self.total_table_impulse += table_impulse
        self.peak_table_force = max(self.peak_table_force, peak_table)
        self.last_contact = {"pad_table_normal_n": peak_pad,
                             "pad_table_impulse_ns": pad_impulse,
                             "debris_table_impulse_ns": debris_impulse,
                             "table_normal_n": peak_table,
                             "table_impulse_ns": table_impulse,
                             "robot_table_contact": robot_table}

    def result_metrics(self):
        bottom = self.model.geom("cleanup_tray_bottom")
        tray = self.data.geom_xpos[bottom.id]
        lower = tray - bottom.size
        upper = tray + bottom.size
        # Allow 1 mm contact compression, but require the entire tie to be
        # inside the suspended tray and resting with low final velocity.
        lower[:2] += [.006, .006]
        upper[1] -= .006
        collected, speeds, ties = [], [], []
        for name in TIE_NAMES:
            geom = self.model.geom(name + "_geom")
            mesh_id = int(geom.dataid[0])
            first = self.model.mesh_vertadr[mesh_id]
            count = self.model.mesh_vertnum[mesh_id]
            vertices = self.model.mesh_vert[first:first + count]
            world_vertices = (vertices @ self.data.geom_xmat[geom.id].reshape(3, 3).T
                              + self.data.geom_xpos[geom.id])
            minimum = world_vertices.min(axis=0)
            maximum = world_vertices.max(axis=0)
            joint = self.model.joint("fj_" + name.removeprefix("cleanup_"))
            velocity = self.data.qvel[joint.dofadr[0]:joint.dofadr[0] + 6]
            speed = float(np.linalg.norm(velocity[:3]))
            inside = (np.all(minimum[:2] >= lower[:2] - .001)
                      and np.all(maximum[:2] <= upper[:2] + .001)
                      and minimum[2] >= upper[2] - .002
                      and maximum[2] < .799
                      and speed < .02 and np.linalg.norm(velocity[3:]) < .2)
            collected.append(bool(inside))
            speeds.append(speed)
            ties.append(((minimum + maximum) / 2).tolist())
        pad = self.model.geom("cleanup_sponge_pad")
        pad_center = self.data.geom_xpos[pad.id]
        pad_normal = self.data.geom_xmat[pad.id].reshape(3, 3)[:, 2]
        sponge_joint = self.model.joint("fj_sponge")
        sponge_velocity = self.data.qvel[sponge_joint.dofadr[0]:sponge_joint.dofadr[0] + 6]
        sponge_returned = bool(
            np.linalg.norm(pad_center[:2] - SPONGE_RETURN_XY) < .025
            and .802 <= pad_center[2] <= .820
            and pad_normal[2] > .98
            and np.linalg.norm(sponge_velocity[:3]) < .02
            and np.linalg.norm(sponge_velocity[3:]) < .2
            and self.finger_contacts() == 0)
        home_error = float(np.max(np.abs(self.state()[:6] - self.home)))
        return {"tie_xyz": ties, "ties_collected": collected,
                "tie_linear_speed_mps": speeds, "collected_count": sum(collected),
                "sponge_xyz": self.data.body("cleanup_sponge").xpos.tolist(),
                "sponge_return_xy": pad_center[:2].tolist(),
                "sponge_returned": sponge_returned,
                "home_joint_max_error_rad": home_error,
                "finger_contacts": self.finger_contacts(),
                "pad_table_peak_n": self.peak_pad_force,
                "pad_table_impulse_ns": self.total_pad_impulse,
                "debris_table_impulse_ns": self.total_debris_table_impulse,
                "table_peak_n": self.peak_table_force,
                "table_impulse_ns": self.total_table_impulse}


def execute_episode(arm, rng, fixed_eval, recorder, viewer=None, realtime=False):
    action = arm.reset(rng, fixed_eval)
    data = arm.data
    phase_name = "settle"
    grasp_seen = False
    regrasp_seen = False
    handoff_released = False
    handoff_placed = False
    regrasp_camera_height_m = 0.
    camera_min_height_after_regrasp_m = float("inf")
    wrist_flip_deg = 0.
    robot_table_seen = False
    first_sweep_collected_count = 0
    corrective_sweep_count = 0
    sweep_yaw_delta_deg = 0.
    slip_done = False
    slip_peak_force = 0.
    rotation = Rotation.from_euler("x", -25, degrees=True).as_matrix() @ \
        Rotation.from_euler("z", 180, degrees=True).as_matrix() @ DOWN_ROTATION
    grasp_rotation = rotation.copy()
    sweep_rotation = (Rotation.from_euler("x", -30, degrees=True).as_matrix() @ \
        Rotation.from_euler("z", 180, degrees=True).as_matrix() @ DOWN_ROTATION) @ WRIST_FLIP

    def tick(command, phase):
        nonlocal phase_name, grasp_seen, robot_table_seen, camera_min_height_after_regrasp_m, slip_peak_force
        phase_name = phase
        started = time.monotonic()
        if viewer is not None and not viewer.is_running():
            raise RuntimeError("Viewer closed")
        if recorder is not None:
            recorder.capture(arm, command, phase)
        try:
            arm.step(command)
        except RuntimeError:
            if recorder is not None:
                recorder.finish_step(arm, completed=False)
            raise
        if recorder is not None:
            recorder.finish_step(arm)
        grasp_seen |= arm.finger_contacts() == 2
        robot_table_seen |= arm.last_contact["robot_table_contact"]
        if phase.startswith("slip_"):
            slip_peak_force = max(slip_peak_force, arm.last_contact["pad_table_normal_n"])
        if robot_table_seen:
            raise RuntimeError("Robot arm touched the tabletop")
        if regrasp_seen and (phase in ("handoff_lift", "route", "retract",
                                           "gather_route", "gather_approach", "gather_sweep",
                                           "return_sponge", "place_sponge")
                               or phase.startswith("sweep_") or phase.startswith("slip_")):
            camera_z = data.cam_xpos[arm.model.camera("wrist_cam").id, 2]
            finger_z = max(data.body("link10").xpos[2], data.body("link12").xpos[2])
            camera_min_height_after_regrasp_m = min(
                camera_min_height_after_regrasp_m, camera_z - finger_z)
            if camera_min_height_after_regrasp_m < .005:
                raise RuntimeError("Wrist camera dropped below the gripper after regrasp")
        if (phase in ("lift", "route", "sweep_approach", "retract", "return_sponge",
                      "handoff_route", "handoff_lift", "gather_route", "gather_approach",
                      "gather_sweep")
                or phase.startswith("sweep_")) and \
                arm.finger_contacts() != 2:
            raise RuntimeError(f"Lost two-finger sponge grasp during {phase}")
        if viewer is not None:
            viewer.sync()
        if realtime:
            time.sleep(max(0., 1 / FPS - (time.monotonic() - started)))

    def move(position, duration, phase, turn_degrees=0.):
        nonlocal action, rotation, slip_done
        start = data.site("tcp_site").xpos.copy()
        start_rotation = rotation.copy()
        seed = action[:6].copy()
        expected_position = position
        # Cartesian waypoints at 10 Hz; every IK target is interpolated at 50 Hz.
        waypoints = max(1, round(duration * 10))
        for index in range(1, waypoints + 1):
            t = index / waypoints
            blend = t * t * (3 - 2 * t)
            target = start + blend * (position - start)
            target_rotation = rotation
            if phase in ("gather_approach", "sweep_approach"):
                level_position, level_rotation = level_sponge(target, rotation)
                delta = Rotation.from_matrix(level_rotation @ rotation.T).as_rotvec()
                target_rotation = Rotation.from_rotvec(blend * delta).as_matrix() @ rotation
                target[2] += blend * (level_position[2] - position[2])
                expected_position = target
            elif phase == "gather_sweep":
                nominal_rotation = (Rotation.from_euler("z", turn_degrees * blend, degrees=True).as_matrix()
                                    @ start_rotation)
                try:
                    q, target, target_rotation = solve_level_sweep(target, nominal_rotation, seed)
                except RuntimeError:
                    follow_ompl_contact_path(position, seed, phase)
                    return
                expected_position = target
            if phase != "gather_sweep":
                try:
                    q = arm.solve_ik(target, target_rotation, seed)
                except RuntimeError:
                    if phase != "sweep_approach" or slip_done:
                        raise
                    table_supported_slip()
                    slip_done = True
                    return move(position, duration, phase)
            for fraction in np.linspace(.2, 1., 5):
                action = np.r_[seed + fraction * (q - seed), action[6]]
                tick(action, phase)
            seed = q
        if phase == "gather_sweep":
            rotation = target_rotation
        for _ in range(10):
            tick(action, phase)
        error = np.linalg.norm(data.site("tcp_site").xpos - expected_position)
        if error > .012:
            raise RuntimeError(f"TCP tracking error in {phase}: {error:.4f} m")

    def grip(width, duration, phase):
        nonlocal action
        start = action[6]
        for fraction in np.linspace(0., 1., round(duration * FPS) + 1)[1:]:
            action = np.r_[action[:6], start + fraction * (width - start)]
            tick(action, phase)

    def table_supported_slip():
        nonlocal rotation
        pad = arm.model.geom("cleanup_sponge_pad")
        tcp_rotation = data.site("tcp_site").xmat.reshape(3, 3).copy()
        pad_rotation = data.geom_xmat[pad.id].reshape(3, 3).copy()
        relative_before = tcp_rotation.T @ pad_rotation
        rotation = tcp_rotation
        move(data.site("tcp_site").xpos + [0., 0., -.007], .6, "slip_press")
        force_limits = {joint: arm.model.actuator_forcerange[arm.servo[joint]].copy()
                        for joint in (10, 12)}
        try:
            for joint in (10, 12):
                arm.model.actuator_forcerange[arm.servo[joint]] = [-4., 4.]
            grip(.023, .8, "slip_release")
            for _ in range(20):
                tick(action, "slip_settle")
        finally:
            for joint, limits in force_limits.items():
                arm.model.actuator_forcerange[arm.servo[joint]] = limits
        grip(0., .8, "slip_close")
        tcp_rotation = data.site("tcp_site").xmat.reshape(3, 3)
        pad_rotation = data.geom_xmat[pad.id].reshape(3, 3)
        relative_change = Rotation.from_matrix(
            (tcp_rotation.T @ pad_rotation) @ relative_before.T).magnitude()
        if (arm.finger_contacts() != 2 or np.degrees(relative_change) < 5.
                or pad_rotation[2, 2] < np.cos(np.deg2rad(2.))
                or abs(pad_lowest_z() - .800) > .003 or slip_peak_force > 10.):
            raise RuntimeError("Table-supported sponge slip did not meet contact limits")
        rotation = tcp_rotation.copy()

    def rotate(target_rotation, duration, phase):
        nonlocal action, rotation
        position = data.site("tcp_site").xpos.copy()
        start_rotation = data.site("tcp_site").xmat.reshape(3, 3).copy()
        delta = Rotation.from_matrix(target_rotation @ start_rotation.T).as_rotvec()
        seed = action[:6].copy()
        count = max(1, round(duration * 10))
        for index in range(1, count + 1):
            t = index / count
            blend = t * t * (3 - 2 * t)
            target = Rotation.from_rotvec(blend * delta).as_matrix() @ start_rotation
            q = arm.solve_ik(position, target, seed)
            for fraction in np.linspace(.2, 1., 5):
                action = np.r_[seed + fraction * (q - seed), action[6]]
                tick(action, phase)
            seed = q
        rotation = target_rotation

    def sweep_arc(center, height, duration, radius, start_deg, end_deg, phase):
        nonlocal action, rotation
        seed = action[:6].copy()
        count = max(1, round(duration * 10))
        for index in range(1, count + 1):
            t = index / count
            blend = t * t * (3 - 2 * t)
            angle = start_deg + blend * (end_deg - start_deg)
            radians = np.deg2rad(angle)
            position = np.r_[center + radius * np.array([np.cos(radians), np.sin(radians)]),
                             height]
            target_rotation = Rotation.from_euler("z", angle + TOOL_YAW_OFFSET_DEG, degrees=True).as_matrix() @ sweep_rotation
            q, position, target_rotation = solve_level_sweep(position, target_rotation, seed)
            for fraction in np.linspace(.2, 1., 5):
                action = np.r_[seed + fraction * (q - seed), action[6]]
                tick(action, phase)
            seed = q
        rotation = target_rotation
        if np.linalg.norm(data.site("tcp_site").xpos - position) > .012:
            raise RuntimeError("TCP tracking error at arc end")

    def move_joints(target, duration, phase):
        nonlocal action
        start = action[:6].copy()
        duration = max(duration, float(np.max(np.abs(target - start))) / 1.5)
        for t in np.linspace(0., 1., round(duration * FPS) + 1)[1:]:
            blend = t * t * (3 - 2 * t)
            action = np.r_[start + blend * (target - start), action[6]]
            tick(action, phase)
        for _ in range(15):
            tick(action, phase)

    def pad_lowest_z():
        geom = arm.model.geom("cleanup_sponge_pad")
        center = data.geom_xpos[geom.id]
        rotation_matrix = data.geom_xmat[geom.id].reshape(3, 3)
        corners = np.array([[sx, sy, sz] for sx in (-1., 1.)
                            for sy in (-1., 1.) for sz in (-1., 1.)]) * geom.size
        return float((corners @ rotation_matrix.T + center)[:, 2].min())

    def level_sponge(position, nominal_rotation):
        tcp = data.site("tcp_site")
        pad = arm.model.geom("cleanup_sponge_pad")
        tcp_rotation = tcp.xmat.reshape(3, 3)
        pad_rotation = data.geom_xmat[pad.id].reshape(3, 3)
        pad_relative_rotation = tcp_rotation.T @ pad_rotation
        normal = (nominal_rotation @ pad_relative_rotation)[:, 2]
        axis = np.cross(normal, [0., 0., 1.])
        sine = np.linalg.norm(axis)
        correction = (Rotation.from_rotvec(axis / sine * np.arctan2(sine, normal[2])).as_matrix()
                      if sine > 1e-8 else np.eye(3))
        target_rotation = correction @ nominal_rotation
        pad_offset = tcp_rotation.T @ (data.geom_xpos[pad.id] - tcp.xpos)
        target_pad_rotation = target_rotation @ pad_relative_rotation
        bottom_offset = ((target_rotation @ pad_offset)[2]
                         - np.abs(target_pad_rotation[2]) @ pad.size)
        target = np.array(position, copy=True)
        target[2] = .801 - bottom_offset
        return target, target_rotation

    def solve_level_sweep(position, nominal_rotation, seed):
        nonlocal sweep_yaw_delta_deg
        target, flat_rotation = level_sponge(position, nominal_rotation)
        for yaw_delta in np.arange(sweep_yaw_delta_deg, 6., 5.):
            candidate = Rotation.from_euler("z", yaw_delta, degrees=True).as_matrix() @ flat_rotation
            try:
                joints = arm.solve_ik(target, candidate, seed)
            except RuntimeError:
                continue
            sweep_yaw_delta_deg = float(yaw_delta)
            return joints, target, candidate
        raise RuntimeError(f"Level sponge sweep has no reachable wrist yaw at {np.round(target, 4).tolist()}")

    def follow_ompl_contact_path(goal, seed, phase):
        nonlocal action, rotation
        from ompl import base as ob, geometric as og

        for _ in range(4):
            start_xy = data.site("tcp_site").xpos[:2].copy()
            contact_rotation = data.site("tcp_site").xmat.reshape(3, 3).copy()

            def contact_target(x, y):
                return level_sponge(np.array([x, y, goal[2]]), contact_rotation)

            goal_xy = None
            for x in np.arange(min(goal[0], start_xy[0]), goal[0] - .051, -.005):
                target, flat_rotation = contact_target(x, goal[1])
                try:
                    arm.solve_ik(target, flat_rotation, seed)
                    goal_xy = np.array([x, goal[1]])
                    break
                except RuntimeError:
                    continue
            if goal_xy is None:
                raise RuntimeError("OMPL contact path has no reachable sweep goal")

            space = ob.RealVectorStateSpace(2)
            bounds = ob.RealVectorBounds(2)
            bounds.setLow(0, min(start_xy[0], goal_xy[0]) - .015)
            bounds.setHigh(0, max(start_xy[0], goal_xy[0]) + .005)
            bounds.setLow(1, start_xy[1] - .002)
            bounds.setHigh(1, goal_xy[1] + .002)
            space.setBounds(bounds)
            setup = og.SimpleSetup(space)

            def reachable(state):
                target, flat_rotation = contact_target(state[0], state[1])
                try:
                    arm.solve_ik(target, flat_rotation, seed)
                    return True
                except RuntimeError:
                    return False

            setup.setStateValidityChecker(reachable)
            setup.getSpaceInformation().setStateValidityCheckingResolution(.02)
            start_state, goal_state = space.allocState(), space.allocState()
            start_state[0], start_state[1] = start_xy
            goal_state[0], goal_state[1] = goal_xy
            setup.setStartAndGoalStates(start_state, goal_state)
            setup.setPlanner(og.RRTConnect(setup.getSpaceInformation()))
            if not setup.solve(1.0):
                raise RuntimeError("OMPL found no reachable contact path")
            path = setup.getSolutionPath()
            path.interpolate(max(2, int(path.length() / .004) + 1))
            for state in path.getStates()[1:]:
                target, flat_rotation = contact_target(state[0], state[1])
                try:
                    joints = arm.solve_ik(target, flat_rotation, seed)
                except RuntimeError:
                    break
                for fraction in np.linspace(.2, 1., 5):
                    action = np.r_[seed + fraction * (joints - seed), action[6]]
                    tick(action, phase)
                seed = joints
            else:
                rotation = flat_rotation
                if np.linalg.norm(data.site("tcp_site").xpos[:2] - goal_xy) <= .012:
                    return
        raise RuntimeError("OMPL contact path remained unreachable after replanning")

    def retract_from_table():
        nonlocal action, rotation
        from ompl import base as ob, geometric as og

        start = data.site("tcp_site").xpos.copy()
        target = np.array([.420, start[1], .900])
        tool_rotation = data.site("tcp_site").xmat.reshape(3, 3).copy()
        seed = action[:6].copy()
        space = ob.RealVectorStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, target[0] - .005)
        bounds.setHigh(0, start[0] + .005)
        bounds.setLow(1, start[2] - .001)
        bounds.setHigh(1, target[2] + .005)
        space.setBounds(bounds)
        setup = og.SimpleSetup(space)

        def reachable(state):
            if state[1] < start[2] + .8 * (start[0] - state[0]) - .003:
                return False
            try:
                arm.solve_ik([state[0], start[1], state[1]], tool_rotation, seed)
                return True
            except RuntimeError:
                return False

        setup.setStateValidityChecker(reachable)
        setup.getSpaceInformation().setStateValidityCheckingResolution(.01)
        first, last = space.allocState(), space.allocState()
        first[0], first[1] = start[0], start[2]
        last[0], last[1] = target[0], target[2]
        setup.setStartAndGoalStates(first, last)
        setup.setPlanner(og.RRTConnect(setup.getSpaceInformation()))
        if not setup.solve(1.0):
            raise RuntimeError("OMPL found no reachable retreat from the table")
        path = setup.getSolutionPath()
        path.interpolate(max(2, int(path.length() / .004) + 1))
        for state in path.getStates()[1:]:
            joints = arm.solve_ik([state[0], start[1], state[1]], tool_rotation, seed)
            for fraction in np.linspace(.2, 1., 5):
                action = np.r_[seed + fraction * (joints - seed), action[6]]
                tick(action, "retract")
            seed = joints
        rotation = tool_rotation
        if np.linalg.norm(data.site("tcp_site").xpos - target) > .012:
            raise RuntimeError("TCP tracking error after OMPL retreat")

    def tie_bounds(name):
        geom = arm.model.geom(name + "_geom")
        mesh_id = int(geom.dataid[0])
        first = arm.model.mesh_vertadr[mesh_id]
        count = arm.model.mesh_vertnum[mesh_id]
        vertices = arm.model.mesh_vert[first:first + count]
        world = vertices @ data.geom_xmat[geom.id].reshape(3, 3).T + data.geom_xpos[geom.id]
        return world.min(axis=0), world.max(axis=0)

    def move_home(duration):
        nonlocal action
        start = action[:6].copy()
        for t in np.linspace(0., 1., round(duration * FPS) + 1)[1:]:
            blend = t * t * (3 - 2 * t)
            action = np.r_[start + blend * (arm.home - start), action[6]]
            tick(action, "return_home")

    try:
        for _ in range(FPS):
            tick(action, "settle")
        grasp = data.site("sponge_grasp").xpos.copy()
        pregrasp = grasp + [0, 0, .042]
        q_pregrasp = arm.solve_ik(pregrasp, rotation,
                                 np.array([2.21, 1.57, -1.97, -1.20, 2.62, 1.02]))
        move_joints(q_pregrasp, 3.0, "approach")
        grip(.055, .4, "preshape")
        move(grasp, 1.5, "descend")
        grip(0., 1.0, "close")
        for _ in range(25):
            tick(action, "grasp_hold")
        if arm.finger_contacts() != 2:
            raise RuntimeError("Sponge grasp did not establish two-finger contact")
        move(data.site("tcp_site").xpos + [0, 0, .045], 1.5, "lift")
        if data.body("cleanup_sponge").xpos[2] < .84:
            raise RuntimeError("Sponge failed the lift check")
        handoff_rotation = Rotation.from_euler("z", HANDOFF_YAW_DEG,
                                              degrees=True).as_matrix() @ grasp_rotation
        handoff_joint_target = arm.solve_ik(np.array([.442, -.045, .880]),
                                             handoff_rotation, action[:6])
        move_joints(handoff_joint_target, 2.5, "handoff_route")
        rotation = handoff_rotation
        handoff_xy = data.site("tcp_site").xpos[:2] + SPONGE_HANDOFF_XY - \
            data.geom_xpos[arm.pad_geom, :2]
        move(np.r_[handoff_xy, data.site("tcp_site").xpos[2]], 1.0, "handoff_route")
        handoff_z = data.site("tcp_site").xpos[2] + (.801 - pad_lowest_z())
        move(np.r_[handoff_xy, handoff_z], 1.0, "handoff_place")
        for _ in range(15):
            tick(action, "handoff_place")
        grip(.055, .8, "handoff_release")
        handoff_released = arm.finger_contacts() == 0
        move(np.r_[handoff_xy, handoff_z + .030], 1.0, "handoff_depart")
        pad_center = data.geom_xpos[arm.pad_geom]
        sponge_joint = arm.model.joint("fj_sponge")
        sponge_speed = np.linalg.norm(data.qvel[sponge_joint.dofadr[0]:sponge_joint.dofadr[0] + 3])
        handoff_placed = bool(np.linalg.norm(pad_center[:2] - SPONGE_HANDOFF_XY) < .025
                              and .802 <= pad_center[2] <= .820
                              and sponge_speed < .02)
        if not (handoff_released and handoff_placed):
            raise RuntimeError("Sponge did not settle at the regrasp point")
        regrasp = data.site("sponge_grasp").xpos.copy()
        regrasp[2] -= .028
        move(np.r_[data.site("tcp_site").xpos[:2], .900], 1.0, "handoff_depart")
        pre_flip_rotation = data.site("tcp_site").xmat.reshape(3, 3).copy()
        flip_target = action[:6].copy()
        flip_target[5] += np.pi
        move_joints(flip_target, 2.5, "handoff_flip")
        rotation = handoff_rotation @ WRIST_FLIP
        move(regrasp + [0., 0., .035], 1.0, "handoff_flip")
        move(regrasp, 1.0, "handoff_regrasp")
        grip(0., .8, "handoff_close")
        for _ in range(15):
            tick(action, "handoff_hold")
        regrasp_seen = arm.finger_contacts() == 2
        camera = arm.model.camera("wrist_cam").id
        regrasp_camera_height_m = float(data.cam_xpos[camera, 2] - max(
            data.body("link10").xpos[2], data.body("link12").xpos[2]))
        wrist_flip_deg = float(np.rad2deg(Rotation.from_matrix(
            data.site("tcp_site").xmat.reshape(3, 3) @ pre_flip_rotation.T).magnitude()))
        if not (regrasp_seen and regrasp_camera_height_m > .005
                and wrist_flip_deg > 175.):
            raise RuntimeError("Camera-above-gripper 180-degree regrasp failed")
        for gather_index, name in enumerate(sorted(TIE_NAMES, key=lambda item: tie_bounds(item)[0][1])):
            minimum, maximum = tie_bounds(name)
            if (minimum[1] + maximum[1]) / 2 >= .25:
                continue
            sweep_x = (minimum[0] + maximum[0]) / 2
            start_y = minimum[1] - .075
            if gather_index == 0:
                sweep_yaw_delta_deg = 0.
                move(np.array([sweep_x, start_y, data.site("tcp_site").xpos[2]]),
                     1.0, "gather_sweep", turn_degrees=10.)
            else:
                move(np.array([sweep_x, start_y, .900]), 1.0, "gather_route")
                gather_z = data.site("tcp_site").xpos[2] + (.801 - pad_lowest_z())
                move(np.array([sweep_x, start_y, gather_z]), 1.0, "gather_approach")
            sweep_yaw_delta_deg = 0.
            move(np.array([sweep_x, .25, data.site("tcp_site").xpos[2]]),
                 2.5, "gather_sweep", turn_degrees=20. if gather_index == 0 else 10.)
            retract_from_table()
        gathered_y = [(tie_bounds(name)[0][1] + tie_bounds(name)[1][1]) / 2
                      for name in TIE_NAMES]
        if min(gathered_y) < .18 or max(gathered_y) - min(gathered_y) > .125:
            raise RuntimeError("Cable ties did not gather near the middle")
        center = data.body("link0").xpos[:2].copy()
        gathered_xy = np.array([(tie_bounds(name)[0][:2] + tie_bounds(name)[1][:2]) / 2
                                for name in TIE_NAMES]).mean(axis=0)
        start_deg = float(np.clip(np.rad2deg(np.arctan2(
            gathered_xy[1] - center[1], gathered_xy[0] - center[0])) - 12., 5., 40.))
        start_angle = np.deg2rad(start_deg)
        start_xy = center + ARC_RADIUS * np.array([np.cos(start_angle), np.sin(start_angle)])
        inner_xy = center + .310 * np.array([np.cos(start_angle), np.sin(start_angle)])
        start_rotation = Rotation.from_euler("z", start_deg + TOOL_YAW_OFFSET_DEG, degrees=True).as_matrix() @ sweep_rotation
        inner_joint_target = arm.solve_ik(np.r_[inner_xy, .900], rotation, action[:6])
        move_joints(inner_joint_target, 2.0, "route")
        rotate(start_rotation, 1.0, "route")
        move(np.r_[start_xy, .860], 1.0, "route")
        sweep_z = data.site("tcp_site").xpos[2] + (.801 - pad_lowest_z())
        move(np.r_[start_xy, sweep_z], 1.0, "sweep_approach")
        sweep_yaw_delta_deg = 0.
        sweep_arc(center, sweep_z, 5.5, ARC_RADIUS, start_deg, ARC_END_DEG,
                  "sweep_1")
        safe_joint_target = arm.solve_ik(np.array([.180, .350, .900]), rotation,
                                         action[:6])
        move_joints(safe_joint_target, 2.0, "retract")
        first_sweep_collected_count = arm.result_metrics()["collected_count"]
        if first_sweep_collected_count < len(TIE_NAMES):
            q_regrasp = q_pregrasp.copy()
            q_regrasp[5] += np.pi
            move_joints(q_regrasp, 2.5, "return_sponge")
            rotation = grasp_rotation @ WRIST_FLIP
            pad_xy = data.geom_xpos[arm.pad_geom, :2].copy()
            handoff_xy = data.site("tcp_site").xpos[:2] + SPONGE_HANDOFF_XY - pad_xy
            move(np.r_[handoff_xy, data.site("tcp_site").xpos[2]], 1.0, "return_sponge")
            handoff_z = data.site("tcp_site").xpos[2] + (.801 - pad_lowest_z())
            move(np.r_[handoff_xy, handoff_z], 1.0, "handoff_place")
            grip(.055, .8, "handoff_release")
            move(np.r_[handoff_xy, handoff_z + .035], 1.0, "handoff_depart")
            regrasp = data.site("sponge_grasp").xpos.copy()
            move(regrasp, 1.0, "handoff_regrasp")
            grip(0., .8, "handoff_close")
            for _ in range(15):
                tick(action, "handoff_hold")
            if arm.finger_contacts() != 2:
                raise RuntimeError("Sponge regrasp after first sweep failed")
            move(data.site("tcp_site").xpos + [0, 0, .035], 1.0, "handoff_lift")
        while (arm.result_metrics()["collected_count"] < len(TIE_NAMES)
               and corrective_sweep_count < MAX_CORRECTIVE_SWEEPS):
            corrective_sweep_count += 1
            collected = arm.result_metrics()["ties_collected"]
            name = next(name for name, inside in zip(TIE_NAMES, collected) if not inside)
            minimum, maximum = tie_bounds(name)
            target_xy = (minimum[:2] + maximum[:2]) / 2
            relative = target_xy - center
            radius = float(np.clip(np.linalg.norm(relative), .43, .50))
            correction_start_deg = float(np.clip(
                np.rad2deg(np.arctan2(relative[1], relative[0])) - 12., 5., 60.))
            correction_angle = np.deg2rad(correction_start_deg)
            second_xy = center + radius * np.array([np.cos(correction_angle), np.sin(correction_angle)])
            second_rotation = Rotation.from_euler("z", correction_start_deg + TOOL_YAW_OFFSET_DEG,
                                                  degrees=True).as_matrix() @ sweep_rotation
            second_joint_target = arm.solve_ik(np.r_[second_xy, .860], second_rotation,
                                               action[:6])
            move_joints(second_joint_target, 3.0, "route")
            rotation = second_rotation
            second_z = data.site("tcp_site").xpos[2] + (.801 - pad_lowest_z())
            move(np.r_[second_xy, second_z], 1.0, "sweep_approach")
            sweep_yaw_delta_deg = 0.
            sweep_arc(center, second_z, 3.5, radius, correction_start_deg,
                      ARC_END_DEG, f"sweep_{corrective_sweep_count + 1}")
            move(np.r_[data.site("tcp_site").xpos[:2], .900], 1.0, "retract")
        if arm.result_metrics()["collected_count"] != len(TIE_NAMES):
            raise RuntimeError("Debris remains after corrective sweeps; sponge return skipped")
        q_return = q_pregrasp.copy()
        q_return[5] += np.pi
        move_joints(q_return, 2.5, "return_sponge")
        rotation = grasp_rotation @ WRIST_FLIP
        pad_xy = data.geom_xpos[arm.pad_geom, :2].copy()
        place_xy = data.site("tcp_site").xpos[:2] + SPONGE_RETURN_XY - pad_xy
        move(np.r_[place_xy, data.site("tcp_site").xpos[2]], 1.0, "return_sponge")
        place_z = data.site("tcp_site").xpos[2] + (.801 - pad_lowest_z())
        move(np.r_[place_xy, place_z], 1.0, "place_sponge")
        for _ in range(15):
            tick(action, "place_sponge")
        grip(.101, .8, "release_sponge")
        move(np.r_[place_xy, place_z + .005], 1.0, "depart_sponge")
        move_home(3.)
        for _ in range(FPS // 2):
            tick(action, "verify")
        metrics = arm.result_metrics()
        success = bool(grasp_seen and handoff_released and handoff_placed and regrasp_seen
                       and regrasp_camera_height_m > .005 and wrist_flip_deg > 175.
                       and camera_min_height_after_regrasp_m > .005
                       and metrics["collected_count"] == len(TIE_NAMES)
                       and metrics["sponge_returned"]
                       and metrics["home_joint_max_error_rad"] < .04
                       and abs(arm.state()[6] - .101) < .005
                       and not robot_table_seen)
        reason = "success" if success else "Cleanup, sponge return, or Z home incomplete"
    except RuntimeError as exc:
        success, reason = False, str(exc)
        metrics = arm.result_metrics()
    return {"success": success, "reason": reason, "final_phase": phase_name,
            "two_finger_contact_seen": grasp_seen, "robot_table_contact": robot_table_seen,
            "handoff_released": handoff_released, "handoff_placed": handoff_placed,
            "regrasp_seen": regrasp_seen,
            "regrasp_camera_height_m": regrasp_camera_height_m,
            "camera_min_height_after_regrasp_m": camera_min_height_after_regrasp_m,
            "wrist_flip_deg": wrist_flip_deg,
            "first_sweep_collected_count": first_sweep_collected_count,
            "corrective_sweep_count": corrective_sweep_count,
            "obstacle_present": False, "initial_sponge_xyz": recorder.initial_sponge_xyz,
            "initial_tie_xyz": recorder.initial_tie_xyz, **metrics}


class Recorder:
    def __init__(self, arm, folder, images=True):
        self.folder = folder
        self.frames = []
        self.contacts = []
        self.initial_sponge_xyz = None
        self.initial_tie_xyz = None
        self.renderer = arm.mj.Renderer(arm.model, height=256, width=256) if images else None
        self.fixed_renderer = arm.mj.Renderer(arm.model, height=360, width=640) if images else None
        self.render_option = arm.mj.MjvOption()
        self.render_option.geomgroup[3:] = 0
        if images:
            for camera in ("cam_fixed", "cam_wrist"):
                (folder / camera).mkdir(parents=True, exist_ok=False)
        else:
            folder.mkdir(parents=True, exist_ok=False)

    def capture(self, arm, action, phase):
        index = len(self.frames)
        if self.initial_sponge_xyz is None:
            self.initial_sponge_xyz = arm.data.body("cleanup_sponge").xpos.tolist()
            self.initial_tie_xyz = [arm.data.body(name).xpos.tolist() for name in TIE_NAMES]
        if self.renderer is not None:
            import cv2
            for camera, renderer, folder in (("overview", self.fixed_renderer, "cam_fixed"),
                                             ("wrist_cam", self.renderer, "cam_wrist")):
                renderer.update_scene(arm.data, camera=camera, scene_option=self.render_option)
                image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(self.folder / folder / f"{index:05d}.jpg"), image):
                    raise OSError("Failed to save camera frame")
        self.frames.append((arm.state().copy(), np.array(action, copy=True), arm.data.time,
                            arm.data.qpos.copy(), arm.data.qvel.copy(),
                            arm.data.body("cleanup_sponge").xpos.copy(),
                            np.array([arm.data.body(name).xpos.copy() for name in TIE_NAMES]),
                            phase))

    def finish_step(self, arm, completed=True):
        contact = arm.last_contact.copy()
        pad = arm.model.geom("cleanup_sponge_pad")
        pad_rotation = arm.data.geom_xmat[pad.id].reshape(3, 3)
        half_height = np.abs(pad_rotation[2]) @ pad.size
        contact["pad_tilt_deg"] = float(np.degrees(np.arccos(np.clip(pad_rotation[2, 2], -1, 1))))
        contact["pad_corner_spread_m"] = float(2 * np.abs(pad_rotation[2, :2]) @ pad.size[:2])
        contact["pad_bottom_z_m"] = float(arm.data.geom_xpos[pad.id, 2] - half_height)
        if not completed:
            contact = {key: float("nan") for key in contact}
        contact["physics_step_completed"] = completed
        self.contacts.append(contact)

    def finish(self, arm, result):
        if self.renderer is not None:
            self.renderer.close()
            self.fixed_renderer.close()
        if self.frames:
            states, actions, times, sim_qpos, sim_qvel, sponge, ties, phases = zip(*self.frames)
            np.savez_compressed(self.folder / "joint_data.npz", qpos=np.float32(states),
                                actions=np.float32(actions), actions_exact=np.float64(actions),
                                timestamps=np.array(times), sim_qpos=np.array(sim_qpos),
                                sim_qvel=np.array(sim_qvel), sponge_xyz=np.array(sponge),
                                tie_xyz=np.array(ties), phase=np.array(phases),
                                pad_table_normal_n=np.array([c["pad_table_normal_n"] for c in self.contacts]),
                                pad_table_impulse_ns=np.array([c["pad_table_impulse_ns"] for c in self.contacts]),
                                pad_tilt_deg=np.array([c["pad_tilt_deg"] for c in self.contacts]),
                                pad_corner_spread_m=np.array([c["pad_corner_spread_m"] for c in self.contacts]),
                                pad_bottom_z_m=np.array([c["pad_bottom_z_m"] for c in self.contacts]),
                                debris_table_impulse_ns=np.array([c["debris_table_impulse_ns"] for c in self.contacts]),
                                table_normal_n=np.array([c["table_normal_n"] for c in self.contacts]),
                                table_impulse_ns=np.array([c["table_impulse_ns"] for c in self.contacts]),
                                physics_step_completed=np.array([c["physics_step_completed"] for c in self.contacts]),
                                final_sim_qpos=arm.data.qpos.copy(), final_sim_qvel=arm.data.qvel.copy())
        phase_metrics = {}
        for phase in sorted({frame[-1] for frame in self.frames}):
            records = [c for frame, c in zip(self.frames, self.contacts)
                       if frame[-1] == phase and c["physics_step_completed"]]
            phase_metrics[phase] = {
                key: float(sum(c[key] for c in records))
                for key in ("pad_table_impulse_ns", "debris_table_impulse_ns", "table_impulse_ns")}
        metadata = {**result, "contact_impulses_by_phase": phase_metrics, "schema": "agv_dummyx_workbench_cleanup_zip_tie_v3", "fps": FPS,
                    "state_names": STATE_NAMES, "units": ["rad"] * 6 + ["m"],
                    "action_semantics": "absolute position target held for the next 20 ms",
                    "contact_metrics": "MuJoCo normal contact truth, offline only; impulses integrate all physics substeps, forces are frame maxima of summed contact normals",
                    "success_definition": "Sponge is released and settled at the handoff point, then regrasped after a 180-degree wrist roll with camera above both fingers; camera remains above fingers from regrasp through sponge placement; all three whole cable ties rest in the outboard tray; sponge rests in the return zone after release; arm is within 0.04 rad of Z home with gripper open; no robot-table penetration over 1 mm",
                    "sponge_handoff_target_xy_m": SPONGE_HANDOFF_XY.tolist(),
                    "handoff_yaw_deg": HANDOFF_YAW_DEG,
                    "wrist_camera_constraint": "camera center at least 5 mm above both finger body origins from regrasp through sponge placement",
                    "sponge_return_target_xy_m": SPONGE_RETURN_XY.tolist(),
                    "home_reference": "models/agv_dummyx/z_pose.json",
                    "expert_sweep": "Gather each tie from its measured bounds toward world Y=0.25 m, then sweep from the measured group angle toward the tray; corrective arcs target remaining ties",
                    "tray_arc_radius_m": ARC_RADIUS,
                    "tray_arc_end_deg": ARC_END_DEG,
                    "tool_yaw_offset_deg": TOOL_YAW_OFFSET_DEG,
                    "slip_force_schedule": "During slip_release, finger actuator force limits are 4 N per side, then restored before slip_close; seven-dimensional actions are unchanged",
                    "max_corrective_sweeps": MAX_CORRECTIVE_SWEEPS,
                    "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "frames": len(self.frames), "images_saved": self.renderer is not None,
                    "fixed_camera": {"camera_name": "overview", "saved_resolution_wh": [640, 360]},
                    "mujoco_version": arm.mj.__version__,
                    "scene_sha256": hashlib.sha256(SCENE.read_bytes()).hexdigest(),
                    "model_files_sha256": model_manifest(),
                    "model_assumptions": {"fixed_agv": True, "object_attached": False,
                                          "fixed_suspended_tray": True,
                                          "cup_in_collection_scene": False}}
        (self.folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (self.folder / "instruction.txt").write_text(INSTRUCTION + "\n")


def next_index(paths, prefix):
    numbers = [int(path.name[len(prefix):]) for path in paths
               if path.is_dir() and path.name[len(prefix):].isdigit()]
    return max(numbers, default=-1) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="Number of successful episodes to collect")
    parser.add_argument("--max_attempts", type=int, default=None)
    parser.add_argument("--fixed_eval", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no_images", action="store_true", help="Physics diagnostics only")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--pause_on_failure", action="store_true",
                        help="Keep the viewer open at a failed attempt until it is closed")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/agv_dummyx_workbench_cleanup_zip_ties")
    args = parser.parse_args()
    if args.num_episodes < 1 or (args.max_attempts is not None and args.max_attempts < 1):
        parser.error("episode and attempt counts must be positive")
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
    arm = CleanupArm()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    episode_index = next_index(output.glob("ep_*"), "ep_")
    attempt_index = max(next_index(output.glob("attempt_*"), "attempt_"),
                        next_index((output / "failures").glob("attempt_*"), "attempt_"))
    runs = output / "runs"
    run_index = max((int(path.stem[4:]) for path in runs.glob("run_*.json")
                     if path.stem[4:].isdigit()), default=-1) + 1
    rng = np.random.default_rng(args.seed)
    results, successes = [], 0
    attempts_allowed = args.max_attempts or 10 * args.num_episodes
    if args.headless:
        context = nullcontext(None)
    else:
        import mujoco.viewer
        context = mujoco.viewer.launch_passive(arm.model, arm.data)
    with context as viewer:
        if viewer is not None:
            with viewer.lock():
                viewer.cam.lookat[:] = [.55, .10, .90]
                viewer.cam.distance = 2.4
                viewer.cam.azimuth = 145
                viewer.cam.elevation = -35
                viewer.opt.geomgroup[3:] = 0
        for attempt in range(attempts_allowed):
            if successes >= args.num_episodes:
                break
            folder = output / f"attempt_{attempt_index + attempt:04d}"
            recorder = Recorder(arm, folder, images=not args.no_images)
            try:
                result = execute_episode(arm, rng, args.fixed_eval, recorder, viewer, args.realtime)
            except RuntimeError as exc:
                result = {"success": False, "reason": str(exc)}
            recorder.finish(arm, result)
            if result["success"]:
                folder.rename(output / f"ep_{episode_index + successes:04d}")
                successes += 1
            else:
                (output / "failures").mkdir(exist_ok=True)
                folder.rename(output / "failures" / folder.name)
            results.append(result)
            print(f"Attempt {attempt + 1}: {result['reason']} ({successes}/{args.num_episodes})",
                  flush=True)
            if args.pause_on_failure and not result["success"] and viewer is not None:
                print("Viewer paused at failure; close the window to exit.", flush=True)
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(.05)
                break
            if viewer is not None and not viewer.is_running():
                break
    runs.mkdir(exist_ok=True)
    summary_path = runs / f"run_{run_index:04d}.json"
    summary_path.write_text(json.dumps({"requested_successes": args.num_episodes,
                                        "successes": successes, "attempts": len(results),
                                        "seed": args.seed, "fixed_eval": args.fixed_eval,
                                        "training_data": not args.no_images,
                                        "results": results}, indent=2) + "\n")
    print(f"Output: {output} (summary: {summary_path})")
    return 0 if successes == args.num_episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
