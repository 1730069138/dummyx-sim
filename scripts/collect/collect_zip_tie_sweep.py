"""Collect randomized sponge sweeping demonstrations at the short table edge.

The expert directly grasps a 90-degree rotated sponge at the right side of
the tabletop with its wrist camera above the gripper. It sweeps two whole
cable ties off the tabletop toward a suspended tray, returns the sponge, and moves to Z home.
The cup is absent from collection. No contact-solver force is used to choose
an action. Contact loads are logged only for offline evaluation.

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
SCENE = ROOT / "models/agv_dummyx/workbench_cleanup_zip_ties_short_table/scene.xml"
ARM_SOURCE = ROOT / "models/arm_description"
FPS = 50
STATE_NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper_width"]
TIE_NAMES = [f"cleanup_tie_{i}" for i in range(2)]
INSTRUCTIONS = (
    "Pick up the sponge, sweep the cable ties off the table, then put the sponge back where it was.",
    "Grab the sponge, sweep the cable ties off the tabletop, and return the sponge to its original place.",
    "Take the sponge, sweep the cable ties off the workbench, then place the sponge back.",
    "Use the sponge to sweep the cable ties off the table, then return it to where it started.",
    "Lift the sponge, sweep the cable ties off the tabletop, and put the sponge back in its original spot.",
)
DOWN_ROTATION = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
SPONGE_RETURN_XY = np.array([.430, -.175])
SPONGE_RETURN_YAW_DEG = 180.
SPONGE_START_ROTATION = Rotation.from_euler("z", 90, degrees=True).as_matrix()
DIRECT_GRASP_YAW_DEG = 60.
DIRECT_GRASP_OFFSET = Rotation.from_euler("z", 180, degrees=True).as_matrix()


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
        self.mj.mj_forward(self.model, self.data)
        region = self.model.site("cleanup_tie_spawn_region")
        lower = self.data.site("cleanup_tie_spawn_region").xpos[:2] - region.size[:2]
        upper = self.data.site("cleanup_tie_spawn_region").xpos[:2] + region.size[:2]
        for name, center in zip(TIE_NAMES, ((.550, .040), (.550, .120))):
            joint = self.data.joint("fj_" + name.removeprefix("cleanup_"))
            geom = self.model.geom(name + "_geom")
            mesh_id = int(geom.dataid[0])
            first = self.model.mesh_vertadr[mesh_id]
            vertices = self.model.mesh_vert[first:first + self.model.mesh_vertnum[mesh_id]]
            yaw = 0. if fixed_eval else rng.uniform(-180., 180.)
            joint.qpos[3:7] = Rotation.from_euler("z", yaw, degrees=True).as_quat()[[3, 0, 1, 2]]
            self.mj.mj_forward(self.model, self.data)
            world_xy = (vertices @ self.data.geom_xmat[geom.id].reshape(3, 3).T
                        + self.data.geom_xpos[geom.id])[:, :2]
            offsets = world_xy - self.data.body(name).xpos[:2]
            position_min = lower - offsets.min(axis=0)
            position_max = upper - offsets.max(axis=0)
            if np.any(position_min > position_max):
                raise RuntimeError(f"{name} cannot fit in spawn region")
            joint.qpos[:2] = center if fixed_eval else rng.uniform(position_min, position_max)
            self.mj.mj_forward(self.model, self.data)
            world_xy = (vertices @ self.data.geom_xmat[geom.id].reshape(3, 3).T
                        + self.data.geom_xpos[geom.id])[:, :2]
            if np.any(world_xy < lower) or np.any(world_xy > upper):
                raise RuntimeError(f"{name} outline outside spawn region")
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
        table = self.model.geom("workbench_top")
        lower = self.data.geom_xpos[table.id] - table.size
        upper = self.data.geom_xpos[table.id] + table.size
        # A tie is cleared once its whole outline has left the tabletop.
        # The 1 mm edge tolerance accounts for mesh/contact discretization.
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
            off_table = (minimum[0] >= upper[0] - .001
                         or maximum[0] <= lower[0] + .001
                         or minimum[1] >= upper[1] - .001
                         or maximum[1] <= lower[1] + .001
                         or maximum[2] <= upper[2] - .001)
            collected.append(bool(off_table))
            speeds.append(speed)
            ties.append(((minimum + maximum) / 2).tolist())
        pad = self.model.geom("cleanup_sponge_pad")
        pad_center = self.data.geom_xpos[pad.id]
        pad_rotation = self.data.geom_xmat[pad.id].reshape(3, 3)
        pad_normal = pad_rotation[:, 2]
        pad_yaw = np.rad2deg(np.arctan2(pad_rotation[1, 1], pad_rotation[0, 1]))
        sponge_yaw_error_deg = abs((pad_yaw - SPONGE_RETURN_YAW_DEG + 180.) % 360. - 180.)
        sponge_orientation_error_deg = float(np.rad2deg(Rotation.from_matrix(
            SPONGE_START_ROTATION.T @ pad_rotation).magnitude()))
        pad_corners = np.array([[x, y, z] for x in (-1., 1.)
                                for y in (-1., 1.) for z in (-1., 1.)]) * pad.size
        pad_world_corners = pad_corners @ pad_rotation.T + pad_center
        parking = self.model.site("cleanup_sponge_parking")
        parking_center = self.data.site("cleanup_sponge_parking").xpos[:2]
        sponge_in_parking_region = bool(
            np.all(pad_world_corners[:, :2].min(axis=0) >= parking_center - parking.size[:2])
            and np.all(pad_world_corners[:, :2].max(axis=0) <= parking_center + parking.size[:2]))
        sponge_joint = self.model.joint("fj_sponge")
        sponge_velocity = self.data.qvel[sponge_joint.dofadr[0]:sponge_joint.dofadr[0] + 6]
        sponge_returned = bool(
            np.linalg.norm(pad_center[:2] - SPONGE_RETURN_XY) < .010
            and .802 <= pad_center[2] <= .820
            and pad_normal[2] > .98
            and sponge_orientation_error_deg < 5.
            and sponge_in_parking_region
            and np.linalg.norm(sponge_velocity[:3]) < .02
            and np.linalg.norm(sponge_velocity[3:]) < .2
            and self.finger_contacts() == 0)
        home_error = float(np.max(np.abs(self.state()[:6] - self.home)))
        return {"tie_xyz": ties, "ties_collected": collected,
                "tie_linear_speed_mps": speeds, "collected_count": sum(collected),
                "sponge_xyz": self.data.body("cleanup_sponge").xpos.tolist(),
                "sponge_return_xy": pad_center[:2].tolist(),
                "sponge_returned": sponge_returned,
                "sponge_yaw_error_deg": sponge_yaw_error_deg,
                "sponge_orientation_error_deg": sponge_orientation_error_deg,
                "sponge_in_parking_region": sponge_in_parking_region,
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
    direct_grasp_seen = False
    direct_camera_height_m = 0.
    camera_min_height_after_grasp_m = float("inf")
    robot_table_seen = False
    first_sweep_collected_count = 0
    corrective_sweep_count = 0
    rotation = Rotation.from_euler("x", -25, degrees=True).as_matrix() @ \
        Rotation.from_euler("z", 180, degrees=True).as_matrix() @ DOWN_ROTATION
    grasp_rotation = rotation.copy()

    def tick(command, phase):
        nonlocal phase_name, grasp_seen, robot_table_seen, camera_min_height_after_grasp_m
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
        if robot_table_seen:
            raise RuntimeError("Robot arm touched the tabletop")
        if direct_grasp_seen and (phase in ("lift", "route", "retract",
                                            "return_sponge", "place_sponge")
                               or phase.startswith("sweep_")):
            camera_z = data.cam_xpos[arm.model.camera("wrist_cam").id, 2]
            finger_z = max(data.body("link10").xpos[2], data.body("link12").xpos[2])
            camera_min_height_after_grasp_m = min(
                camera_min_height_after_grasp_m, camera_z - finger_z)
            if camera_min_height_after_grasp_m < .005:
                raise RuntimeError("Wrist camera dropped below the gripper after grasp")
        if (phase in ("lift", "route", "sweep_approach", "retract", "return_sponge",
                      "place_sponge")
                or phase.startswith("sweep_")) and \
                arm.finger_contacts() != 2:
            raise RuntimeError(f"Lost two-finger sponge grasp during {phase}")
        if viewer is not None:
            viewer.sync()
        if realtime:
            time.sleep(max(0., 1 / FPS - (time.monotonic() - started)))

    def move(position, duration, phase):
        nonlocal action
        start = data.site("tcp_site").xpos.copy()
        seed = action[:6].copy()
        expected_position = position
        # Cartesian waypoints at 10 Hz; every IK target is interpolated at 50 Hz.
        waypoints = max(1, round(duration * 10))
        for index in range(1, waypoints + 1):
            t = index / waypoints
            blend = t * t * (3 - 2 * t)
            target = start + blend * (position - start)
            target_rotation = rotation
            if phase == "sweep_approach":
                level_position, level_rotation = level_sponge(target, rotation)
                delta = Rotation.from_matrix(level_rotation @ rotation.T).as_rotvec()
                target_rotation = Rotation.from_rotvec(blend * delta).as_matrix() @ rotation
                target[2] += blend * (level_position[2] - position[2])
                expected_position = target
            joints = arm.solve_ik(target, target_rotation, seed)
            for fraction in np.linspace(.2, 1., 5):
                action = np.r_[seed + fraction * (joints - seed), action[6]]
                tick(action, phase)
            seed = joints
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
        direct_rotation = (Rotation.from_euler("z", DIRECT_GRASP_YAW_DEG,
                                              degrees=True).as_matrix()
                           @ grasp_rotation @ DIRECT_GRASP_OFFSET)
        rotation = direct_rotation
        pregrasp = grasp + [0., 0., .042]
        q_pregrasp = arm.solve_ik(pregrasp, rotation,
                                 np.array([2.21, 1.57, -1.97, -1.20, 2.62, 1.02]))
        move_joints(q_pregrasp, 3., "approach")
        grip(.055, .4, "preshape")
        move(grasp + [0., 0., -.028], 1.5, "descend")
        grip(0., 1., "close")
        for _ in range(25):
            tick(action, "grasp_hold")
        direct_grasp_seen = arm.finger_contacts() == 2
        camera = arm.model.camera("wrist_cam").id
        direct_camera_height_m = float(data.cam_xpos[camera, 2] - max(
            data.body("link10").xpos[2], data.body("link12").xpos[2]))
        if not direct_grasp_seen or direct_camera_height_m <= .005:
            raise RuntimeError("Camera-above-gripper direct grasp failed")
        move(data.site("tcp_site").xpos + [0., 0., .045], 1.5, "lift")
        if data.body("cleanup_sponge").xpos[2] < .84:
            raise RuntimeError("Sponge failed the lift check")
        sweep_rotation = data.site("tcp_site").xmat.reshape(3, 3).copy()
        start_joint_target = arm.solve_ik([.443, -.060, .850], sweep_rotation,
                                          action[:6])
        move_joints(start_joint_target, 2., "route")
        rotation = sweep_rotation
        move(np.array([.443, -.060, .850]), .6, "route")
        sweep_start, _ = level_sponge(np.array([.443, -.060, .850]), rotation)
        move(sweep_start, 1., "sweep_approach")
        seed = action[:6].copy()
        for index in range(1, 53):
            fraction = index / 52
            y = -.060 + .380 * fraction
            x = .443 - .015 * max(0., (y - .200) / .120)
            nominal = (Rotation.from_euler("z", 70 * fraction, degrees=True).as_matrix()
                       @ sweep_rotation)
            target, flat_rotation = level_sponge(np.array([x, y, .850]), nominal)
            joints = arm.solve_ik(target, flat_rotation, seed)
            for step in np.linspace(.2, 1., 5):
                action = np.r_[seed + step * (joints - seed), action[6]]
                tick(action, "sweep_1")
            seed = joints
        rotation = data.site("tcp_site").xmat.reshape(3, 3).copy()
        move(np.array([.360, .300, .860]), 1., "retract")
        for _ in range(150):
            tick(action, "verify")
        first_sweep_collected_count = arm.result_metrics()["collected_count"]
        while arm.result_metrics()["collected_count"] != len(TIE_NAMES) and corrective_sweep_count < 2:
            current = arm.result_metrics()
            remaining = [np.array(xy[:2]) for xy, collected in
                         zip(current["tie_xyz"], current["ties_collected"]) if not collected]
            if not remaining or any(xy[1] > .31 for xy in remaining):
                break
            corrective_sweep_count += 1
            target_tie = min(remaining, key=lambda xy: xy[0])
            start_y = max(-.06, target_tie[1] - .10)
            rotation = sweep_rotation
            desired_x = np.clip(target_tie[0] - .05, .47, .49)
            for start_x in np.arange(desired_x, .469, -.01):
                try:
                    q_start = arm.solve_ik([start_x, start_y, .850], rotation, action[:6])
                except RuntimeError:
                    continue
                break
            else:
                raise RuntimeError("No reachable X lane for remaining cable tie")
            move_joints(q_start, 2., "route")
            move(np.array([start_x, start_y, .850]), .6, "route")
            sweep_start, _ = level_sponge(np.array([start_x, start_y, .850]), rotation)
            move(sweep_start, 1., "sweep_approach")
            seed = action[:6].copy()
            for index in range(1, 31):
                fraction = index / 30
                y = start_y + (.30 - start_y) * fraction
                x = start_x - .020 * max(0., (y - .200) / .100)
                nominal = (Rotation.from_euler("z", 70 * fraction, degrees=True).as_matrix()
                           @ sweep_rotation)
                target, flat_rotation = level_sponge(np.array([x, y, .850]), nominal)
                joints = arm.solve_ik(target, flat_rotation, seed)
                for step in np.linspace(.2, 1., 5):
                    action = np.r_[seed + step * (joints - seed), action[6]]
                    tick(action, f"sweep_{corrective_sweep_count + 1}")
                seed = joints
            rotation = data.site("tcp_site").xmat.reshape(3, 3).copy()
            move(np.array([.360, .300, .860]), 1., "retract")
            for _ in range(150):
                tick(action, "verify")
        if arm.result_metrics()["collected_count"] != len(TIE_NAMES):
            raise RuntimeError("Cable tie remains on tabletop after corrective sweep; sponge return skipped")
        q_return = q_pregrasp.copy()
        move_joints(q_return, 2.5, "return_sponge")
        pad_axis = data.geom_xmat[arm.pad_geom].reshape(3, 3)[:, 1]
        pad_yaw = np.rad2deg(np.arctan2(pad_axis[1], pad_axis[0]))
        yaw_correction = (SPONGE_RETURN_YAW_DEG - pad_yaw + 180.) % 360. - 180.
        tcp = data.site("tcp_site")
        rotation = (Rotation.from_euler("z", yaw_correction, degrees=True).as_matrix()
                    @ tcp.xmat.reshape(3, 3))
        aligned_joints = arm.solve_ik(tcp.xpos.copy(), rotation, action[:6])
        move_joints(aligned_joints, 1.5, "return_sponge")
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
        success = bool(grasp_seen and direct_grasp_seen
                       and direct_camera_height_m > .005
                       and camera_min_height_after_grasp_m > .005
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
            "direct_grasp_seen": direct_grasp_seen,
            "direct_camera_height_m": direct_camera_height_m,
            "camera_min_height_after_grasp_m": camera_min_height_after_grasp_m,
            "first_sweep_collected_count": first_sweep_collected_count,
            "corrective_sweep_count": corrective_sweep_count,
            "obstacle_present": False, "initial_sponge_xyz": recorder.initial_sponge_xyz,
            "initial_tie_xyz": recorder.initial_tie_xyz, **metrics}


class Recorder:
    def __init__(self, arm, folder, instruction, images=True):
        self.folder = folder
        self.instruction = instruction
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
        tie_region = arm.model.site("cleanup_tie_spawn_region")
        metadata = {**result, "contact_impulses_by_phase": phase_metrics, "schema": "agv_dummyx_workbench_cleanup_zip_tie_short_table_v7", "fps": FPS,
                    "state_names": STATE_NAMES, "units": ["rad"] * 6 + ["m"],
                    "action_semantics": "absolute position target held for the next 20 ms",
                    "contact_metrics": "MuJoCo normal contact truth, offline only; impulses integrate all physics substeps, forces are frame maxima of summed contact normals",
                    "success_definition": "The initially rotated sponge is grasped once with camera at least 5 mm above both fingers through sponge placement; both whole cable tie outlines leave the tabletop with 1 mm edge tolerance, without requiring tray containment or low speed; the whole sponge pad rests inside the marked right-side parking region within 5 degrees of its initial 3D orientation; arm is within 0.04 rad of Z home with gripper open; no robot-table penetration over 1 mm",
                    "wrist_camera_constraint": "camera center at least 5 mm above both finger body origins from initial grasp through sponge placement",
                    "sponge_return_target_xy_m": SPONGE_RETURN_XY.tolist(),
                    "home_reference": "models/agv_dummyx/z_pose.json",
                    "tie_spawn_region_center_xy_m": arm.data.site("cleanup_tie_spawn_region").xpos[:2].tolist(),
                    "tie_spawn_region_half_size_xy_m": tie_region.size[:2].tolist(),
                    "tie_spawn_randomization": "Each tie independently has uniform yaw in [-180, 180] degrees and a uniform feasible center position for that yaw; both complete mesh outlines remain inside the marked region, and the ties may touch. --fixed_eval selects the baseline layout.",
                    "expert_sweep": "Sweep along +Y off the tabletop; after verification, replan up to two corrective sweeps only for ties still on the tabletop before sponge return",
                    "ties_collected_field_meaning": "Legacy field name: each boolean means the whole tie outline is off the tabletop, not necessarily inside the tray",
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
        (self.folder / "instruction.txt").write_text(self.instruction + "\n")


def next_index(paths, prefix):
    numbers = [int(path.name[len(prefix):]) for path in paths
               if path.is_dir() and path.name[len(prefix):].isdigit()]
    return max(numbers, default=-1) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="Number of successful episodes to collect")
    parser.add_argument("--max_attempts", type=int, default=None)
    parser.add_argument("--fixed_eval", action="store_true",
                        help="Use the deterministic tie layout instead of episode randomization")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no_images", action="store_true", help="Physics diagnostics only")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--pause_on_failure", action="store_true",
                        help="Keep the viewer open at a failed attempt until it is closed")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/agv_dummyx_workbench_cleanup_zip_ties_short_table")
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
            instruction = INSTRUCTIONS[(episode_index + successes) % len(INSTRUCTIONS)]
            recorder = Recorder(arm, folder, instruction, images=not args.no_images)
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
