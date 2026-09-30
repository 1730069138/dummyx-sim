"""Collect the screwdriver-to-bin task with the verified-zero arm in MuJoCo.

State/action: joint1..joint6 [rad], total gripper opening [m]. The action is the
position-servo target held over the following 20 ms. Physics state is written
only at reset; grasping and release use contacts, without an object attachment.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "models/arm_description"
SCENE = ROOT / "models/agv_dummyx/scene.xml"
sys.path.insert(0, str(ROOT / "scripts/tools"))

FPS = 50
YAW_RANGE_RAD = (np.pi / 3, 2 * np.pi / 3)
STATE_NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper_width"]
INSTRUCTION = "Pick up the screwdriver and place it into the box."
COLLECTION_INSTRUCTIONS = (
    INSTRUCTION,
    "Grasp the screwdriver and put it in the box.",
    "Move the screwdriver from the table into the box.",
    "Place the screwdriver in the box.",
    "Take the screwdriver from the workbench and put it into the box.",
)
DOWN_ROTATION = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])


def sample_pose(rng, region, fixed_eval=False):
    if fixed_eval:
        return np.array([.415, -.16]), np.pi / 2
    return np.array([rng.uniform(*region["x_range_m"]),
                     rng.uniform(*region["y_range_m"])]), rng.uniform(*YAW_RANGE_RAD)


def model_manifest():
    files = {SCENE.parent / name for name in ("z_pose.json", "spawn_region.json", "obstacle.xml")}
    files.add(SOURCE / "scripts/gripper_kinematics.py")
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
    return {str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def load_coupling():
    spec = importlib.util.spec_from_file_location("hardstop_gripper", SOURCE / "scripts/gripper_kinematics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HardstopArm:
    def __init__(self, scene=SCENE):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.joints = [self.model.joint(n) for n in STATE_NAMES[:6]]
        self.qadr = np.array([j.qposadr[0] for j in self.joints])
        self.dadr = np.array([j.dofadr[0] for j in self.joints])
        self.limits = np.array([j.range for j in self.joints])
        self.servo = {i: self.model.actuator(f"servo_joint{i}").id for i in (1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13)}
        self.servo_zero = {i: self.model.qpos0[self.model.joint(f"joint{i}").qposadr[0]]
                           for i in self.servo}
        self.coupling = load_coupling()
        self.crank_samples = np.linspace(0, self.coupling.LOWER, 4097)
        self.width_samples = np.array([-2 * self.coupling.coupled_positions(q)["joint10"]
                                       for q in self.crank_samples])
        self.steps = round(1 / FPS / self.model.opt.timestep)
        if not np.isclose(self.steps * self.model.opt.timestep, 1 / FPS):
            raise ValueError("Physics timestep must divide the recording interval")
        self.target_body = self.model.body("real_screwdriver").id
        self.target_geoms = np.flatnonzero(self.model.geom_bodyid == self.target_body)
        self.finger_bodies = {self.model.body("link10").id, self.model.body("link12").id}
        self.home = None

    def state(self):
        d = self.data
        width = -float(d.joint("joint10").qpos[0] + d.joint("joint12").qpos[0])
        return np.r_[d.qpos[self.qadr], width]

    def solve_ik(self, position, rotation, seed):
        d = self.ik_data
        d.qpos[:] = self.data.qpos

        def residual(q):
            d.qpos[self.qadr] = q
            self.mj.mj_forward(self.model, d)
            tcp = d.site("tcp_site")
            angular = Rotation.from_matrix(rotation @ tcp.xmat.reshape(3, 3).T).as_rotvec()
            return np.r_[5 * (tcp.xpos - position), angular]

        lower, upper = self.limits[:, 0] + 0.01, self.limits[:, 1] - 0.01
        result = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper),
                               max_nfev=60, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        error = residual(result.x)
        if np.linalg.norm(error[:3]) / 5 > 0.002 or np.linalg.norm(error[3:]) > 0.025:
            raise RuntimeError(f"IK unreachable: xyz={np.round(position, 4).tolist()}, "
                               f"position_error={np.linalg.norm(error[:3]) / 5:.4f}m, "
                               f"angle_error={np.linalg.norm(error[3:]):.4f}rad")
        return result.x

    def command(self, action):
        if np.shape(action) != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("Expected seven finite action values")
        if np.any(action[:6] < self.limits[:, 0]) or np.any(action[:6] > self.limits[:, 1]):
            raise ValueError("Arm target outside source URDF limits")
        width = float(action[6])
        # A saved float32 0.101 is slightly greater than the decimal limit.
        if not -1e-8 <= width <= 0.101 + 1e-8:
            raise ValueError("Gripper opening outside [0, 0.101] m")
        width = np.clip(width, 0., 0.101)
        for i in range(1, 7):
            self.data.ctrl[self.servo[i]] = action[i - 1] - self.servo_zero[i]
        for i in (10, 12):
            self.data.ctrl[self.servo[i]] = -width / 2 - self.servo_zero[i]
        # Decorative crank and rods follow the *measured* jaw opening, so they
        # stop with the fingers when a grasped object blocks closure.
        measured_width = np.clip(self.state()[6], 0, 0.101)
        crank = np.interp(measured_width, self.width_samples, self.crank_samples)
        self.data.ctrl[self.servo[7]] = crank - self.servo_zero[7]
        coupled = self.coupling.coupled_positions(float(crank))
        for i in (11, 13):
            self.data.ctrl[self.servo[i]] = coupled[f"joint{i}"] - self.servo_zero[i]

    def step(self, action):
        for _ in range(self.steps):
            self.command(action)
            self.mj.mj_step(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)) or np.any(self.data.warning.number):
            raise RuntimeError("MuJoCo reported a numerical warning or non-finite state")

    def reset(self, xy=(.415, -.16), yaw=np.pi / 2):
        m, d = self.model, self.data
        self.mj.mj_resetData(m, d)
        initial = json.loads((SCENE.parent / "z_pose.json").read_text())["joint_positions"]
        for name, value in initial.items():
            d.joint(name).qpos[:] = value
        d.joint("fj_screwdriver").qpos[:] = [*xy, .8135, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.mj.mj_forward(m, d)
        self.home = d.qpos[self.qadr].copy()
        return np.r_[self.home, .101]

    def tool_vertices(self):
        points = []
        for g in self.target_geoms:
            if not self.model.geom_contype[g]:
                continue
            mesh = self.model.geom_dataid[g]
            start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            points.append(self.model.mesh_vert[start:start + count] @
                          self.data.geom_xmat[g].reshape(3, 3).T + self.data.geom_xpos[g])
        return np.concatenate(points)

    def finger_contacts(self):
        fingers = set()
        for c in self.data.contact:
            b1, b2 = self.model.geom_bodyid[[c.geom1, c.geom2]]
            if b1 == self.target_body and b2 in self.finger_bodies:
                fingers.add(int(b2))
            if b2 == self.target_body and b1 in self.finger_bodies:
                fingers.add(int(b1))
        return len(fingers)

    def forbidden_contact(self):
        """Arm contact with table/bin, or non-adjacent self-contact."""
        m = self.model
        for c in self.data.contact:
            if c.dist >= -0.001:
                continue
            names = [m.geom(int(g)).name for g in (c.geom1, c.geom2)]
            robot = [n.startswith("collision_") for n in names]
            if all(robot):
                return names
            if any(robot) and self.target_body not in m.geom_bodyid[[c.geom1, c.geom2]]:
                return names
        return None

    def placement_metrics(self):
        """Accept a stable tool resting in the bin, including one leaning on its rim."""
        d = self.data
        vertices = self.tool_vertices()
        box = d.body("drop_box").xpos
        tool_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        inside = bool(np.all(np.abs(tool_center[:2] - box[:2]) < [.098, .103]))
        bottom = d.geom("bin_bottom").xpos[2] + self.model.geom("bin_bottom").size[2]
        rim = d.geom("bin_rim_x-1").xpos[2] + self.model.geom("bin_rim_x-1").size[2]
        height_ok = bool(bottom - .002 <= vertices[:, 2].min() < rim)
        bin_body = self.model.body("drop_box").id
        bin_contact = any({int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])}
                          == {self.target_body, bin_body} for c in d.contact)
        speed = float(np.linalg.norm(d.joint("fj_screwdriver").qvel[:3]))
        return {"inside_bin": inside, "height_ok": height_ok, "bin_contact": bin_contact,
                "object_speed_m_s": speed,
                "finger_contacts": self.finger_contacts(), "gripper_width_m": float(self.state()[6]),
                "object_xyz": d.body("real_screwdriver").xpos.tolist()}


def execute_episode(arm, rng, fixed_eval, record, viewer=None, realtime=False,
                    trajectory_type="normal", drop_location="random"):
    from agv_spawn_region import load_spawn_region, update_viewer_spawn_region
    arm.reset()
    region = load_spawn_region(arm.model, arm.data)
    xy, yaw = sample_pose(rng, region, fixed_eval)
    action = arm.reset(xy, yaw)
    d = arm.data
    bin_center = d.body("drop_box").xpos.copy()
    # The TCP grips near the handle, 52 mm behind the tool's geometric center.
    # Offset X so the complete screwdriver, rather than the TCP, lands centrally.
    place_target = np.array([bin_center[0] - .052, bin_center[1], .872])
    start_height = float(d.body("real_screwdriver").xpos[2])
    max_height, grasp_seen = start_height, False
    recovery_event_frame = None
    actual_drop_location = None
    miss_offset_xy = None
    scheduled_drop_phase = None
    scheduled_drop_fraction = None

    class PlannedDrop(Exception):
        pass
    rotation = (Rotation.from_euler("y", -10, degrees=True).as_matrix() @
                Rotation.from_euler("z", yaw).as_matrix() @ DOWN_ROTATION)
    place_rotation = (Rotation.from_euler("y", -25, degrees=True).as_matrix() @
                      Rotation.from_euler("z", 90, degrees=True).as_matrix() @ DOWN_ROTATION)
    retained_frames = 0
    tilt, flip = None, None
    phase_name = "sample_validation"

    def tick(command, phase):
        nonlocal max_height, grasp_seen, phase_name
        phase_name = phase
        start = time.monotonic()
        if viewer is not None and not viewer.is_running():
            raise RuntimeError("Viewer closed")
        if record is not None:
            record(arm, command, phase)
        arm.step(command)
        max_height = max(max_height, float(d.body("real_screwdriver").xpos[2]))
        grasp_seen |= arm.finger_contacts() == 2
        contact = arm.forbidden_contact()
        if contact:
            raise RuntimeError(f"Forbidden contact: {contact}")
        if phase in ("lift", "drop_transfer", "height_adjust", "transfer", "above_bin", "lower") and arm.finger_contacts() != 2:
            raise RuntimeError(f"Lost two-finger grasp during {phase}")
        if phase in ("transfer", "above_bin"):
            heights = arm.tool_vertices()[:, 2]
            if heights.min() <= .853:
                raise RuntimeError(f"Tool carry height below box rim in {phase}")
        if viewer is not None:
            update_viewer_spawn_region(viewer, region)
            viewer.sync()
        if realtime:
            time.sleep(max(0., 1 / FPS - (time.monotonic() - start)))

    def move(position, target_rotation, duration, phase):
        nonlocal action, phase_name, scheduled_drop_phase
        phase_name = phase
        start_pos = d.site("tcp_site").xpos.copy()
        start_rot = Rotation.from_matrix(d.site("tcp_site").xmat.reshape(3, 3))
        delta_rot = (Rotation.from_matrix(target_rotation) * start_rot.inv()).as_rotvec()
        frames = round(duration * FPS)
        # Plan before stepping physics, including the longer wrist rotation when
        # the shortest orientation interpolation crosses a joint limit.
        angle = np.linalg.norm(delta_rot)
        alternatives = [delta_rot]
        if angle > .1:
            alternatives.append(delta_rot * (1 - 2 * np.pi / angle))
        last_error = None
        for turn in alternatives:
            seed = action[:6].copy()
            path = []
            try:
                for index in range(1, frames + 1):
                    fraction = index / frames
                    blend = fraction * fraction * (3 - 2 * fraction)
                    pos = start_pos + blend * (position - start_pos)
                    rot = (Rotation.from_rotvec(blend * turn) * start_rot).as_matrix()
                    q = arm.solve_ik(pos, rot, seed)
                    if np.max(np.abs(q - seed)) > .15:
                        raise RuntimeError(f"Discontinuous IK branch in {phase}")
                    path.append(q)
                    seed = q
                break
            except RuntimeError as exc:
                last_error = exc
                path = []
        if not path:
            raise last_error
        # Subdivide targets to respect the existing 2 rad/s command limit.
        for path_index, q in enumerate(path, 1):
            initial = action[:6].copy()
            subdivisions = max(1, int(np.ceil(np.max(np.abs(q - initial)) * FPS / 1.9)))
            for fraction in np.linspace(0., 1., subdivisions + 1)[1:]:
                action = np.r_[initial + fraction * (q - initial), action[6]]
                tick(action, phase)
            if phase == scheduled_drop_phase and path_index >= round(frames * scheduled_drop_fraction):
                scheduled_drop_phase = None
                raise PlannedDrop
        for _ in range(15):
            tick(action, phase)
        error = np.linalg.norm(d.site("tcp_site").xpos - position)
        if error > 0.008:
            raise RuntimeError(f"TCP tracking error in {phase}: {error:.4f}m")

    def grip(width, duration, phase):
        nonlocal action
        start_width = action[6]
        frames = round(duration * FPS)
        for index in range(1, frames + 1):
            fraction = index / frames
            action = np.r_[action[:6], start_width + fraction * (width - start_width)]
            tick(action, phase)

    def move_joints(target, duration, phase):
        nonlocal action, phase_name
        phase_name = phase
        initial = action[:6].copy()
        duration = max(duration, float(np.max(np.abs(target - initial))) * .95)
        for t in np.linspace(0., 1., round(duration * FPS) + 1)[1:]:
            blend = t * t * (3 - 2 * t)
            action = np.r_[initial + blend * (target - initial), action[6]]
            tick(action, phase)
        for _ in range(15):
            tick(action, phase)

    def grasp_at_current_pose(phase_prefix):
        nonlocal rotation, place_rotation, tilt, flip
        obj = d.body("real_screwdriver").xpos.copy()
        quat = d.joint("fj_screwdriver").qpos[3:7].copy()
        tool_rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
        tool_axis = tool_rotation.apply([1., 0., 0.])
        if abs(tool_axis[2]) > .25:
            raise RuntimeError("Recovery tool did not settle flat on the table")
        grasp_yaw = float(np.arctan2(tool_axis[1], tool_axis[0]))
        solution = None
        seed = action[:6].copy()
        for candidate_flip in (False, True):
            candidate_yaw = grasp_yaw + (np.pi if candidate_flip else 0.)
            for candidate_tilt in (10, 15, 20, 25, 30, 35, 40, 45):
                candidate_rotation = (Rotation.from_euler("y", -candidate_tilt, degrees=True).as_matrix() @
                                      Rotation.from_euler("z", candidate_yaw).as_matrix() @ DOWN_ROTATION)
                candidate_place = (Rotation.from_euler("y", -(candidate_tilt + 20), degrees=True).as_matrix() @
                                   Rotation.from_euler("z", np.pi / 2 + (np.pi if candidate_flip else 0.)).as_matrix() @ DOWN_ROTATION)
                try:
                    q = arm.solve_ik(np.array([obj[0], obj[1], .885]), candidate_rotation, seed)
                    descent_seed = q
                    for z in np.linspace(.885, obj[2] + .002, 25)[1:]:
                        descent_seed = arm.solve_ik(np.array([obj[0], obj[1], z]),
                                                    candidate_rotation, descent_seed)
                    arm.solve_ik(place_target, candidate_place, q)
                    solution = q
                    rotation, place_rotation = candidate_rotation, candidate_place
                    tilt, flip = candidate_tilt, candidate_flip
                    break
                except RuntimeError:
                    pass
            if solution is not None:
                break
        if solution is None:
            raise RuntimeError("Recovery grasp/place orientation unreachable")
        move_joints(solution, 3., f"{phase_prefix}_approach")
        move(np.array([obj[0], obj[1], .885]), rotation, 1., f"{phase_prefix}_align")
        move(obj + [0, 0, .002], rotation, 2., f"{phase_prefix}_descend")
        grip(0., 1.2, f"{phase_prefix}_close")
        for _ in range(25):
            tick(action, f"{phase_prefix}_hold")
        if arm.finger_contacts() != 2:
            raise RuntimeError("Recovery grasp did not establish two-finger contact")
        return obj, grasp_yaw + (np.pi if flip else 0.)

    def recover_dropped_tool():
        nonlocal recovery_event_frame, obj, carry_rotation
        release_xy = d.site("tcp_site").xpos[:2].copy()
        recovery_event_frame = len(record.frames) if record is not None else None
        grip(.101, .2, "drop_open")
        for _ in range(FPS):
            tick(action, "drop_settle")
        fallen = d.body("real_screwdriver").xpos.copy()
        if np.any(np.abs(fallen[:2] - release_xy) > .08):
            raise RuntimeError("Dropped tool moved outside local recovery area")
        if fallen[2] > start_height + .02:
            raise RuntimeError("Dropped tool did not reach the table")
        obj, recovered_yaw = grasp_at_current_pose("regrasp")
        carry_rotation = (Rotation.from_euler("y", -(tilt + 20), degrees=True).as_matrix() @
                          Rotation.from_euler("z", recovered_yaw).as_matrix() @ DOWN_ROTATION)
        move(np.array([obj[0], obj[1], .915]), carry_rotation, 2., "regrasp_lift")
        if arm.finger_contacts() != 2 or d.body("real_screwdriver").xpos[2] < start_height + .05:
            raise RuntimeError("Recovery lift failed")

    def check_missed_tool_workspace():
        vertices = arm.tool_vertices()
        if np.any(vertices[:, :2] < table[:2] - half[:2]) or np.any(vertices[:, :2] > table[:2] + half[:2]):
            raise RuntimeError("Missed grasp knocked tool off the workbench")
        current_xy = d.body("real_screwdriver").xpos[:2]
        if (current_xy[0] < region["x_range_m"][0] - .005 or
                current_xy[0] > region["x_range_m"][1] + .005 or
                current_xy[1] < region["y_range_m"][0] - .005 or
                current_xy[1] > region["y_range_m"][1] + .005):
            raise RuntimeError("Missed grasp knocked tool outside the collection workspace")

    try:
        vertices = arm.tool_vertices()
        table = d.geom("workbench_top").xpos
        half = arm.model.geom("workbench_top").size
        if np.any(vertices[:, :2] < table[:2] - half[:2]) or np.any(vertices[:, :2] > table[:2] + half[:2]):
            raise RuntimeError("Rejected sample: full tool extends beyond tabletop")
        # Conservative XY bounding-box clearance around the future pillar.
        pillar = ET.parse(SCENE.parent / "obstacle.xml").find("worldbody/body[@name='dynamic_pillar']")
        center = np.fromstring(pillar.get("pos"), sep=" ")[:2]
        extent = np.fromstring(pillar.find("geom").get("size"), sep=" ")[:2] + region["obstacle_clearance_m"]
        if np.all(vertices[:, :2].max(axis=0) >= center - extent) and np.all(vertices[:, :2].min(axis=0) <= center + extent):
            raise RuntimeError("Rejected sample: tool overlaps reserved obstacle clearance")
        for _ in range(FPS):
            tick(action, "z_home")
        obj = d.body("real_screwdriver").xpos.copy()
        start_height = float(obj[2])
        seed = np.array([2.49, 2.12, -.71, -1.85, 3.47, 1.81])
        solution = None
        for flip in (False, True):
            grasp_yaw = yaw + (np.pi if flip else 0.)
            for tilt in (10, 15, 20, 25, 30, 35, 40, 45):
                rotation = Rotation.from_euler("y", -tilt, degrees=True).as_matrix() @ Rotation.from_euler("z", grasp_yaw).as_matrix() @ DOWN_ROTATION
                place_rotation = Rotation.from_euler("y", -(tilt + 20), degrees=True).as_matrix() @ Rotation.from_euler("z", np.pi / 2 + (np.pi if flip else 0.)).as_matrix() @ DOWN_ROTATION
                try:
                    q = arm.solve_ik(np.array([obj[0], obj[1], .885]), rotation, seed)
                    arm.solve_ik(obj + [0, 0, .002], rotation, q)
                    arm.solve_ik(place_target, place_rotation, np.array([3.87, 1.89, -1.09, -2.27, 3.18, .2 if flip else 3.24]))
                    solution = q
                    break
                except RuntimeError:
                    pass
            if solution is not None:
                break
        if solution is None:
            raise RuntimeError("Rejected sample: no reachable grasp/place orientation")
        q = solution
        if trajectory_type == "missed_grasp":
            wrong_grasp = None
            wrong_q = None
            for _ in range(20):
                distance = rng.uniform(.05, .07)
                direction = rng.uniform(-np.pi, np.pi)
                offset = distance * np.array([np.cos(direction), np.sin(direction), 0.])
                candidate = obj + offset + [0., 0., .002]
                for miss_tilt in rng.permutation((10, 15, 20, 25, 30, 35, 40, 45)):
                    miss_rotation = (Rotation.from_euler("y", -float(miss_tilt), degrees=True).as_matrix() @
                                     Rotation.from_euler("z", grasp_yaw).as_matrix() @ DOWN_ROTATION)
                    try:
                        candidate_q = arm.solve_ik(np.array([candidate[0], candidate[1], .885]), miss_rotation, seed)
                        approach_q = candidate_q
                        for z in np.linspace(.885, candidate[2], 15)[1:]:
                            candidate_q = arm.solve_ik(np.array([candidate[0], candidate[1], z]), miss_rotation, candidate_q)
                        arm.solve_ik(np.array([candidate[0], candidate[1], .915]), miss_rotation, candidate_q)
                    except RuntimeError:
                        continue
                    wrong_grasp = candidate
                    wrong_q = approach_q
                    miss_offset_xy = offset[:2].tolist()
                    break
                if wrong_grasp is not None:
                    break
            if wrong_grasp is None:
                raise RuntimeError("Rejected sample: no reachable wrong-position grasp")
        move_joints(wrong_q if trajectory_type == "missed_grasp" else q, 3., "approach")
        if trajectory_type == "missed_grasp":
            # Descend to the normal grasp height but with a wrong lateral XY.
            move(np.array([wrong_grasp[0], wrong_grasp[1], .885]), miss_rotation, 1., "miss_approach")
            move(wrong_grasp, miss_rotation, 2., "miss_descend")
            grip(0., 1., "miss_close")
            move(np.array([wrong_grasp[0], wrong_grasp[1], .915]), miss_rotation, 1.5, "miss_lift")
            if max_height >= start_height + .05:
                raise RuntimeError("Wrong-position grasp unexpectedly lifted the tool")
            check_missed_tool_workspace()
            recovery_event_frame = len(record.frames) if record is not None else None
            grip(.101, 1., "miss_reopen")
            move(np.array([wrong_grasp[0], wrong_grasp[1], .885]), miss_rotation, 1., "miss_retreat")
            check_missed_tool_workspace()
            obj, grasp_yaw = grasp_at_current_pose("retry")
        else:
            move(obj + [0, 0, .002], rotation, 2., "descend")
            grip(0., 1.2, "close")
            for _ in range(25):
                tick(action, "grasp_hold")
            if arm.finger_contacts() != 2:
                raise RuntimeError("Grasp did not establish two-finger contact")
        carry_rotation = Rotation.from_euler("y", -(tilt + 20), degrees=True).as_matrix() @ Rotation.from_euler("z", grasp_yaw).as_matrix() @ DOWN_ROTATION
        move(np.array([obj[0], obj[1], .915]), carry_rotation, 2., "lift")
        if d.body("real_screwdriver").xpos[2] < start_height + .05:
            raise RuntimeError("Object failed the 5 cm lift check")
        if trajectory_type == "drop_regrasp":
            actual_drop_location = ("initial" if rng.integers(2) == 0 else "transfer") if drop_location == "random" else drop_location
            if actual_drop_location == "transfer":
                # Drop while moving toward the middle waypoint, before the
                # later transfer from the middle waypoint to the box.
                scheduled_drop_phase = "orient"
                scheduled_drop_fraction = rng.uniform(.35, .95)
                try:
                    move(np.array([.40, 0., .965]), place_rotation, 4., "orient")
                except PlannedDrop:
                    recover_dropped_tool()
                else:
                    raise RuntimeError("Scheduled transfer drop did not occur")
            else:
                recover_dropped_tool()
        # Random grasp yaws need time to rotate into the common placement pose.
        # Check box-rim clearance only after this alignment ends.
        move(np.array([.40, 0., .965]), place_rotation, 4., "orient")
        carry_z = .965
        for _ in range(5):
            heights = arm.tool_vertices()[:, 2]
            if heights.min() > .853:
                break
            rise = .855 - heights.min()
            target = d.site("tcp_site").xpos.copy()
            target[2] += rise
            move(target, place_rotation, max(.5, rise / .03), "height_adjust")
            carry_z = float(d.site("tcp_site").xpos[2])
        else:
            raise RuntimeError("Tool remained below box rim after height adjustment")
        move(np.array([.421, .185, carry_z]), place_rotation, 2., "transfer")
        move(np.array([.421, .185, carry_z]), place_rotation, 1., "above_bin")
        move(place_target, place_rotation, 2., "lower")
        grip(.101, 1., "release")
        move(np.array([.38, .15, .90]), place_rotation, 1.5, "retract")
        default_rotation = Rotation.from_euler("y", -10, degrees=True).as_matrix() @ Rotation.from_euler("z", 90, degrees=True).as_matrix() @ DOWN_ROTATION
        move(np.array([.40, 0., .885]), default_rotation, 1.5, "clear_bin")
        move_joints(arm.home, 3., "return_z")
        # Allow settling, then require a *consecutive* one-second valid window.
        for _ in range(3 * FPS):
            tick(action, "verify")
            metrics = arm.placement_metrics()
            valid = (metrics["inside_bin"] and metrics["height_ok"] and metrics["bin_contact"] and
                     metrics["object_speed_m_s"] < 0.03 and metrics["finger_contacts"] == 0 and
                     metrics["gripper_width_m"] > 0.08)
            retained_frames = retained_frames + 1 if valid else 0
            if retained_frames >= FPS:
                break
        success = bool(retained_frames >= FPS and grasp_seen and max_height >= start_height + 0.05)
        reason = "success" if success else "Final retention/release/stability check failed"
    except RuntimeError as exc:
        success, reason = False, str(exc)
    return {"success": success, "reason": reason, "trajectory_type": trajectory_type,
            "recovery_event_frame": recovery_event_frame, "drop_location": actual_drop_location,
            "miss_offset_xy": miss_offset_xy,
            "initial_xy": xy.tolist(), "initial_yaw": float(yaw),
            "spawn_region": region, "yaw_sampling_rad": list(YAW_RANGE_RAD),
            "obstacle_present": False, "grasp_tilt_deg": tilt, "grasp_flipped": flip, "final_phase": phase_name, "max_lift_m": max_height - start_height, "two_finger_contact_seen": grasp_seen,
            "retention_frames": retained_frames, "required_retention_frames": FPS,
            **arm.placement_metrics()}


class Recorder:
    def __init__(self, arm, folder, images=True, instruction=INSTRUCTION):
        self.folder = folder
        self.instruction = instruction
        self.frames = []
        self.renderer = arm.mj.Renderer(arm.model, height=256, width=256) if images else None
        self.fixed_renderer = arm.mj.Renderer(arm.model, height=360, width=640) if images else None
        self.render_option = arm.mj.MjvOption()
        self.render_option.geomgroup[3:] = 0
        if images:
            for camera in ("cam_fixed", "cam_wrist"):
                (folder / camera).mkdir(parents=True, exist_ok=False)
        else:
            folder.mkdir(parents=True, exist_ok=False)

    def __call__(self, arm, action, phase):
        index = len(self.frames)
        if self.renderer is not None:
            import cv2
            self.fixed_renderer.update_scene(arm.data, camera="overview", scene_option=self.render_option)
            fixed = cv2.cvtColor(self.fixed_renderer.render(), cv2.COLOR_RGB2BGR)
            self.renderer.update_scene(arm.data, camera="wrist_cam", scene_option=self.render_option)
            wrist = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
            for directory, image in (("cam_fixed", fixed), ("cam_wrist", wrist)):
                if not cv2.imwrite(str(self.folder / directory / f"{index:05d}.jpg"), image):
                    raise OSError("Failed to save camera frame")
        self.frames.append((arm.state().copy(), np.array(action, copy=True), arm.data.time,
                            arm.data.qpos.copy(), arm.data.qvel.copy(),
                            arm.data.body("real_screwdriver").xpos.copy(), phase))

    def finish(self, arm, result):
        if self.renderer is not None:
            self.renderer.close()
            self.fixed_renderer.close()
        if self.frames:
            state, actions, timestamps, sim_qpos, sim_qvel, objects, phases = zip(*self.frames)
            np.savez_compressed(self.folder / "joint_data.npz", qpos=np.float32(state),
                                actions=np.float32(actions), timestamps=np.array(timestamps),
                                actions_exact=np.array(actions, dtype=np.float64),
                                sim_qpos=np.array(sim_qpos), sim_qvel=np.array(sim_qvel),
                                object_xyz=np.array(objects), phase=np.array(phases),
                                final_sim_qpos=arm.data.qpos.copy(), final_sim_qvel=arm.data.qvel.copy())
        metadata = {**result, "schema": "agv_dummyx_screwdriver_v3", "fps": FPS,
                    "state_names": STATE_NAMES, "units": ["rad"] * 6 + ["m"],
                    "action_semantics": "absolute position targets held for the next 20 ms",
                    "actions_exact": "float64 executed targets retained for deterministic physics replay",
                    "frames": len(self.frames), "images_saved": self.renderer is not None,
                    "fixed_camera": {"camera_name": "overview", "model": "D415 RGB visual model",
                                     "raw_resolution_wh": [640, 360], "rgb_fov_deg": [69.4, 42.5],
                                     "saved_resolution_wh": [640, 360], "preprocess": "none"},
                    "mujoco_version": arm.mj.__version__,
                    "scene_sha256": hashlib.sha256(SCENE.read_bytes()).hexdigest(),
                    "model_files_sha256": model_manifest(),
                    "model_assumptions": {"fixed_agv": True, "object_attached": False,
                                          "debug_region_in_camera": False}}
        (self.folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (self.folder / "instruction.txt").write_text(self.instruction + "\n")


TRAJECTORY_TYPES = ("normal", "missed_grasp", "drop_regrasp")


def trajectory_targets(total):
    weights = (0.80, 0.15, 0.05)
    raw = [total * weight for weight in weights]
    targets = [int(value) for value in raw]
    for index in sorted(range(3), key=lambda i: (-(raw[i] - targets[i]), i))[:total - sum(targets)]:
        targets[index] += 1
    return dict(zip(TRAJECTORY_TYPES, targets))


def existing_trajectory_counts(output):
    episodes = {path.name for path in output.glob("ep_*") if path.is_dir()}
    typed = {}
    for path in sorted((output / "runs").glob("run_*.json")):
        typed.update(json.loads(path.read_text()).get("episode_types", {}))
    types_path = output / "runs/episode_types.json"
    if types_path.exists():
        typed.update(json.loads(types_path.read_text()))
    unknown = set(typed) - episodes
    if unknown:
        raise RuntimeError(f"Run records refer to missing episodes: {sorted(unknown)}")
    counts = {name: 0 for name in TRAJECTORY_TYPES}
    for episode in episodes:
        kind = typed.get(episode, "normal")
        if kind not in counts:
            raise RuntimeError(f"Unknown trajectory type for {episode}: {kind}")
        counts[kind] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of successful episodes to collect")
    parser.add_argument("--total_episodes", type=int,
                        help="Final dataset size with 80/15/5 normal/missed-grasp/drop-regrasp quotas")
    parser.add_argument("--max_attempts", type=int, default=None,
                        help="Optional total-attempt limit; default keeps trying until num_episodes succeed")
    parser.add_argument("--fixed_eval", action="store_true", help="Use XY=(0.415,-0.16), yaw=90 degrees every trial")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility; default varies between runs")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no_images", action="store_true", help="Physics diagnostics only; not a training dataset")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--trajectory_type", choices=("normal", "missed_grasp", "drop_regrasp"),
                        default="normal", help="Successful trajectory type to collect")
    parser.add_argument("--drop_location", choices=("random", "initial", "transfer"), default="random",
                        help="For drop_regrasp: choose drop position; random samples initial or transfer")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/agv_dummyx_screwdriver",
                        help="Dataset directory; new runs append numbered ep_* folders")
    args = parser.parse_args()
    if args.num_episodes < 1 or (args.total_episodes is not None and args.total_episodes < 1) or (args.max_attempts is not None and args.max_attempts < 1):
        parser.error("episode and attempt counts must be positive")
    if args.total_episodes is not None and (args.num_episodes != 1 or args.trajectory_type != "normal"):
        parser.error("--total_episodes cannot be combined with --num_episodes or --trajectory_type")
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
    arm = HardstopArm()
    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    existing_counts = existing_trajectory_counts(output)
    if args.total_episodes is None:
        remaining = {name: (args.num_episodes if name == args.trajectory_type else 0)
                     for name in TRAJECTORY_TYPES}
        targets = None
    else:
        targets = trajectory_targets(args.total_episodes)
        if any(existing_counts[name] > targets[name] for name in TRAJECTORY_TYPES):
            parser.error(f"Existing episode counts {existing_counts} exceed target quotas {targets}")
        remaining = {name: targets[name] - existing_counts[name] for name in TRAJECTORY_TYPES}
        print(f"Existing: {existing_counts}; target: {targets}; to collect: {remaining}", flush=True)
    def next_index(paths, prefix):
        numbers = [int(p.name[len(prefix):]) for p in paths if p.is_dir() and p.name[len(prefix):].isdigit()]
        return max(numbers, default=-1) + 1
    episode_index = next_index(output.glob("ep_*"), "ep_")
    attempt_index = next_index(output.glob("attempt_*"), "attempt_")
    attempt_index = max(attempt_index, next_index((output / "failures").glob("attempt_*"), "attempt_"))
    runs = output / "runs"
    types_path = runs / "episode_types.json"
    recorded_types = json.loads(types_path.read_text()) if types_path.exists() else {}
    run_index = max((int(p.stem[4:]) for p in runs.glob("run_*.json")
                     if p.stem[4:].isdigit()), default=-1) + 1
    for path in runs.glob("run_*.json"):
        previous = json.loads(path.read_text())
        attempt_index = max(attempt_index, previous.get("first_attempt_index", 0) + previous["attempts"])
    rng = np.random.default_rng(args.seed)
    results, successes, episode_types = [], 0, {}
    requested_successes = sum(remaining.values())
    if arm.mj.mj_name2id(arm.model, arm.mj.mjtObj.mjOBJ_BODY, "dynamic_pillar") >= 0:
        raise RuntimeError("Collection requires the obstacle-free AGV scene")
    if args.headless:
        context = nullcontext(None)
    else:
        import mujoco.viewer
        context = mujoco.viewer.launch_passive(arm.model, arm.data)
    with context as viewer:
        if viewer is not None:
            with viewer.lock():
                viewer.cam.lookat[:] = [.48, 0., .98]
                viewer.cam.distance = 2.7
                viewer.cam.azimuth = 145
                viewer.cam.elevation = -27
                viewer.opt.geomgroup[3:] = 0
        attempt = 0
        while successes < requested_successes and (args.max_attempts is None or attempt < args.max_attempts):
            trajectory_type = next(name for name in TRAJECTORY_TYPES if remaining[name] > 0)
            folder = output / f"attempt_{attempt_index + attempt:04d}"
            instruction = str(rng.choice(COLLECTION_INSTRUCTIONS))
            recorder = Recorder(arm, folder, images=not args.no_images, instruction=instruction)
            try:
                result = execute_episode(arm, rng, args.fixed_eval, recorder, viewer, args.realtime,
                                         trajectory_type, args.drop_location)
            except RuntimeError as exc:
                result = {"success": False, "reason": str(exc)}
            recorder.finish(arm, {key: value for key, value in result.items()
                                  if key not in ("trajectory_type", "recovery_event_frame", "drop_location",
                                                 "miss_offset_xy")})
            if result["success"]:
                episode_name = f"ep_{episode_index + successes:04d}"
                folder.rename(output / episode_name)
                episode_types[episode_name] = trajectory_type
                recorded_types[episode_name] = trajectory_type
                runs.mkdir(exist_ok=True)
                types_path.write_text(json.dumps(recorded_types, indent=2) + "\n")
                remaining[trajectory_type] -= 1
                successes += 1
            else:
                (output / "failures").mkdir(exist_ok=True)
                folder.rename(output / "failures" / folder.name)
            results.append(result)
            print(f"Attempt {attempt + 1} [{trajectory_type}]: {result['reason']} "
                  f"({successes}/{requested_successes})", flush=True)
            attempt += 1
            if viewer is not None and not viewer.is_running():
                break
    summary = {"requested_successes": requested_successes, "successes": successes,
               "total_episodes_target": args.total_episodes, "trajectory_targets": targets,
               "existing_trajectory_counts": existing_counts, "episode_types": episode_types,
               "attempts": len(results), "first_episode_index": episode_index,
               "first_attempt_index": attempt_index, "seed": args.seed, "fixed_eval": args.fixed_eval,
               "training_data": not args.no_images,
               "sampling": "Independent uniform handle-center XY and planar yaw in [pi/3, 2*pi/3); successes are feasibility-filtered",
               "yaw_quadrant_attempts": np.bincount([int(r["initial_yaw"] / (np.pi / 2)) % 4 for r in results if "initial_yaw" in r], minlength=4).tolist(),
               "yaw_quadrant_successes": np.bincount([int(r["initial_yaw"] / (np.pi / 2)) % 4 for r in results if r["success"]], minlength=4).tolist(),
               "coverage_warning": "Yaw is sampled only within 90 +/- 30 degrees; successful yaw distribution can be biased within that interval",
               "results": results}
    runs.mkdir(exist_ok=True)
    summary_path = runs / f"run_{run_index:04d}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Output: {output} (summary: {summary_path})")
    return 0 if successes == requested_successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
