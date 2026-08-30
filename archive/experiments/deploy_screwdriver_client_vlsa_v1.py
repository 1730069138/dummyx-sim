"""
VLSA / AEGIS first reproduction for the user's MuJoCo screwdriver pick-and-place scene.

Scope of this V1:
- Keeps the existing pi0 / OpenPI WebSocket policy and MuJoCo task unchanged.
- Reproduces the translational AEGIS control core:
    end-effector ellipsoid + obstacle ellipsoid + auxiliary sphere state + CBF-QP.
- Adapts the user's absolute joint-target VLA output to Cartesian velocity, applies
  the safety correction, then maps only the Cartesian correction back to joint space.
- Uses the known MuJoCo pillar geometry as an ORACLE obstacle ellipsoid for control-core
  validation. This is intentionally NOT the final VLSA perception reproduction.

Next version should replace build_oracle_obstacle_ellipsoid() with:
VLM -> GroundingDINO -> two-view RGB-D point cloud -> filtering -> MVEE.

Required runtime packages:
    mujoco, numpy, cv2, cvxpy, osqp, openpi_client
"""

import argparse
import os
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

try:
    import cvxpy as cp
except ImportError as exc:
    raise ImportError(
        "VLSA V1 requires cvxpy and OSQP. Install with: pip install cvxpy osqp"
    ) from exc

from openpi_client import websocket_client_policy


# =============================================================================
# Constants derived from the uploaded MuJoCo model / meshes
# =============================================================================
# Simulation timestep in dummy_apf.xml is 0.001 s and each policy action is held
# for 50 simulator substeps -> 0.05 s/action -> 20 Hz safety-control frequency.
SERVO_SUBSTEPS = 50

# End-effector ellipsoid, expressed in the link7 local frame.
# Derived from the union of piper_base.STL + both finger meshes over q=[0, 0.04].
# The axis-aligned ellipsoid is scaled to enclose all sampled mesh vertices and
# padded by 5 mm. Q_EEF_DIAG contains ellipsoid semi-axis lengths [m].
Q_EEF_DIAG = np.array([0.10463687, 0.05344414, 0.10119984], dtype=np.float64)
EEF_CENTER_IN_LINK7 = np.array([0.0, 0.00475000, 0.06999907], dtype=np.float64)

# dynamic_pillar is a MuJoCo box with half extents [0.02, 0.02, 0.08] m.
# The MVEE of a centered 3D box has semi-axes sqrt(3) * half_extents.
PILLAR_HALF_EXTENTS = np.array([0.02, 0.02, 0.08], dtype=np.float64)
Q_PILLAR_ORACLE_DIAG = np.sqrt(3.0) * PILLAR_HALF_EXTENTS

# AEGIS translational implementation settings.
CBF_ALPHA = 10.0
Z_ASCENT_GAIN = 10.0
QP_W_V = 1.0 / 25.0
QP_W_Z = 1.0

# Joint-space adapter. The original project already uses a DLS inverse in PACE.
DLS_RHO = 0.05
MAX_JOINT_DELTA = 0.25


# =============================================================================
# Math helpers: AEGIS ellipsoid CBF
# =============================================================================
def vector_hat(v):
    """Skew-symmetric matrix so that vector_hat(a) @ b == a x b."""
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def project_tangent(z):
    """Projection onto the tangent plane of the unit sphere at z."""
    z = z / (np.linalg.norm(z) + 1e-12)
    return np.eye(3) - np.outer(z, z)


def compute_h_ellipsoids(p_i, q_i_diag, r_i, p_j, q_j_diag, r_j, z, eps=1e-10):
    """
    AEGIS supporting-hyperplane CBF h for two ellipsoids.

    q_i_diag / q_j_diag are ellipsoid semi-axis lengths, and r_i / r_j map each
    ellipsoid's local axes into the world frame.
    """
    q_i = np.diag(q_i_diag)
    q_j = np.diag(q_j_diag)
    qbar_i = r_i @ q_i @ r_i.T
    qbar_j = r_j @ q_j @ r_j.T
    qbar_i_inv = np.linalg.inv(qbar_i)

    z = z / (np.linalg.norm(z) + eps)
    a = qbar_i_inv @ z
    denom = np.linalg.norm(a) + eps
    obstacle_support = np.linalg.norm(qbar_j @ a)
    center_term = (p_j - p_i).T @ a
    return float((-obstacle_support + center_term - 1.0) / denom)


def compute_cbf_coeffs_world_translation(
    p_i, q_i_diag, r_i, p_j, q_j_diag, r_j, z, eps=1e-10
):
    """
    Return coefficients for the translational AEGIS CBF-QP.

    We solve velocity directly in WORLD coordinates. Therefore the coefficient on
    v_world is eta_row (the official implementation multiplies eta by R_i because
    its optimization variable is expressed in the end-effector local frame).

    Returns:
        a_v_world: dh/d(v_world), shape (3,)
        a_u_z:      coefficient on the auxiliary control u_z, shape (3,)
        h:          current CBF value
        mu_row:     dh/dz, used for nominal auxiliary gradient-ascent control
    """
    q_i = np.diag(q_i_diag)
    q_j = np.diag(q_j_diag)
    qbar_i = r_i @ q_i @ r_i.T
    qbar_j = r_j @ q_j @ r_j.T

    qbar_i_inv = np.linalg.inv(qbar_i)
    qbar_i_inv2 = qbar_i_inv @ qbar_i_inv
    qbar_j2 = qbar_j @ qbar_j

    z = z / (np.linalg.norm(z) + eps)
    a_vec = qbar_i_inv @ z
    denom = np.linalg.norm(a_vec) + eps
    b_vec = qbar_j @ a_vec
    term1 = np.linalg.norm(b_vec) + eps
    sigma = term1 * denom + eps
    center_delta = p_j - p_i
    rho = 1.0 - center_delta.T @ a_vec + term1

    # dh / d p_i (world translation coefficient)
    eta_row = -(z.T @ qbar_i_inv) / denom

    term_mu_1 = (rho / (denom**3 + eps)) * (z.T @ qbar_i_inv2)
    term_mu_2 = (center_delta.T @ qbar_i_inv) / denom
    term_mu_3 = (
        z.T @ qbar_i_inv @ qbar_j2 @ qbar_i_inv
    ) / sigma
    mu_row = term_mu_1 + term_mu_2 - term_mu_3

    a_u_z = (mu_row @ project_tangent(z)).ravel()
    h = compute_h_ellipsoids(p_i, q_i_diag, r_i, p_j, q_j_diag, r_j, z)
    return eta_row.ravel(), a_u_z, h, mu_row.ravel()


def damped_pinv(jacobian, rho=DLS_RHO):
    """Damped least-squares right pseudoinverse for a 3x6 translational Jacobian."""
    return jacobian.T @ np.linalg.inv(
        jacobian @ jacobian.T + (rho**2) * np.eye(jacobian.shape[0])
    )


class AEGISTranslationalLayer:
    """Translational VLSA/AEGIS CBF-QP for one end-effector/obstacle ellipsoid pair."""

    def __init__(self, obstacle_center, obstacle_rotation, obstacle_axes, action_dt):
        self.p_obs = np.asarray(obstacle_center, dtype=np.float64).copy()
        self.r_obs = np.asarray(obstacle_rotation, dtype=np.float64).copy()
        self.q_obs = np.asarray(obstacle_axes, dtype=np.float64).copy()
        self.action_dt = float(action_dt)
        self.z = None

    def reset(self, eef_center):
        direction = self.p_obs - np.asarray(eef_center, dtype=np.float64)
        norm = np.linalg.norm(direction)
        self.z = direction / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])

    def filter(self, eef_center, eef_rotation, v_nominal_world):
        if self.z is None:
            self.reset(eef_center)

        a_v, a_u_z, h, mu_row = compute_cbf_coeffs_world_translation(
            eef_center,
            Q_EEF_DIAG,
            eef_rotation,
            self.p_obs,
            self.q_obs,
            self.r_obs,
            self.z,
        )

        u_z_nom = Z_ASCENT_GAIN * mu_row
        u_ref = np.hstack([v_nominal_world, u_z_nom])
        u = cp.Variable(6)
        weight = np.diag([QP_W_V, QP_W_V, QP_W_V, QP_W_Z, QP_W_Z, QP_W_Z])
        objective = cp.Minimize(cp.quad_form(u - u_ref, weight))
        constraints = [a_v @ u[:3] + a_u_z @ u[3:] + CBF_ALPHA * h >= 0.0]
        problem = cp.Problem(objective, constraints)

        t0 = time.perf_counter()
        qp_ok = True
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            qp_ok = u.value is not None and problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        except Exception:
            qp_ok = False
        qp_ms = (time.perf_counter() - t0) * 1000.0

        if qp_ok:
            v_safe = np.asarray(u.value[:3], dtype=np.float64)
            u_z = np.asarray(u.value[3:], dtype=np.float64)
        else:
            # Transparent fallback. We record qp_ok=False so such a rollout must not
            # be used as evidence of a safety guarantee.
            v_safe = np.asarray(v_nominal_world, dtype=np.float64).copy()
            u_z = u_z_nom.copy()

        dz = project_tangent(self.z) @ u_z
        self.z = self.z + dz * self.action_dt
        z_norm = np.linalg.norm(self.z)
        if z_norm > 1e-9:
            self.z /= z_norm

        intervention = float(np.linalg.norm(v_safe - v_nominal_world))
        return v_safe, {
            "h": h,
            "qp_ok": qp_ok,
            "qp_status": str(problem.status),
            "qp_ms": qp_ms,
            "intervention": intervention,
            "z": self.z.copy(),
        }


# =============================================================================
# MuJoCo geometry / action adapter
# =============================================================================
def get_body_rotation(data, body_id):
    return data.xmat[body_id].reshape(3, 3).copy()


def get_eef_ellipsoid_pose(data, link7_id):
    r_link7 = get_body_rotation(data, link7_id)
    p_link7 = data.xpos[link7_id].copy()
    p_center = p_link7 + r_link7 @ EEF_CENTER_IN_LINK7
    return p_center, r_link7


def get_point_jacobian(model, data, point_world, body_id):
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp, jacr, point_world, body_id)
    return jacp


def build_oracle_obstacle_ellipsoid(data, pillar_body_id):
    """
    V1 debug-only obstacle geometry.

    The pose is frozen at episode initialization, matching the original AEGIS flow
    where perception builds an obstacle ellipsoid before execution. V2 should replace
    this function with the full vision-language + RGB-D + MVEE pipeline.
    """
    p = data.xpos[pillar_body_id].copy()
    r = get_body_rotation(data, pillar_body_id)
    return p, r, Q_PILLAR_ORACLE_DIAG.copy()


def adapt_joint_target_through_vlsa(
    model,
    data,
    current_action,
    safety_layer,
    link7_id,
    action_dt,
):
    """
    Convert the user's absolute joint target to nominal Cartesian velocity, apply
    AEGIS, then map only the safety correction back into joint space.
    """
    q = data.qpos[:6].copy()
    q_delta_nom = np.clip(current_action[:6] - q, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
    qdot_nom = q_delta_nom / action_dt

    eef_center, eef_rotation = get_eef_ellipsoid_pose(data, link7_id)
    jacp_full = get_point_jacobian(model, data, eef_center, link7_id)
    j_pos = jacp_full[:, :6]
    v_nom = j_pos @ qdot_nom

    v_safe, debug = safety_layer.filter(eef_center, eef_rotation, v_nom)
    delta_v = v_safe - v_nom
    delta_qdot = damped_pinv(j_pos, DLS_RHO) @ delta_v
    q_delta_safe = (qdot_nom + delta_qdot) * action_dt
    q_delta_safe = np.clip(q_delta_safe, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)

    ctrl = np.zeros(8, dtype=np.float64)
    ctrl[:6] = q + q_delta_safe
    gripper_val = 0.04 if current_action[6] > 0.02 else 0.0
    ctrl[6:8] = gripper_val

    debug.update(
        {
            "eef_center": eef_center.copy(),
            "v_nom": v_nom.copy(),
            "v_safe": v_safe.copy(),
            "q_delta_nom": q_delta_nom.copy(),
            "q_delta_safe": q_delta_safe.copy(),
        }
    )
    return ctrl, debug


def build_baseline_ctrl(data, current_action):
    q = data.qpos[:6].copy()
    q_delta = np.clip(current_action[:6] - q, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
    ctrl = np.zeros(8, dtype=np.float64)
    ctrl[:6] = q + q_delta
    ctrl[6:8] = 0.04 if current_action[6] > 0.02 else 0.0
    return ctrl


# =============================================================================
# Existing paper metrics retained from the user's deployment script
# =============================================================================
def get_target_table_force(model, data, valid_body_ids):
    f_xyz = np.zeros(3)
    for i in range(data.ncon):
        contact = data.contact[i]
        if 0.18 < contact.pos[2] < 0.22:
            geom1 = contact.geom1
            geom2 = contact.geom2
            body1 = model.geom_bodyid[geom1]
            body2 = model.geom_bodyid[geom2]
            if body1 in valid_body_ids or body2 in valid_body_ids:
                c_force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, i, c_force)
                c_mat = contact.frame.reshape(3, 3)
                f_world = c_mat.T @ c_force[:3]
                if f_world[2] < 0:
                    f_world = -f_world
                f_xyz += f_world
    return f_xyz


def reset_scene(model, data, use_obstacle=True, target_xy=None):
    if target_xy is not None:
        target_x, target_y = target_xy
    else:
        target_x = np.random.uniform(0.20, 0.35)
        target_y = np.random.uniform(-0.15, -0.05)

    target_z = 0.22
    target_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fj_screwdriver")
    if target_jnt_id != -1:
        target_adr = model.jnt_qposadr[target_jnt_id]
        data.qpos[target_adr : target_adr + 3] = [target_x, target_y, target_z]
        data.qpos[target_adr + 3 : target_adr + 7] = [1, 0, 0, 0]

    pillar_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pillar_joint")
    if pillar_jnt_id != -1:
        q_adr = model.jnt_qposadr[pillar_jnt_id]
        v_adr = model.jnt_dofadr[pillar_jnt_id]
        if use_obstacle:
            data.qpos[q_adr : q_adr + 3] = [0.11, -0.1, 0.38]
            data.qpos[q_adr + 3 : q_adr + 7] = [1, 0, 0, 0]
        else:
            data.qpos[q_adr : q_adr + 3] = [10.0, 10.0, -10.0]
        data.qvel[v_adr : v_adr + 6] = 0.0

    data.qpos[:8] = 0.0
    data.qvel[:8] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def save_episode_video(frames, folder, episode_idx):
    if not frames:
        return
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"ep_{episode_idx:03d}_{time.strftime('%H%M%S')}.mp4")
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


# =============================================================================
# Main deployment / evaluation loop
# =============================================================================
def main(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = args.xml_path or os.path.join(base_dir, "dummyx_apf_scene.xml")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer_rgb = mujoco.Renderer(model, height=256, width=256)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    action_dt = SERVO_SUBSTEPS * model.opt.timestep
    if not np.isclose(action_dt, 0.05, atol=1e-8):
        print(
            f"⚠️ Current action period is {action_dt:.6f} s, not 0.05 s. "
            "The adapter will use the actual MuJoCo period."
        )

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "real_screwdriver")
    pillar_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_pillar")
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    link7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    link8_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link8")
    link9_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link9")

    if min(target_body_id, pillar_body_id, tcp_id, link7_id) < 0:
        raise RuntimeError("Required MuJoCo body/site names are missing from the XML model.")

    valid_collision_bodies = [target_body_id, link7_id]
    if link8_id != -1:
        valid_collision_bodies.append(link8_id)
    if link9_id != -1:
        valid_collision_bodies.append(link9_id)

    fixed_targets = None
    if args.fixed_eval:
        fixed_xy = (0.28, -0.10)
        fixed_targets = [fixed_xy for _ in range(args.num_episodes)]
        print(f"🔒 Fixed evaluation target: X={fixed_xy[0]:.2f}, Y={fixed_xy[1]:.2f}")

    use_obstacle = not args.no_obstacle
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_{args.mode}_oracleVLSA_obs{int(use_obstacle)}"
    record_root = os.path.join(base_dir, "recordings", run_name)
    os.makedirs(record_root, exist_ok=True)

    success_count = 0
    collision_count = 0
    episode_peak_forces = []
    succ_peak_forces = []
    all_impulses = []
    succ_impulses = []
    episode_peak_torques = []
    all_qp_times = []
    all_h_values = []
    qp_failures = 0

    print("\n" + "=" * 84)
    print("VLSA/AEGIS V1 — translational CBF-QP control-core reproduction")
    print(f"mode={args.mode} | obstacle={use_obstacle} | episodes={args.num_episodes}")
    if args.mode == "vlsa":
        print("Obstacle source: ORACLE MuJoCo pillar ellipsoid (debug only, not final VLSA perception)")
        print(f"EEF ellipsoid semi-axes: {Q_EEF_DIAG} m")
        print(f"EEF center in link7:     {EEF_CENTER_IN_LINK7} m")
        print(f"Pillar ellipsoid axes:   {Q_PILLAR_ORACLE_DIAG} m")
    print("=" * 84 + "\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for episode in range(args.num_episodes):
            if not viewer.is_running():
                break

            target_xy = fixed_targets[episode] if fixed_targets is not None else None
            reset_scene(model, data, use_obstacle=use_obstacle, target_xy=target_xy)
            viewer.sync()

            safety_layer = None
            if args.mode == "vlsa" and use_obstacle:
                p_obs, r_obs, q_obs = build_oracle_obstacle_ellipsoid(data, pillar_body_id)
                eef_center0, _ = get_eef_ellipsoid_pose(data, link7_id)
                safety_layer = AEGISTranslationalLayer(p_obs, r_obs, q_obs, action_dt)
                safety_layer.reset(eef_center0)

            step_counter = 0
            in_box_counter = 0
            is_success = False
            episode_collision = False
            action_chunk_cache = None
            video_frames = []

            ep_peak_force = 0.0
            ep_impulse = 0.0
            ep_torques = []
            ep_qp_times = []
            ep_h = []
            ep_interventions = []
            ep_qp_ok = []
            ep_v_nom = []
            ep_v_safe = []
            ep_z = []
            ep_q_pos_vla = []
            ep_q_dot_vla = []

            record_cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, record_cam)
            record_cam.lookat[:] = [0.25, -0.05, 0.22]
            record_cam.distance = 1.0
            record_cam.azimuth = 180
            record_cam.elevation = -20

            while viewer.is_running() and step_counter < args.max_steps:
                if step_counter % 8 == 0 or action_chunk_cache is None:
                    renderer_rgb.update_scene(data, camera="rear_cam")
                    img_external = renderer_rgb.render()
                    renderer_rgb.update_scene(data, camera="wrist_cam")
                    img_wrist = renderer_rgb.render()
                    result = policy.infer(
                        {
                            "observation/image": img_external,
                            "observation/wrist_image": img_wrist,
                            "observation/state": data.qpos[:8].copy(),
                            "prompt": "Pick up the screwdriver and drop it into the box.",
                        }
                    )
                    action_chunk_cache = result["actions"]

                renderer_rgb.update_scene(data, camera=record_cam)
                video_frames.append(renderer_rgb.render())

                current_action = action_chunk_cache[step_counter % 8]
                raw_q_pos_vla = current_action[:6].copy()
                raw_q_dot_vla = raw_q_pos_vla - data.qpos[:6].copy()
                ep_q_pos_vla.append(raw_q_pos_vla)
                ep_q_dot_vla.append(raw_q_dot_vla)

                debug = None
                if safety_layer is not None:
                    base_ctrl, debug = adapt_joint_target_through_vlsa(
                        model,
                        data,
                        current_action,
                        safety_layer,
                        link7_id,
                        action_dt,
                    )
                    ep_qp_times.append(debug["qp_ms"])
                    ep_h.append(debug["h"])
                    ep_interventions.append(debug["intervention"])
                    ep_qp_ok.append(debug["qp_ok"])
                    ep_v_nom.append(debug["v_nom"])
                    ep_v_safe.append(debug["v_safe"])
                    ep_z.append(debug["z"])
                    if not debug["qp_ok"]:
                        qp_failures += 1
                else:
                    base_ctrl = build_baseline_ctrl(data, current_action)

                max_force_step = 0.0
                max_force_xyz_step = np.zeros(3)
                for _ in range(SERVO_SUBSTEPS):
                    current_f_xyz = get_target_table_force(model, data, valid_collision_bodies)
                    force_norm = np.linalg.norm(current_f_xyz)
                    if force_norm > max_force_step:
                        max_force_step = force_norm
                        max_force_xyz_step = current_f_xyz.copy()
                    ep_peak_force = max(ep_peak_force, force_norm)
                    ep_impulse += force_norm * model.opt.timestep

                    data.ctrl[:8] = base_ctrl
                    mujoco.mj_step(model, data)
                    tau = data.qfrc_actuator[:6].copy()
                    ep_torques.append(np.max(np.abs(tau)))

                if use_obstacle:
                    pillar_up = data.xmat[pillar_body_id].reshape(3, 3)[2, 2]
                    if pillar_up < 0.9:
                        episode_collision = True

                current_tcp = data.site_xpos[tcp_id].copy()
                screw_pos = data.xpos[target_body_id]
                in_box = (
                    abs(screw_pos[0] - (-0.05)) < 0.12
                    and abs(screw_pos[1]) < 0.12
                    and screw_pos[2] < 0.34
                )
                is_released = (current_action[6] > 0.02) or (
                    np.linalg.norm(current_tcp - screw_pos) > 0.06
                )
                in_box_counter = in_box_counter + 1 if in_box and is_released else 0
                if in_box_counter >= 5:
                    is_success = True
                    break

                viewer.sync()
                step_counter += 1

            if is_success:
                success_count += 1
            if episode_collision:
                collision_count += 1

            ep_peak_tau = float(np.max(ep_torques)) if ep_torques else 0.0
            episode_peak_torques.append(ep_peak_tau)
            episode_peak_forces.append(ep_peak_force)
            all_impulses.append(ep_impulse)
            if is_success:
                succ_peak_forces.append(ep_peak_force)
                succ_impulses.append(ep_impulse)

            if ep_qp_times:
                all_qp_times.extend(ep_qp_times)
                all_h_values.extend(ep_h)

            save_folder = os.path.join(record_root, "success" if is_success else "fail")
            os.makedirs(save_folder, exist_ok=True)
            save_episode_video(video_frames, save_folder, episode + 1)

            np.savez(
                os.path.join(save_folder, f"data_ep_{episode + 1:03d}.npz"),
                episode_success=np.array(is_success),
                episode_knockdown=np.array(episode_collision),
                episode_peak_contact_force=np.array(ep_peak_force),
                episode_contact_impulse=np.array(ep_impulse),
                episode_peak_joint_torque=np.array(ep_peak_tau),
                q_pos_vla=np.asarray(ep_q_pos_vla),
                q_dot_vla=np.asarray(ep_q_dot_vla),
                v_nom=np.asarray(ep_v_nom),
                v_safe=np.asarray(ep_v_safe),
                cbf_h=np.asarray(ep_h),
                qp_time_ms=np.asarray(ep_qp_times),
                qp_ok=np.asarray(ep_qp_ok, dtype=bool),
                intervention_norm=np.asarray(ep_interventions),
                z_state=np.asarray(ep_z),
                eef_axes=Q_EEF_DIAG,
                eef_center_in_link7=EEF_CENTER_IN_LINK7,
                obstacle_axes=Q_PILLAR_ORACLE_DIAG,
                obstacle_source=np.array("oracle_mujoco_box_mvee"),
            )

            qp_text = ""
            if ep_qp_times:
                qp_text = (
                    f" | h_min={np.min(ep_h):.4f}"
                    f" | QP={np.mean(ep_qp_times):.3f} ms"
                    f" | intervention={np.mean(ep_interventions):.4f} m/s"
                )
            print(
                f"Episode {episode + 1}/{args.num_episodes}: "
                f"{'SUCCESS' if is_success else 'FAIL'} | "
                f"{'COLLISION' if episode_collision else 'NO COLLISION'} | "
                f"peakF={ep_peak_force:.2f} N | impulse={ep_impulse:.2f} N*s | "
                f"peakTau={ep_peak_tau:.2f} N*m{qp_text}"
            )

    episodes_done = max(len(episode_peak_forces), 1)
    sr = 100.0 * success_count / episodes_done
    cr = 100.0 * collision_count / episodes_done if use_obstacle else None
    avg_peak_f_succ = float(np.mean(succ_peak_forces)) if succ_peak_forces else None
    avg_imp_all = float(np.mean(all_impulses)) if all_impulses else 0.0
    avg_imp_succ = float(np.mean(succ_impulses)) if succ_impulses else None
    std_imp_succ = float(np.std(succ_impulses)) if succ_impulses else None
    avg_tau = float(np.mean(episode_peak_torques)) if episode_peak_torques else 0.0
    avg_qp_ms = float(np.mean(all_qp_times)) if all_qp_times else None
    min_h = float(np.min(all_h_values)) if all_h_values else None

    lines = [
        "================ VLSA/AEGIS V1 SUMMARY ================",
        f"mode: {args.mode}",
        f"obstacle: {use_obstacle}",
        f"obstacle_source: {'oracle_mujoco_box_mvee' if args.mode == 'vlsa' else 'none'}",
        f"episodes: {episodes_done}",
        f"SR: {sr:.1f}%",
        f"CR: {cr:.1f}%" if cr is not None else "CR: —",
        f"f_hat_success: {avg_peak_f_succ:.2f} N" if avg_peak_f_succ is not None else "f_hat_success: —",
        f"I_all: {avg_imp_all:.2f} N*s",
        f"I_success: {avg_imp_succ:.2f} N*s" if avg_imp_succ is not None else "I_success: —",
        f"sigma_I_success: {std_imp_succ:.2f} N*s" if std_imp_succ is not None else "sigma_I_success: —",
        f"tau_bar_max: {avg_tau:.2f} N*m",
        f"QP_mean: {avg_qp_ms:.3f} ms" if avg_qp_ms is not None else "QP_mean: —",
        f"h_min: {min_h:.6f}" if min_h is not None else "h_min: —",
        f"QP_failures: {qp_failures}",
        "========================================================",
    ]
    summary = "\n".join(lines)
    print("\n" + summary + "\n")

    summary_path = os.path.join(record_root, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLSA/AEGIS V1 control-core reproduction for screwdriver pick-and-place"
    )
    parser.add_argument("--host", default="localhost", help="OpenPI WebSocket server IP")
    parser.add_argument("--port", default=8000, type=int, help="OpenPI WebSocket server port")
    parser.add_argument("--num_episodes", default=10, type=int, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", default=600, type=int, help="Maximum policy steps per episode")
    parser.add_argument(
        "--mode",
        choices=["baseline", "vlsa"],
        default="vlsa",
        help="baseline: original VLA adapter; vlsa: AEGIS translational CBF-QP",
    )
    parser.add_argument("--fixed_eval", action="store_true", help="Use fixed screwdriver XY=(0.28,-0.10)")
    parser.add_argument("--no_obstacle", action="store_true", help="Remove dynamic pillar")
    parser.add_argument(
        "--xml_path",
        default=None,
        help="Optional explicit path to dummyx_apf_scene.xml",
    )
    main(parser.parse_args())
