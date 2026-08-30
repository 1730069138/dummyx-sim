"""
VLSA / AEGIS V2 reproduction for the user's MuJoCo screwdriver pick-and-place scene.

What V2 adds over V1.1
----------------------
1) Full VLSA perception path for obstacle geometry:
   rear_cam RGB + fixed RGB
       -> VLM obstacle identification (GLM-4.5V, optional fixed text override for debug)
       -> GroundingDINO 2D grounding in both views
       -> MuJoCo RGB-D back-projection into world coordinates
       -> workspace crop
       -> remove farthest 20% from centroid
       -> DBSCAN largest cluster
       -> convex hull
       -> CVXPY MVEE
       -> obstacle ellipsoid (center, rotation, semi-axes)

2) The V1.1 full-action AEGIS CBF-QP control core is retained:
   pi0 joint targets -> nominal 6D EEF twist -> 9D AEGIS QP
   -> safe 6D twist -> DLS joint correction -> position actuator command.

3) Live MuJoCo debug visualization (NO generated debug images):
   - EEF ellipsoid: translucent blue
   - controlled obstacle MVEE: translucent red
   - optional oracle reference ellipsoid: translucent yellow
   Use --show_ellipsoids and optionally --show_oracle_reference.

Scientific-use note
-------------------
- --obstacle_source perception is the intended VLSA reproduction.
- --obstacle_source oracle is retained only to isolate/control-test the CBF-QP.
- --obstacle_text bypasses the VLM and is DEBUG ONLY; omit it for the final
  VLSA perception experiment.
- If perception fails in an episode, the safety layer is disabled for that
  episode (rather than silently falling back to oracle geometry), matching
  the fact that upstream perception failure is part of the method's behavior.

Required core packages:
    mujoco, numpy, cv2, cvxpy, osqp, scipy, scikit-learn, openpi_client

Additional packages for perception:
    torch, torchvision, pillow, GroundingDINO

Additional package for the original VLM identification stage:
    zai
and environment variable:
    ZHIPU_API_KEY
"""

import argparse
import base64
import csv
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
        "VLSA V2 requires cvxpy. Install with: pip install cvxpy osqp scs"
    ) from exc

from openpi_client import websocket_client_policy


# =============================================================================
# Constants derived from the uploaded MuJoCo model / meshes
# =============================================================================
SERVO_SUBSTEPS = 50

# EEF ellipsoid in link7 local frame, derived from the uploaded gripper meshes.
Q_EEF_DIAG = np.array([0.10463687, 0.05344414, 0.10119984], dtype=np.float64)
EEF_CENTER_IN_LINK7 = np.array([0.0, 0.00475000, 0.06999907], dtype=np.float64)

# Oracle reference only. dynamic_pillar box half extents in the user's XML.
PILLAR_HALF_EXTENTS = np.array([0.02, 0.02, 0.08], dtype=np.float64)
Q_PILLAR_ORACLE_DIAG = np.sqrt(3.0) * PILLAR_HALF_EXTENTS

# AEGIS settings following the released full-action implementation structure.
CBF_ALPHA = 10.0
Z_ASCENT_GAIN = 10.0
QP_W_V = 1.0 / 25.0
QP_W_Z = 1.0

DLS_RHO = 0.05
MAX_JOINT_DELTA = 0.25

TASK_PROMPT = "Pick up the screwdriver and drop it into the box."


# =============================================================================
# AEGIS ellipsoid-CBF math
# =============================================================================
def vector_hat(v):
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def project_tangent(z):
    z = z / (np.linalg.norm(z) + 1e-12)
    return np.eye(3) - np.outer(z, z)


def compute_h_ellipsoids(p_i, q_i_diag, r_i, p_j, q_j_diag, r_j, z, eps=1e-10):
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


def compute_cbf_coeffs_world_twist(
    p_i, q_i_diag, r_i, p_j, q_j_diag, r_j, z, eps=1e-10
):
    """
    Official AEGIS derivative structure, expressed for world-frame v and omega.

    The released code returns coefficients for local/body-frame control variables
    after multiplying by R_i. MuJoCo mj_jac() gives world-frame linear/angular
    velocity, so this adapter uses eta_row and zeta_tilde directly.
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

    eta_row = -(z.T @ qbar_i_inv) / denom

    term_mu_1 = (rho / (denom**3 + eps)) * (z.T @ qbar_i_inv2)
    term_mu_2 = (center_delta.T @ qbar_i_inv) / denom
    term_mu_3 = (
        z.T @ qbar_i_inv @ qbar_j2 @ qbar_i_inv
    ) / sigma
    mu_row = term_mu_1 + term_mu_2 - term_mu_3

    tmp1 = z.T @ qbar_i_inv2 @ vector_hat(z)
    left_vec = z.T @ qbar_i_inv @ qbar_j2
    ja_vec = vector_hat(a_vec)
    tmp2 = left_vec @ (ja_vec - qbar_i_inv @ vector_hat(z))
    part_a = center_delta.T @ qbar_i_inv @ vector_hat(z)
    part_b = z.T @ qbar_i_inv @ vector_hat(center_delta)
    tmp3 = part_a + part_b
    zeta_tilde = (
        rho * tmp1 / (denom**3 + eps)
        + tmp2 / sigma
        + tmp3 / denom
    )

    a_u_z = (mu_row @ project_tangent(z)).ravel()
    h = compute_h_ellipsoids(p_i, q_i_diag, r_i, p_j, q_j_diag, r_j, z)

    return (
        eta_row.ravel(),
        np.asarray(zeta_tilde).ravel(),
        a_u_z,
        h,
        mu_row.ravel(),
    )


def damped_pinv(jacobian, rho=DLS_RHO):
    return jacobian.T @ np.linalg.inv(
        jacobian @ jacobian.T + (rho**2) * np.eye(jacobian.shape[0])
    )


class AEGISFullActionLayer:
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

    def update_obstacle_pose(self, center, rotation, axes=None):
        """Update the obstacle ellipsoid pose without resetting the CBF auxiliary state."""
        self.p_obs = np.asarray(center, dtype=np.float64).copy()
        self.r_obs = np.asarray(rotation, dtype=np.float64).copy()
        if axes is not None:
            self.q_obs = np.asarray(axes, dtype=np.float64).copy()

    def current_h(self, eef_center, eef_rotation):
        if self.z is None:
            self.reset(eef_center)
        return compute_h_ellipsoids(
            eef_center,
            Q_EEF_DIAG,
            eef_rotation,
            self.p_obs,
            self.q_obs,
            self.r_obs,
            self.z,
        )

    def filter(self, eef_center, eef_rotation, v_nominal_world, omega_nominal_world):
        if self.z is None:
            self.reset(eef_center)

        a_v, a_omega, a_u_z, h, mu_row = compute_cbf_coeffs_world_twist(
            eef_center,
            Q_EEF_DIAG,
            eef_rotation,
            self.p_obs,
            self.q_obs,
            self.r_obs,
            self.z,
        )

        u_z_nom = Z_ASCENT_GAIN * mu_row
        u_ref = np.hstack([v_nominal_world, omega_nominal_world, u_z_nom])

        u = cp.Variable(9)
        weight = np.diag(
            [
                QP_W_V, QP_W_V, QP_W_V,
                QP_W_V, QP_W_V, QP_W_V,
                QP_W_Z, QP_W_Z, QP_W_Z,
            ]
        )
        objective = cp.Minimize(cp.quad_form(u - u_ref, weight))
        constraints = [
            a_v @ u[:3]
            + a_omega @ u[3:6]
            + a_u_z @ u[6:9]
            + CBF_ALPHA * h
            >= 0.0
        ]
        problem = cp.Problem(objective, constraints)

        t0 = time.perf_counter()
        qp_ok = True
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            qp_ok = u.value is not None and problem.status in (
                cp.OPTIMAL,
                cp.OPTIMAL_INACCURATE,
            )
        except Exception:
            qp_ok = False
        qp_ms = (time.perf_counter() - t0) * 1000.0

        if qp_ok:
            v_safe = np.asarray(u.value[:3], dtype=np.float64)
            omega_safe = np.asarray(u.value[3:6], dtype=np.float64)
            u_z = np.asarray(u.value[6:9], dtype=np.float64)
        else:
            v_safe = np.asarray(v_nominal_world, dtype=np.float64).copy()
            omega_safe = np.asarray(omega_nominal_world, dtype=np.float64).copy()
            u_z = u_z_nom.copy()

        dz = project_tangent(self.z) @ u_z
        self.z = self.z + dz * self.action_dt
        z_norm = np.linalg.norm(self.z)
        if z_norm > 1e-9:
            self.z /= z_norm

        delta_twist = np.hstack(
            [v_safe - v_nominal_world, omega_safe - omega_nominal_world]
        )

        return v_safe, omega_safe, {
            "h": h,
            "qp_ok": qp_ok,
            "qp_status": str(problem.status),
            "qp_ms": qp_ms,
            "intervention": float(np.linalg.norm(delta_twist)),
            "z": self.z.copy(),
        }


# =============================================================================
# MuJoCo kinematics / geometry
# =============================================================================
def get_body_rotation(data, body_id):
    return data.xmat[body_id].reshape(3, 3).copy()


def get_eef_ellipsoid_pose(data, link7_id):
    r_link7 = get_body_rotation(data, link7_id)
    p_link7 = data.xpos[link7_id].copy()
    p_center = p_link7 + r_link7 @ EEF_CENTER_IN_LINK7
    return p_center, r_link7


def get_point_jacobians(model, data, point_world, body_id):
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp, jacr, point_world, body_id)
    return jacp, jacr


def build_oracle_obstacle_ellipsoid(data, pillar_body_id):
    p = data.xpos[pillar_body_id].copy()
    r = get_body_rotation(data, pillar_body_id)
    return p, r, Q_PILLAR_ORACLE_DIAG.copy()


# =============================================================================
# Live ellipsoid visualization in the MuJoCo viewer
# =============================================================================
def _append_debug_ellipsoid(scene, center, rotation, axes, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        np.asarray(axes, dtype=np.float64),
        np.asarray(center, dtype=np.float64),
        np.asarray(rotation, dtype=np.float64).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def update_debug_ellipsoids(
    viewer,
    data,
    link7_id,
    safety_layer,
    show_ellipsoids,
    show_oracle_reference,
    pillar_body_id,
):
    """
    Draw directly in viewer.user_scn. Nothing is saved as an image.

    Colors:
      EEF:                  blue
      control obstacle:     red
      oracle reference:     yellow
    """
    viewer.user_scn.ngeom = 0
    if not show_ellipsoids:
        return

    eef_center, eef_rotation = get_eef_ellipsoid_pose(data, link7_id)
    _append_debug_ellipsoid(
        viewer.user_scn,
        eef_center,
        eef_rotation,
        Q_EEF_DIAG,
        [0.05, 0.35, 1.0, 0.28],
    )

    if safety_layer is not None:
        _append_debug_ellipsoid(
            viewer.user_scn,
            safety_layer.p_obs,
            safety_layer.r_obs,
            safety_layer.q_obs,
            [1.0, 0.1, 0.1, 0.28],
        )

    if show_oracle_reference and pillar_body_id >= 0:
        p_ref, r_ref, q_ref = build_oracle_obstacle_ellipsoid(data, pillar_body_id)
        _append_debug_ellipsoid(
            viewer.user_scn,
            p_ref,
            r_ref,
            q_ref,
            [1.0, 0.85, 0.05, 0.18],
        )


# =============================================================================
# VLSA perception: VLM -> GroundingDINO -> RGB-D -> filter -> MVEE
# =============================================================================
class GroundingDINOWrapper:
    def __init__(self, config_path, checkpoint_path, device):
        try:
            import torch
            from PIL import Image
            import groundingdino.datasets.transforms as T
            from groundingdino.util.inference import load_model, predict
        except ImportError as exc:
            raise ImportError(
                "Perception mode requires GroundingDINO, torch, torchvision and pillow."
            ) from exc

        self.torch = torch
        self.Image = Image
        self.T = T
        self.predict_fn = predict
        self.device = device
        self.model = load_model(config_path, checkpoint_path, device=device)

        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def detect_best_box(self, image_rgb, caption, box_threshold, text_threshold):
        image_pil = self.Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
        image_tensor, _ = self.transform(image_pil, None)

        boxes, logits, phrases = self.predict_fn(
            model=self.model,
            image=image_tensor,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        if boxes is None or len(boxes) == 0:
            return None

        logits_np = logits.detach().cpu().numpy()
        best = int(np.argmax(logits_np))
        cx, cy, bw, bh = boxes[best].detach().cpu().numpy()

        h, w = image_rgb.shape[:2]
        x1 = int(np.floor((cx - bw / 2.0) * w))
        y1 = int(np.floor((cy - bh / 2.0) * h))
        x2 = int(np.ceil((cx + bw / 2.0) * w))
        y2 = int(np.ceil((cy + bh / 2.0) * h))

        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, x1 + 1, w))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, y1 + 1, h))

        phrase = phrases[best] if phrases and best < len(phrases) else caption
        return {
            "xyxy": (x1, y1, x2, y2),
            "confidence": float(logits_np[best]),
            "phrase": str(phrase),
        }


def identify_obstacle_with_glm(image_rgb, instruction):
    """
    Reproduce the released AEGIS VLM obstacle-identification stage without
    writing an intermediate image file.
    """
    try:
        from zai import ZhipuAiClient
    except ImportError as exc:
        raise ImportError(
            "VLM obstacle identification requires the 'zai' package. "
            "Install it or use --obstacle_text for a debug fixed query."
        ) from exc

    api_key = os.environ.get("ZHIPU_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ZHIPU_API_KEY is not set. For a debug run you may pass "
            "--obstacle_text 'gray pillar'; omit that override for the final VLSA run."
        )

    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    if not ok:
        raise RuntimeError("Failed to encode VLM input image.")

    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    client = ZhipuAiClient(api_key=api_key)

    preferred = [
        "gray pillar",
        "screwdriver",
        "storage box",
    ]

    response = client.chat.completions.create(
        model="glm-4.5v",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"The robot must follow this instruction: {instruction}. "
                            "Based on both the instruction and the image, identify exactly "
                            "one non-robot object that is most likely to obstruct the robot's "
                            "motion during task execution. Output a uniquely identifiable "
                            "obstacle name including color and object type, preferably from "
                            f"this list when applicable: {preferred}. Output only the object "
                            "name, with no additional words."
                        ),
                    },
                ],
            }
        ],
        temperature=0.1,
        top_p=0.1,
        thinking={"type": "enabled"},
    )

    text = response.choices[0].message.content
    return (
        text.replace("<|begin_of_box|>", "")
        .replace("<|end_of_box|>", "")
        .strip()
    )


def render_rgb_depth(renderer, data, camera_name):
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    return rgb, depth


def camera_intrinsics_from_fovy(model, camera_id, height, width):
    fovy = np.deg2rad(float(model.cam_fovy[camera_id]))
    fy = 0.5 * height / np.tan(0.5 * fovy)
    fx = fy  # MuJoCo assumes square pixels for this camera model.
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return fx, fy, cx, cy


def bbox_depth_to_world_points(model, data, camera_name, depth, bbox_xyxy):
    """
    MuJoCo fixed cameras look along local -Z, with +X right and +Y up.
    Renderer depth is metric distance to the camera plane (optical-axis depth).
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise RuntimeError(f"Camera not found: {camera_name}")

    h, w = depth.shape
    fx, fy, cx, cy = camera_intrinsics_from_fovy(model, cam_id, h, w)

    x1, y1, x2, y2 = bbox_xyxy
    uu, vv = np.meshgrid(
        np.arange(x1, x2, dtype=np.float64),
        np.arange(y1, y2, dtype=np.float64),
    )
    zz = depth[y1:y2, x1:x2].astype(np.float64)

    valid = np.isfinite(zz) & (zz > 1e-5)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)

    u = uu[valid]
    v = vv[valid]
    z_depth = zz[valid]

    x_cam = (u - cx) * z_depth / fx
    y_cam = -(v - cy) * z_depth / fy
    z_cam = -z_depth

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    p_cam = data.cam_xpos[cam_id].copy()
    r_cam = data.cam_xmat[cam_id].reshape(3, 3).copy()
    points_world = points_cam @ r_cam.T + p_cam
    return points_world


def filter_obstacle_points(points, args):
    """
    VLSA preprocessing structure:
      workspace crop -> retain nearest 80% to centroid -> DBSCAN largest cluster.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        return points

    xmin, xmax, ymin, ymax, zmin, zmax = args.workspace
    keep = (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
        & (points[:, 2] >= zmin)
        & (points[:, 2] <= zmax)
    )
    points = points[keep]
    if len(points) == 0:
        return points

    # Released VLSA code removes the farthest 20% before DBSCAN.
    center = points.mean(axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    keep_count = max(1, int(len(points) * 0.8))
    points = points[np.argsort(distances)[:keep_count]]

    if len(points) < args.dbscan_min_samples:
        return points

    try:
        from sklearn.cluster import DBSCAN
    except ImportError as exc:
        raise ImportError(
            "Perception mode requires scikit-learn for DBSCAN: pip install scikit-learn"
        ) from exc

    labels = DBSCAN(
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
    ).fit_predict(points)

    valid_labels = labels[labels >= 0]
    if len(valid_labels) > 0:
        largest = np.bincount(valid_labels).argmax()
        points = points[labels == largest]

    return points


def mvee_cvxpy(points):
    """Same convex MVEE parameterization as the released VLSA utility."""
    points = np.asarray(points, dtype=np.float64)
    n, d = points.shape
    m_var = cp.Variable((d, d), PSD=True)
    g_var = cp.Variable(d)

    objective = cp.Minimize(-cp.log_det(m_var))
    constraints = [cp.norm(m_var @ points[i] - g_var) <= 1 for i in range(n)]
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.SCS, verbose=False)
    except cp.SolverError:
        problem.solve(verbose=False)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"MVEE optimization failed: {problem.status}")

    m_opt = np.asarray(m_var.value, dtype=np.float64)
    g_opt = np.asarray(g_var.value, dtype=np.float64)
    center = np.linalg.solve(m_opt, g_opt)
    a_mat = m_opt.T @ m_opt
    return center, a_mat


def fit_mvee(points):
    try:
        from scipy.spatial import ConvexHull
    except ImportError as exc:
        raise ImportError("Perception mode requires scipy.") from exc

    if len(points) < 4:
        raise RuntimeError(f"Not enough points for 3D MVEE: {len(points)}")

    # QJ handles nearly coplanar visible surfaces more robustly.
    hull = ConvexHull(points, qhull_options="QJ")
    hull_points = points[hull.vertices]

    center, a_mat = mvee_cvxpy(hull_points)
    eigvals, eigvecs = np.linalg.eigh(a_mat)
    eigvals = np.clip(eigvals, 1e-15, None)
    axes = 1.0 / np.sqrt(eigvals)

    order = np.argsort(axes)[::-1]
    axes = axes[order]
    rotation = eigvecs[:, order]

    # Convert possible reflection to SO(3) without changing the ellipsoid.
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0

    return center, rotation, axes, hull_points


def build_perception_obstacle_ellipsoid(
    model,
    data,
    renderer,
    dino,
    args,
):
    rear_rgb, rear_depth = render_rgb_depth(renderer, data, "rear_cam")
    fixed_rgb, fixed_depth = render_rgb_depth(renderer, data, "fixed")

    if args.obstacle_text:
        obstacle_text = args.obstacle_text.strip()
        vlm_used = False
    else:
        obstacle_text = identify_obstacle_with_glm(rear_rgb, TASK_PROMPT)
        vlm_used = True

    print(f"  [VLSA perception] obstacle query: {obstacle_text!r}"
          + (" (VLM)" if vlm_used else " (fixed debug override)"))

    all_points = []
    detection_log = {}

    for camera_name, rgb, depth in (
        ("rear_cam", rear_rgb, rear_depth),
        ("fixed", fixed_rgb, fixed_depth),
    ):
        det = dino.detect_best_box(
            rgb,
            obstacle_text,
            args.box_threshold,
            args.text_threshold,
        )
        detection_log[camera_name] = det

        if det is None:
            print(f"  [GroundingDINO] {camera_name}: no detection")
            continue

        pts = bbox_depth_to_world_points(
            model,
            data,
            camera_name,
            depth,
            det["xyxy"],
        )
        print(
            f"  [GroundingDINO] {camera_name}: "
            f"bbox={det['xyxy']} conf={det['confidence']:.3f} "
            f"phrase={det['phrase']!r} points={len(pts)}"
        )
        if len(pts) > 0:
            all_points.append(pts)

    if not all_points:
        return None, {
            "obstacle_text": obstacle_text,
            "raw_points": np.empty((0, 3)),
            "filtered_points": np.empty((0, 3)),
            "detections": detection_log,
            "reason": "no_grounded_points",
        }

    raw_points = np.vstack(all_points)
    filtered_points = filter_obstacle_points(raw_points, args)

    print(
        f"  [PointCloud] raw={len(raw_points)} "
        f"filtered={len(filtered_points)}"
    )

    if len(filtered_points) < 4:
        return None, {
            "obstacle_text": obstacle_text,
            "raw_points": raw_points,
            "filtered_points": filtered_points,
            "detections": detection_log,
            "reason": "insufficient_filtered_points",
        }

    try:
        center, rotation, axes, hull_points = fit_mvee(filtered_points)
    except Exception as exc:
        return None, {
            "obstacle_text": obstacle_text,
            "raw_points": raw_points,
            "filtered_points": filtered_points,
            "detections": detection_log,
            "reason": f"mvee_failed:{type(exc).__name__}:{exc}",
        }

    print(f"  [MVEE] center={np.array2string(center, precision=4)}")
    print(f"  [MVEE] axes  ={np.array2string(axes, precision=4)}")

    return (center, rotation, axes), {
        "obstacle_text": obstacle_text,
        "raw_points": raw_points,
        "filtered_points": filtered_points,
        "hull_points": hull_points,
        "detections": detection_log,
        "reason": "ok",
    }


# =============================================================================
# Joint-action adapter
# =============================================================================
def adapt_joint_target_through_vlsa(
    model,
    data,
    current_action,
    safety_layer,
    link7_id,
    action_dt,
):
    q = data.qpos[:6].copy()
    q_delta_nom = np.clip(
        current_action[:6] - q,
        -MAX_JOINT_DELTA,
        MAX_JOINT_DELTA,
    )
    qdot_nom = q_delta_nom / action_dt

    eef_center, eef_rotation = get_eef_ellipsoid_pose(data, link7_id)
    jacp_full, jacr_full = get_point_jacobians(
        model,
        data,
        eef_center,
        link7_id,
    )
    j_pos = jacp_full[:, :6]
    j_rot = jacr_full[:, :6]
    j_twist = np.vstack([j_pos, j_rot])

    v_nom = j_pos @ qdot_nom
    omega_nom = j_rot @ qdot_nom

    v_safe, omega_safe, debug = safety_layer.filter(
        eef_center,
        eef_rotation,
        v_nom,
        omega_nom,
    )

    delta_twist = np.hstack(
        [v_safe - v_nom, omega_safe - omega_nom]
    )
    delta_qdot = damped_pinv(j_twist, DLS_RHO) @ delta_twist

    q_delta_safe = (qdot_nom + delta_qdot) * action_dt
    q_delta_safe = np.clip(
        q_delta_safe,
        -MAX_JOINT_DELTA,
        MAX_JOINT_DELTA,
    )

    ctrl = np.zeros(8, dtype=np.float64)
    ctrl[:6] = q + q_delta_safe
    gripper_val = 0.04 if current_action[6] > 0.02 else 0.0
    ctrl[6:8] = gripper_val

    debug.update(
        {
            "eef_center": eef_center.copy(),
            "v_nom": v_nom.copy(),
            "v_safe": v_safe.copy(),
            "omega_nom": omega_nom.copy(),
            "omega_safe": omega_safe.copy(),
            "q_delta_nom": q_delta_nom.copy(),
            "q_delta_safe": q_delta_safe.copy(),
        }
    )
    return ctrl, debug


def build_baseline_ctrl(data, current_action):
    q = data.qpos[:6].copy()
    q_delta = np.clip(
        current_action[:6] - q,
        -MAX_JOINT_DELTA,
        MAX_JOINT_DELTA,
    )
    ctrl = np.zeros(8, dtype=np.float64)
    ctrl[:6] = q + q_delta
    ctrl[6:8] = 0.04 if current_action[6] > 0.02 else 0.0
    return ctrl


# =============================================================================
# Existing paper metrics
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

    target_jnt_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "fj_screwdriver",
    )
    if target_jnt_id != -1:
        adr = model.jnt_qposadr[target_jnt_id]
        data.qpos[adr : adr + 3] = [target_x, target_y, 0.22]
        data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]

    pillar_jnt_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "pillar_joint",
    )
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
    filename = os.path.join(
        folder,
        f"ep_{episode_idx:03d}_{time.strftime('%H%M%S')}.mp4",
    )
    height, width, _ = frames[0].shape
    writer = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (width, height),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def resolve_groundingdino_paths(args):
    def first_existing(candidates):
        for path in candidates:
            if path and os.path.isfile(path):
                return os.path.abspath(path)
        return None

    config = first_existing(
        [
            args.groundingdino_config,
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "GroundingDINO/GroundingDINO_SwinT_OGC.py",
        ]
    )
    checkpoint = first_existing(
        [
            args.groundingdino_checkpoint,
            "GroundingDINO/groundingdino_swint_ogc.pth",
        ]
    )

    if config is None:
        raise FileNotFoundError(
            "GroundingDINO config not found. Pass --groundingdino_config PATH."
        )
    if checkpoint is None:
        raise FileNotFoundError(
            "GroundingDINO checkpoint not found. Pass --groundingdino_checkpoint PATH."
        )
    return config, checkpoint



def _vec3_to_text(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if len(v) < 3:
        return "[nan nan nan]"
    return f"[{v[0]: .4f} {v[1]: .4f} {v[2]: .4f}]"


def save_and_print_trace(trace_records, save_folder, episode_idx, window, manual_step=None):
    """
    Save/print a compact window around the strongest VLSA intervention.

    Each record is one policy step. The nominal/safe twists and h correspond to
    the action computed at the beginning of that policy step; the positions and
    box distances are sampled immediately after its 50 MuJoCo servo substeps.
    """
    if not trace_records or window <= 0:
        return None

    valid = [r for r in trace_records if np.isfinite(r["intervention"])]
    if not valid:
        return None

    if manual_step is None:
        center_record = max(valid, key=lambda r: r["intervention"])
    else:
        center_record = min(valid, key=lambda r: abs(r["step"] - manual_step))

    center_step = int(center_record["step"])
    lo = center_step - int(window)
    hi = center_step + int(window)
    selected = [r for r in trace_records if lo <= r["step"] <= hi]

    csv_path = os.path.join(
        save_folder,
        f"trace_ep_{episode_idx:03d}_center_{center_step:04d}.csv",
    )
    fieldnames = [
        "step",
        "h",
        "intervention",
        "cbf_applied",
        "shadow_only",
        "gripper_cmd",
        "screw_z",
        "tcp_box_dist",
        "screw_box_dist",
        "eef_x", "eef_y", "eef_z",
        "screw_x", "screw_y", "screw_z_pos",
        "v_nom_x", "v_nom_y", "v_nom_z",
        "v_safe_x", "v_safe_y", "v_safe_z",
        "dv_x", "dv_y", "dv_z",
        "omega_nom_x", "omega_nom_y", "omega_nom_z",
        "omega_safe_x", "omega_safe_y", "omega_safe_z",
        "domega_x", "domega_y", "domega_z",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in selected:
            row = {
                "step": r["step"],
                "h": r["h"],
                "intervention": r["intervention"],
                "cbf_applied": int(r["cbf_applied"]),
                "shadow_only": int(r["shadow_only"]),
                "gripper_cmd": r["gripper_cmd"],
                "screw_z": r["screw_z"],
                "tcp_box_dist": r["tcp_box_dist"],
                "screw_box_dist": r["screw_box_dist"],
            }
            for prefix, vec in (
                ("eef", r["eef_pos"]),
                ("screw", r["screw_pos"]),
                ("v_nom", r["v_nom"]),
                ("v_safe", r["v_safe"]),
                ("dv", r["dv"]),
                ("omega_nom", r["omega_nom"]),
                ("omega_safe", r["omega_safe"]),
                ("domega", r["domega"]),
            ):
                suffixes = ("x", "y", "z")
                if prefix == "screw":
                    keys = ("screw_x", "screw_y", "screw_z_pos")
                else:
                    keys = tuple(f"{prefix}_{s}" for s in suffixes)
                for key, value in zip(keys, np.asarray(vec).reshape(-1)[:3]):
                    row[key] = float(value)
            writer.writerow(row)

    txt_path = os.path.join(
        save_folder,
        f"trace_ep_{episode_idx:03d}_center_{center_step:04d}.txt",
    )

    lines = []
    lines.append(
        f"Peak-intervention trace: episode={episode_idx}, "
        f"center_step={center_step}, window=±{window}"
    )
    lines.append(
        "Columns: step | h | intv | tcp->box | screw->box | "
        "v_nom -> v_safe | dv | omega_nom -> omega_safe | domega"
    )

    for r in selected:
        mark = " <<<" if r["step"] == center_step else ""
        lines.append(
            f"{r['step']:4d} | "
            f"h={r['h']: .5f} | "
            f"intv={r['intervention']: .5f} | "
            f"tcp_box={r['tcp_box_dist']: .4f} | "
            f"screw_box={r['screw_box_dist']: .4f} | "
            f"v {_vec3_to_text(r['v_nom'])} -> {_vec3_to_text(r['v_safe'])} | "
            f"dv={_vec3_to_text(r['dv'])} | "
            f"w {_vec3_to_text(r['omega_nom'])} -> {_vec3_to_text(r['omega_safe'])} | "
            f"dw={_vec3_to_text(r['domega'])}"
            f"{mark}"
        )

    trace_text = "\n".join(lines)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(trace_text + "\n")

    print("\n" + "-" * 110)
    print(trace_text)
    print("-" * 110)
    print(f"Trace CSV saved to: {csv_path}")
    print(f"Trace TXT saved to: {txt_path}\n")

    return {
        "center_step": center_step,
        "csv_path": csv_path,
        "txt_path": txt_path,
    }


# =============================================================================
# Main
# =============================================================================
def main(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = args.xml_path or os.path.join(base_dir, "dummyx_apf_scene.xml")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=256, width=256)
    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    action_dt = SERVO_SUBSTEPS * model.opt.timestep

    target_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "real_screwdriver"
    )
    pillar_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_pillar"
    )
    tcp_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site"
    )
    link7_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "link7"
    )
    link8_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "link8"
    )
    link9_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "link9"
    )

    if min(target_body_id, pillar_body_id, tcp_id, link7_id) < 0:
        raise RuntimeError("Required MuJoCo body/site names are missing.")

    valid_collision_bodies = [target_body_id, link7_id]
    if link8_id != -1:
        valid_collision_bodies.append(link8_id)
    if link9_id != -1:
        valid_collision_bodies.append(link9_id)

    dino = None
    if (
        args.mode in ("vlsa", "shadow")
        and args.obstacle_source == "perception"
        and not args.no_obstacle
    ):
        config_path, checkpoint_path = resolve_groundingdino_paths(args)
        print(f"Loading GroundingDINO config: {config_path}")
        print(f"Loading GroundingDINO checkpoint: {checkpoint_path}")
        dino = GroundingDINOWrapper(
            config_path,
            checkpoint_path,
            args.device,
        )

    fixed_targets = None
    if args.fixed_eval:
        fixed_xy = (0.28, -0.10)
        fixed_targets = [fixed_xy for _ in range(args.num_episodes)]
        print(
            f"Fixed evaluation target: "
            f"X={fixed_xy[0]:.2f}, Y={fixed_xy[1]:.2f}"
        )

    use_obstacle = not args.no_obstacle
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"run_{timestamp}_{args.mode}_"
        f"{args.obstacle_source}_obs{int(use_obstacle)}"
    )
    record_root = os.path.join(base_dir, "recordings", run_name)
    os.makedirs(record_root, exist_ok=True)

    success_count = 0
    collision_count = 0
    perception_failures = 0
    episode_peak_forces = []
    succ_peak_forces = []
    all_impulses = []
    succ_impulses = []
    episode_peak_torques = []
    all_qp_times = []
    all_h_values = []
    qp_failures = 0

    print("\n" + "=" * 90)
    print("VLSA/AEGIS V2.3 — perception MVEE + full-action CBF-QP + peak-intervention trace")
    print(
        f"mode={args.mode} | obstacle={use_obstacle} | "
        f"obstacle_source={args.obstacle_source} | "
        f"episodes={args.num_episodes}"
    )
    print(f"EEF ellipsoid semi-axes: {Q_EEF_DIAG} m")
    print(f"EEF center in link7:     {EEF_CENTER_IN_LINK7} m")
    if args.show_ellipsoids:
        print(
            "Live debug ellipsoids ON: "
            "blue=EEF, red=control obstacle, yellow=oracle reference"
        )
        if args.obstacle_source == "oracle":
            print(
                "Oracle red ellipsoid is LIVE: it follows dynamic_pillar xpos/xmat "
                "every policy step."
            )
    if args.obstacle_text:
        print(
            "DEBUG: --obstacle_text bypasses the VLM; "
            "do not use that override in the final VLSA result."
        )
    print("=" * 90 + "\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for episode in range(args.num_episodes):
            if not viewer.is_running():
                break

            target_xy = (
                fixed_targets[episode]
                if fixed_targets is not None
                else None
            )
            reset_scene(
                model,
                data,
                use_obstacle=use_obstacle,
                target_xy=target_xy,
            )
            viewer.sync()

            safety_layer = None
            perception_meta = {
                "reason": "not_used",
                "raw_points": np.empty((0, 3)),
                "filtered_points": np.empty((0, 3)),
                "obstacle_text": "",
            }

            p_obs = np.full(3, np.nan)
            r_obs = np.full((3, 3), np.nan)
            q_obs = np.full(3, np.nan)

            if args.mode in ("vlsa", "shadow") and use_obstacle:
                if args.obstacle_source == "oracle":
                    p_obs, r_obs, q_obs = build_oracle_obstacle_ellipsoid(
                        data,
                        pillar_body_id,
                    )
                    perception_meta["reason"] = "oracle_debug"
                else:
                    result, perception_meta = build_perception_obstacle_ellipsoid(
                        model,
                        data,
                        renderer,
                        dino,
                        args,
                    )
                    if result is not None:
                        p_obs, r_obs, q_obs = result
                    else:
                        perception_failures += 1
                        print(
                            f"  [VLSA perception] FAILED: "
                            f"{perception_meta['reason']}. "
                            "Safety layer disabled for this episode."
                        )

                if np.isfinite(p_obs).all():
                    eef_center0, _ = get_eef_ellipsoid_pose(
                        data,
                        link7_id,
                    )
                    safety_layer = AEGISFullActionLayer(
                        p_obs,
                        r_obs,
                        q_obs,
                        action_dt,
                    )
                    safety_layer.reset(eef_center0)

                    p_oracle, _, _ = build_oracle_obstacle_ellipsoid(
                        data,
                        pillar_body_id,
                    )
                    print(
                        f"  [Geometry check] perceived/oracle center error = "
                        f"{np.linalg.norm(p_obs - p_oracle):.4f} m"
                    )

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
            ep_omega_nom = []
            ep_omega_safe = []
            ep_z = []
            ep_screw_z = []
            ep_tcp_box_dist = []
            ep_screw_box_dist = []
            ep_gripper_cmd = []
            ep_eef_pos = []
            ep_screw_pos = []
            ep_cbf_applied = []
            ep_shadow_only = []
            ep_intervention_step = []
            ep_h_step = []
            ep_q_pos_vla = []
            ep_q_dot_vla = []
            trace_records = []

            record_cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, record_cam)
            record_cam.lookat[:] = [0.25, -0.05, 0.22]
            record_cam.distance = 1.0
            record_cam.azimuth = 180
            record_cam.elevation = -20

            box_center = np.array([-0.05, 0.0, 0.24])

            while viewer.is_running() and step_counter < args.max_steps:
                if step_counter % 8 == 0 or action_chunk_cache is None:
                    renderer.disable_depth_rendering()
                    renderer.update_scene(data, camera="rear_cam")
                    img_external = renderer.render()
                    renderer.update_scene(data, camera="wrist_cam")
                    img_wrist = renderer.render()

                    result = policy.infer(
                        {
                            "observation/image": img_external,
                            "observation/wrist_image": img_wrist,
                            "observation/state": data.qpos[:8].copy(),
                            "prompt": TASK_PROMPT,
                        }
                    )
                    action_chunk_cache = result["actions"]

                renderer.disable_depth_rendering()
                renderer.update_scene(data, camera=record_cam)
                video_frames.append(renderer.render())

                current_action = action_chunk_cache[step_counter % 8]
                raw_q_pos_vla = current_action[:6].copy()
                raw_q_dot_vla = raw_q_pos_vla - data.qpos[:6].copy()
                ep_q_pos_vla.append(raw_q_pos_vla)
                ep_q_dot_vla.append(raw_q_dot_vla)
                ep_gripper_cmd.append(float(current_action[6]))

                debug = None
                if safety_layer is not None:
                    # In oracle-debug mode the pillar is a free body. Keep the
                    # controlled obstacle ellipsoid attached to the current
                    # MuJoCo body pose. Perception mode does not read ground truth.
                    if args.obstacle_source == "oracle":
                        p_live, r_live, q_live = build_oracle_obstacle_ellipsoid(
                            data,
                            pillar_body_id,
                        )
                        safety_layer.update_obstacle_pose(
                            p_live,
                            r_live,
                            q_live,
                        )

                    safe_ctrl_candidate, debug = adapt_joint_target_through_vlsa(
                        model,
                        data,
                        current_action,
                        safety_layer,
                        link7_id,
                        action_dt,
                    )

                    # Shadow mode is the key diagnosis:
                    # compute the exact VLSA CBF-QP that WOULD be applied, evolve
                    # its auxiliary z state, and log the intervention, but execute
                    # the untouched pi0/baseline joint target.
                    if args.mode == "shadow":
                        base_ctrl = build_baseline_ctrl(data, current_action)
                        ep_shadow_only.append(True)
                        ep_cbf_applied.append(False)
                    else:
                        base_ctrl = safe_ctrl_candidate
                        ep_shadow_only.append(False)
                        ep_cbf_applied.append(True)

                    ep_qp_times.append(debug["qp_ms"])
                    ep_h.append(debug["h"])
                    ep_interventions.append(debug["intervention"])
                    ep_intervention_step.append(step_counter)
                    ep_h_step.append(step_counter)
                    ep_qp_ok.append(debug["qp_ok"])
                    ep_v_nom.append(debug["v_nom"])
                    ep_v_safe.append(debug["v_safe"])
                    ep_omega_nom.append(debug["omega_nom"])
                    ep_omega_safe.append(debug["omega_safe"])
                    ep_z.append(debug["z"])
                    if not debug["qp_ok"]:
                        qp_failures += 1
                else:
                    base_ctrl = build_baseline_ctrl(
                        data,
                        current_action,
                    )
                    ep_cbf_applied.append(False)
                    ep_shadow_only.append(False)

                for _ in range(SERVO_SUBSTEPS):
                    current_f_xyz = get_target_table_force(
                        model,
                        data,
                        valid_collision_bodies,
                    )
                    force_norm = np.linalg.norm(current_f_xyz)
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
                screw_pos = data.xpos[target_body_id].copy()
                eef_center_now, _ = get_eef_ellipsoid_pose(
                    data,
                    link7_id,
                )

                ep_screw_z.append(float(screw_pos[2]))
                ep_tcp_box_dist.append(
                    float(np.linalg.norm(current_tcp - box_center))
                )
                ep_screw_box_dist.append(
                    float(np.linalg.norm(screw_pos - box_center))
                )
                ep_eef_pos.append(eef_center_now)
                ep_screw_pos.append(screw_pos)

                if debug is not None:
                    v_nom_trace = np.asarray(debug["v_nom"], dtype=np.float64)
                    v_safe_trace = np.asarray(debug["v_safe"], dtype=np.float64)
                    w_nom_trace = np.asarray(debug["omega_nom"], dtype=np.float64)
                    w_safe_trace = np.asarray(debug["omega_safe"], dtype=np.float64)
                    trace_records.append(
                        {
                            "step": int(step_counter),
                            "h": float(debug["h"]),
                            "intervention": float(debug["intervention"]),
                            "cbf_applied": bool(args.mode == "vlsa"),
                            "shadow_only": bool(args.mode == "shadow"),
                            "gripper_cmd": float(current_action[6]),
                            "screw_z": float(screw_pos[2]),
                            "tcp_box_dist": float(
                                np.linalg.norm(current_tcp - box_center)
                            ),
                            "screw_box_dist": float(
                                np.linalg.norm(screw_pos - box_center)
                            ),
                            "eef_pos": eef_center_now.copy(),
                            "screw_pos": screw_pos.copy(),
                            "v_nom": v_nom_trace.copy(),
                            "v_safe": v_safe_trace.copy(),
                            "dv": (v_safe_trace - v_nom_trace).copy(),
                            "omega_nom": w_nom_trace.copy(),
                            "omega_safe": w_safe_trace.copy(),
                            "domega": (w_safe_trace - w_nom_trace).copy(),
                        }
                    )

                in_box = (
                    abs(screw_pos[0] - (-0.05)) < 0.12
                    and abs(screw_pos[1]) < 0.12
                    and screw_pos[2] < 0.34
                )
                is_released = (
                    current_action[6] > 0.02
                    or np.linalg.norm(current_tcp - screw_pos) > 0.06
                )
                in_box_counter = (
                    in_box_counter + 1
                    if in_box and is_released
                    else 0
                )

                update_debug_ellipsoids(
                    viewer,
                    data,
                    link7_id,
                    safety_layer,
                    args.show_ellipsoids,
                    args.show_oracle_reference,
                    pillar_body_id,
                )
                viewer.sync()

                if in_box_counter >= 5:
                    is_success = True
                    break

                step_counter += 1

            if is_success:
                success_count += 1
            if episode_collision:
                collision_count += 1

            ep_peak_tau = (
                float(np.max(ep_torques))
                if ep_torques
                else 0.0
            )
            episode_peak_torques.append(ep_peak_tau)
            episode_peak_forces.append(ep_peak_force)
            all_impulses.append(ep_impulse)

            if is_success:
                succ_peak_forces.append(ep_peak_force)
                succ_impulses.append(ep_impulse)

            if ep_qp_times:
                all_qp_times.extend(ep_qp_times)
                all_h_values.extend(ep_h)

            save_folder = os.path.join(
                record_root,
                "success" if is_success else "fail",
            )
            os.makedirs(save_folder, exist_ok=True)
            save_episode_video(
                video_frames,
                save_folder,
                episode + 1,
            )

            np.savez(
                os.path.join(
                    save_folder,
                    f"data_ep_{episode + 1:03d}.npz",
                ),
                episode_success=np.array(is_success),
                episode_knockdown=np.array(episode_collision),
                episode_peak_contact_force=np.array(ep_peak_force),
                episode_contact_impulse=np.array(ep_impulse),
                episode_peak_joint_torque=np.array(ep_peak_tau),
                q_pos_vla=np.asarray(ep_q_pos_vla),
                q_dot_vla=np.asarray(ep_q_dot_vla),
                v_nom=np.asarray(ep_v_nom),
                v_safe=np.asarray(ep_v_safe),
                omega_nom=np.asarray(ep_omega_nom),
                omega_safe=np.asarray(ep_omega_safe),
                cbf_h=np.asarray(ep_h),
                cbf_applied=np.asarray(ep_cbf_applied, dtype=bool),
                shadow_only=np.asarray(ep_shadow_only, dtype=bool),
                intervention_step=np.asarray(ep_intervention_step, dtype=int),
                h_step=np.asarray(ep_h_step, dtype=int),
                screw_z=np.asarray(ep_screw_z),
                tcp_box_dist=np.asarray(ep_tcp_box_dist),
                screw_box_dist=np.asarray(ep_screw_box_dist),
                gripper_cmd=np.asarray(ep_gripper_cmd),
                eef_pos=np.asarray(ep_eef_pos),
                screw_pos=np.asarray(ep_screw_pos),
                qp_time_ms=np.asarray(ep_qp_times),
                qp_ok=np.asarray(ep_qp_ok, dtype=bool),
                intervention_norm=np.asarray(ep_interventions),
                z_state=np.asarray(ep_z),
                eef_axes=Q_EEF_DIAG,
                eef_center_in_link7=EEF_CENTER_IN_LINK7,
                obstacle_center=np.asarray(p_obs),
                obstacle_rotation=np.asarray(r_obs),
                obstacle_axes=np.asarray(q_obs),
                obstacle_source=np.array(args.obstacle_source),
                obstacle_text=np.array(
                    perception_meta.get("obstacle_text", "")
                ),
                perception_status=np.array(
                    perception_meta.get("reason", "")
                ),
                obstacle_raw_points=np.asarray(
                    perception_meta.get(
                        "raw_points",
                        np.empty((0, 3)),
                    )
                ),
                obstacle_filtered_points=np.asarray(
                    perception_meta.get(
                        "filtered_points",
                        np.empty((0, 3)),
                    )
                ),
            )

            if args.mode in ("vlsa", "shadow") and args.trace_window > 0:
                save_and_print_trace(
                    trace_records,
                    save_folder,
                    episode + 1,
                    args.trace_window,
                    args.trace_step,
                )

            max_screw_z = (
                float(np.max(ep_screw_z))
                if ep_screw_z
                else float(data.xpos[target_body_id][2])
            )
            picked_up = max_screw_z > 0.235
            qp_text = (
                f" | picked={'YES' if picked_up else 'NO'}"
                f" | z_obj_max={max_screw_z:.3f}"
            )

            if ep_qp_times:
                h_arr = np.asarray(ep_h)
                int_arr = np.asarray(ep_interventions)
                peak_idx = int(np.argmax(int_arr))
                peak_step = (
                    ep_intervention_step[peak_idx]
                    if peak_idx < len(ep_intervention_step)
                    else -1
                )
                qp_text += (
                    f" | h0={h_arr[0]:.4f}"
                    f" | h_min={np.min(h_arr):.4f}"
                    f" | h<0={100.0*np.mean(h_arr < 0):.1f}%"
                    f" | QP={np.mean(ep_qp_times):.3f} ms"
                    f" | intervention_mean="
                    f"{np.mean(int_arr):.4f}"
                    f" | intervention_max="
                    f"{np.max(int_arr):.4f}@step{peak_step}"
                )
                if args.mode == "shadow":
                    qp_text += " | EXECUTED=BASELINE"
            else:
                qp_text += (
                    f" | safety=OFF"
                    f" | perception="
                    f"{perception_meta.get('reason', 'not_used')}"
                )

            print(
                f"Episode {episode + 1}/{args.num_episodes}: "
                f"{'SUCCESS' if is_success else 'FAIL'} | "
                f"{'COLLISION' if episode_collision else 'NO COLLISION'} | "
                f"peakF={ep_peak_force:.2f} N | "
                f"impulse={ep_impulse:.2f} N*s | "
                f"peakTau={ep_peak_tau:.2f} N*m"
                f"{qp_text}"
            )

    episodes_done = max(len(episode_peak_forces), 1)
    sr = 100.0 * success_count / episodes_done
    cr = (
        100.0 * collision_count / episodes_done
        if use_obstacle
        else None
    )
    avg_peak_f_succ = (
        float(np.mean(succ_peak_forces))
        if succ_peak_forces
        else None
    )
    avg_imp_all = (
        float(np.mean(all_impulses))
        if all_impulses
        else 0.0
    )
    avg_imp_succ = (
        float(np.mean(succ_impulses))
        if succ_impulses
        else None
    )
    std_imp_succ = (
        float(np.std(succ_impulses))
        if succ_impulses
        else None
    )
    avg_tau = (
        float(np.mean(episode_peak_torques))
        if episode_peak_torques
        else 0.0
    )
    avg_qp_ms = (
        float(np.mean(all_qp_times))
        if all_qp_times
        else None
    )
    min_h = (
        float(np.min(all_h_values))
        if all_h_values
        else None
    )

    lines = [
        "================ VLSA/AEGIS V2 SUMMARY ================",
        f"mode: {args.mode}",
        "shadow_note: QP computed but not executed" if args.mode == "shadow" else "shadow_note: —",
        f"obstacle: {use_obstacle}",
        f"obstacle_source: {args.obstacle_source}",
        f"episodes: {episodes_done}",
        f"perception_failures: {perception_failures}",
        f"SR: {sr:.1f}%",
        f"CR: {cr:.1f}%" if cr is not None else "CR: —",
        (
            f"f_hat_success: {avg_peak_f_succ:.2f} N"
            if avg_peak_f_succ is not None
            else "f_hat_success: —"
        ),
        f"I_all: {avg_imp_all:.2f} N*s",
        (
            f"I_success: {avg_imp_succ:.2f} N*s"
            if avg_imp_succ is not None
            else "I_success: —"
        ),
        (
            f"sigma_I_success: {std_imp_succ:.2f} N*s"
            if std_imp_succ is not None
            else "sigma_I_success: —"
        ),
        f"tau_bar_max: {avg_tau:.2f} N*m",
        (
            f"QP_mean: {avg_qp_ms:.3f} ms"
            if avg_qp_ms is not None
            else "QP_mean: —"
        ),
        (
            f"h_min: {min_h:.6f}"
            if min_h is not None
            else "h_min: —"
        ),
        f"QP_failures: {qp_failures}",
        "=======================================================",
    ]
    summary = "\n".join(lines)
    print("\n" + summary + "\n")

    summary_path = os.path.join(record_root, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "VLSA/AEGIS V2.3: perception MVEE + full-action CBF-QP + peak-intervention trace "
            "for screwdriver pick-and-place"
        )
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="OpenPI WebSocket server IP",
    )
    parser.add_argument(
        "--port",
        default=8000,
        type=int,
        help="OpenPI WebSocket server port",
    )
    parser.add_argument(
        "--num_episodes",
        default=10,
        type=int,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max_steps",
        default=600,
        type=int,
        help="Maximum policy steps per episode",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "vlsa", "shadow"],
        default="vlsa",
        help=(
            "baseline: execute original pi0; "
            "vlsa: execute AEGIS safety correction; "
            "shadow: compute/log AEGIS QP but execute original pi0 unchanged."
        ),
    )
    parser.add_argument(
        "--obstacle_source",
        choices=["oracle", "perception"],
        default="perception",
        help=(
            "perception: VLSA VLM+DINO+RGB-D+MVEE; "
            "oracle: debug-only MuJoCo geometry"
        ),
    )
    parser.add_argument(
        "--fixed_eval",
        action="store_true",
        help="Use fixed screwdriver XY=(0.28,-0.10)",
    )
    parser.add_argument(
        "--no_obstacle",
        action="store_true",
        help="Remove dynamic pillar",
    )
    parser.add_argument(
        "--xml_path",
        default=None,
        help="Optional explicit path to dummyx_apf_scene.xml",
    )

    # Live visualization.
    parser.add_argument(
        "--show_ellipsoids",
        action="store_true",
        help="Show live EEF and obstacle ellipsoids in the MuJoCo viewer.",
    )
    parser.add_argument(
        "--show_oracle_reference",
        action="store_true",
        help=(
            "With --show_ellipsoids, also show the oracle pillar ellipsoid "
            "in yellow for geometry debugging. It is not used for control."
        ),
    )

    # VLM / GroundingDINO.
    parser.add_argument(
        "--obstacle_text",
        default=None,
        help=(
            "DEBUG ONLY: fixed GroundingDINO query such as 'gray pillar'. "
            "If omitted, GLM-4.5V identifies the obstacle."
        ),
    )
    parser.add_argument(
        "--groundingdino_config",
        default=None,
        help="Path to GroundingDINO config.",
    )
    parser.add_argument(
        "--groundingdino_checkpoint",
        default=None,
        help="Path to groundingdino_swint_ogc.pth.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="GroundingDINO device.",
    )
    parser.add_argument(
        "--box_threshold",
        default=0.35,
        type=float,
    )
    parser.add_argument(
        "--text_threshold",
        default=0.25,
        type=float,
    )

    # User-scene point-cloud preprocessing.
    parser.add_argument(
        "--workspace",
        nargs=6,
        type=float,
        default=[-0.20, 0.55, -0.25, 0.35, 0.22, 0.60],
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help=(
            "Workspace crop in world coordinates. Defaults are adapted only "
            "to this uploaded MuJoCo scene."
        ),
    )
    parser.add_argument(
        "--dbscan_eps",
        default=0.01,
        type=float,
        help=(
            "DBSCAN radius in meters. Scene/resolution-specific adaptation; "
            "the VLSA method itself is unchanged."
        ),
    )
    parser.add_argument(
        "--dbscan_min_samples",
        default=20,
        type=int,
    )

    # Peak-intervention trace.
    parser.add_argument(
        "--trace_window",
        default=15,
        type=int,
        help=(
            "For vlsa/shadow modes, print and save +/- N policy steps around "
            "the strongest intervention. Set 0 to disable. Default: 15."
        ),
    )
    parser.add_argument(
        "--trace_step",
        default=None,
        type=int,
        help=(
            "Optional manual trace center step. If omitted, the strongest "
            "intervention step is selected automatically."
        ),
    )

    main(parser.parse_args())
