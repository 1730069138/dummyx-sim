"""Independent, physical-unit AEGIS 9-D QP adapter for the external benchmark.

Source: THU-RCSCT/vlsa-aegis@ce5683c069983baef4e66025aef9da30d9d67ca2.
Only the current frozen policy target is read. No PACE controller, force feedback,
task attraction, policy inference, future target, or physics integration is used.
The official LIBERO scaled/body action is converted algebraically to world twist.
This is an action-layer adaptation, not the original VLM/DINO/MVEE pipeline.
"""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
import time
import numpy as np


OFFICIAL_COMMIT = "ce5683c069983baef4e66025aef9da30d9d67ca2"
ALGORITHM = "vlsa_aegis_independent_physical_world_9d_v1"
DEPLOY = Path(__file__).resolve().parents[2]
SOURCE_HASHES = {
    "agv_vlsa.py": "eb13a931585f5b5b485527839372dffae928a38ba4883436d81155aaaeb75237",
    "deploy_screwdriver_client_vlsa.py": "db98f578432fe4efc05965cbb92813e9c4bb43d38b4c3615d5a2fb708a0feaef",
}


def build_gripper_enclosure(arm, root_name="link9"):
    """Enclose all collidable gripper meshes over their complete slide ranges.

    A mesh vertex below only prismatic joints is affine in the joint positions
    in the fixed root frame. Every legal vertex therefore lies in the convex
    hull of joint-range corner vertices. Their AABB is enclosed by the ellipsoid
    with the same center and sqrt(3) times its half-extents. Reject other joint
    types rather than extrapolating a finite sampling result.
    """
    arm_mj, model = arm.mj, arm.model
    root = model.body(root_name).id
    data = arm_mj.MjData(model)
    data.qpos[:] = arm.data.qpos
    geoms, joints, chains = [], set(), {}
    for geom in range(model.ngeom):
        body, chain = int(model.geom_bodyid[geom]), []
        while body and body != root:
            chain += list(range(model.body_jntadr[body], model.body_jntadr[body]+model.body_jntnum[body]))
            body = int(model.body_parentid[body])
        if body != root or not model.geom_contype[geom]:
            continue
        if model.geom_type[geom] != arm_mj.mjtGeom.mjGEOM_MESH:
            raise ValueError("Gripper enclosure requires collision mesh geometry")
        for joint in chain:
            if model.jnt_type[joint] != arm_mj.mjtJoint.mjJNT_SLIDE or not model.jnt_limited[joint]:
                raise ValueError("Continuous gripper enclosure proof supports bounded prismatic joints only")
            if not np.isfinite(model.jnt_range[joint]).all():
                raise ValueError("Nonfinite gripper slide range")
        geoms.append(geom)
        joints.update(chain)
        chains[model.geom(geom).name] = [model.joint(joint).name for joint in chain]
    if not geoms or len(joints) > 8:
        raise ValueError("Missing gripper collision meshes or unsupported corner count")
    joints = sorted(joints)
    vertices = []
    corners = list(itertools.product((0, 1), repeat=len(joints)))
    for corner in corners:
        for joint, side in zip(joints, corner):
            data.qpos[model.jnt_qposadr[joint]] = model.jnt_range[joint, side]
        arm_mj.mj_kinematics(model, data)
        origin, rotation = data.xpos[root], data.xmat[root].reshape(3, 3)
        for geom in geoms:
            mesh = int(model.geom_dataid[geom])
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            world = model.mesh_vert[start:start+count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            vertices.append((world-origin) @ rotation)
    points = np.concatenate(vertices)
    lower, upper = points.min(axis=0)-1e-10, points.max(axis=0)+1e-10
    center, axes = (lower+upper)/2., np.sqrt(3.)*(upper-lower)/2.
    if np.any(axes <= 0) or not np.isfinite(axes).all():
        raise ValueError("Degenerate gripper envelope")
    score = np.sum(((points-center)/axes)**2, axis=1)
    if score.max() > 1.+1e-10:
        raise RuntimeError("Gripper enclosure verification failed")
    manifest = dict(root_body=root_name, root_body_id=int(root),
                    method="full prismatic-range corner convex hull -> AABB -> sqrt(3) ellipsoid",
                    continuous_range_basis="every collision vertex is affine in bounded slide positions; ellipsoid is convex and encloses the AABB",
                    collision_geoms=[model.geom(g).name for g in geoms], collision_joint_chains=chains,
                    slide_ranges={model.joint(j).name:model.jnt_range[j].tolist() for j in joints},
                    independent_range_corners=len(corners), corner_vertex_count=len(points),
                    corner_max_quadratic=float(score.max()), aabb_lower=lower.tolist(), aabb_upper=upper.tolist(),
                    local_center=center.tolist(), axes=axes.tolist(),
                    corner_vertices_sha256=hashlib.sha256(points.astype('<f8').tobytes()).hexdigest(),
                    scope="all collidable meshes in link9 gripper subtree; excludes visual-only decorations, wrist/forearm, and carried tool")
    return root, center, axes, manifest


class AEGISPhysicalQP:
    """Official ellipsoid CBF in [v_world, omega_world, u_z] coordinates.

    Official u_body = 5 * action_body and constraint coefficient 0.2 imply
    x_world = 0.2 * blockdiag(R,R) * u_body. Its 1/25 cost becomes an identity
    cost on x_world. There are no correction bounds or measured-servo offsets.
    """
    def __init__(self, center, rotation, axes, dt, eef_axes, reference):
        self.p_obs = np.asarray(center, float).copy()
        self.r_obs = np.asarray(rotation, float).copy()
        self.q_obs = np.asarray(axes, float).copy()
        self.eef_axes = np.asarray(eef_axes, float).copy()
        self.dt = float(dt)
        if self.dt <= 0 or not np.isfinite(self.dt):
            raise ValueError("Invalid integration interval")
        if np.any(self.q_obs <= 0) or np.any(self.eef_axes <= 0):
            raise ValueError("Ellipsoid semiaxes must be positive")
        self.reference, self.z = reference, None

    def reset(self, eef_center):
        direction = self.p_obs - np.asarray(eef_center, float)
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            raise ValueError("Official auxiliary-state initialization is undefined at coincident centers")
        self.z = direction / norm

    def filter(self, eef_center, eef_rotation, v_nom, omega_nom):
        import cvxpy as cp
        if self.z is None:
            self.reset(eef_center)
        z_before = self.z.copy()
        av, aw, az, h, mu = self.reference.compute_cbf_coeffs_world_twist(
            eef_center, self.eef_axes, eef_rotation, self.p_obs,
            self.q_obs, self.r_obs, z_before)
        nominal = np.r_[v_nom, omega_nom]
        uz_nom = 10. * mu
        ref = np.r_[nominal, uz_nom]
        u = cp.Variable(9)
        a = np.r_[av, aw, az]
        problem = cp.Problem(cp.Minimize(cp.sum_squares(u - ref)),
                             [a @ u + 10. * h >= 0.])
        started = time.perf_counter_ns()
        error = None
        try:
            problem.solve(solver=cp.OSQP, warm_start=False, verbose=False)
            ok = u.value is not None and problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        except Exception as exc:
            ok, error = False, f"{type(exc).__name__}: {exc}"
        elapsed = (time.perf_counter_ns() - started) / 1e6
        solved = np.asarray(u.value, float).ravel() if ok else ref.copy()
        if not np.isfinite(solved).all():
            ok, solved, error = False, ref.copy(), "nonfinite_qp_solution"
        dz = self.reference.project_tangent(z_before) @ solved[6:]
        self.z = z_before + self.dt * dz
        self.z /= np.linalg.norm(self.z)
        return solved[:3], solved[3:6], {
            "h": float(h), "qp_ok": bool(ok), "qp_status": str(problem.status),
            "qp_error": error, "qp_ms": elapsed, "a_twist": np.r_[av, aw],
            "a_z": az, "u_z": solved[6:].copy(), "z_before": z_before,
            "z": self.z.copy(), "cbf_lhs_nom": float(a @ ref + 10. * h),
            "cbf_lhs_qp": float(a @ solved + 10. * h),
        }


class VLSAIndependent:
    def __init__(self, arm, obstacle, initial_tool_height=None, **kwargs):
        # The import supplies audited ellipsoid arithmetic, not its legacy QP.
        import deploy_screwdriver_client_vlsa as reference
        from agv_vlsa import capsule_ellipsoid
        for name, expected in SOURCE_HASHES.items():
            actual = hashlib.sha256((DEPLOY / name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Audited VLSA dependency changed: {name}")
        self.arm, self.reference = arm, reference
        self.obstacle = {k: np.asarray(v).copy() for k, v in obstacle.items()}
        self.dt = float(arm.steps * arm.model.opt.timestep)
        self.eef_body, self.eef_local_center, eef_axes, self.enclosure = build_gripper_enclosure(arm)
        center, rotation, axes = capsule_ellipsoid(obstacle)
        self.layer = AEGISPhysicalQP(center, rotation, axes, self.dt,
                                     eef_axes, reference)
        self.layer.reset(self.eef_pose()[0])
        self.current_guard = False
        self.execution_predictor = None
        self.refresh_pending = False
        self.rows, self.failures = [], []

    def eef_pose(self):
        rotation = self.arm.data.xmat[self.eef_body].reshape(3, 3).copy()
        center = self.arm.data.xpos[self.eef_body] + rotation @ self.eef_local_center
        return center.copy(), rotation

    def capsules(self, data, tool_local=None):
        return []

    def process(self, chunk, index, *, execution_stop=None, action_index=None, **kwargs):
        started = time.perf_counter_ns()
        chunk = np.asarray(chunk)
        if chunk.ndim != 2 or chunk.shape[1] != 7 or not 0 <= index < len(chunk):
            raise ValueError("Expected action chunk (N,7) and a valid current index")
        if execution_stop is not None and index >= execution_stop:
            raise ValueError("Current action is outside the scheduled execution prefix")
        raw = np.asarray(chunk[index], float).copy()
        if not np.isfinite(raw).all():
            raise ValueError("Current policy target must be finite")
        arm, reference = self.arm, self.reference
        nominal = raw.copy()
        nominal[:6] = np.clip(nominal[:6], arm.limits[:, 0], arm.limits[:, 1])
        nominal[6] = np.clip(nominal[6], 0., .101)
        q = arm.data.qpos[arm.qadr].copy()
        qdot_nom = (nominal[:6] - q) / self.dt
        center, rotation = self.eef_pose()
        jp, jr = reference.get_point_jacobians(arm.model, arm.data, center, self.eef_body)
        J = np.vstack((jp[:, arm.dadr], jr[:, arm.dadr]))
        twist_nom = J @ qdot_nom
        vs, ws, debug = self.layer.filter(center, rotation, twist_nom[:3], twist_nom[3:])
        twist_safe = np.r_[vs, ws]
        dqdot = reference.damped_pinv(J, reference.DLS_RHO) @ (twist_safe - twist_nom)
        unclipped = nominal[:6] + dqdot * self.dt
        output = nominal.copy()
        output[:6] = np.clip(unclipped, arm.limits[:, 0], arm.limits[:, 1])
        achieved = J @ ((output[:6] - q) / self.dt)
        lhs_mapped = float(debug['a_twist'] @ achieved + debug['a_z'] @ debug['u_z'] + 10. * debug['h'])
        if not debug['qp_ok']:
            self.failures.append(dict(action_index=len(self.rows), status=debug['qp_status'], error=debug['qp_error']))
        row = dict(action_index=len(self.rows) if action_index is None else int(action_index),
                   chunk_index=int(index), current_clearance_m=np.nan, deficit_m=0.,
                   geometry_active=bool(np.linalg.norm(output[:6] - nominal[:6]) > 1e-9),
                   geometry_target=output.copy(), nominal_target=nominal,
                   solver_time_ms=debug['qp_ms'], compute_ms=(time.perf_counter_ns()-started)/1e6,
                   h_m=debug['h'], qp_ok=debug['qp_ok'], cbf_lhs_nom=debug['cbf_lhs_nom'],
                   cbf_lhs_qp=debug['cbf_lhs_qp'], cbf_lhs_mapped=lhs_mapped,
                   qp_constraint_satisfied=debug['qp_ok'] and debug['cbf_lhs_qp'] >= -1e-5,
                   mapped_constraint_satisfied=lhs_mapped >= -1e-5,
                   nominal_twist=twist_nom, qp_twist=twist_safe, mapped_twist=achieved,
                   twist_mapping_residual=achieved-twist_safe,
                   joint_correction_rad=output[:6]-nominal[:6],
                   legal_joint_clip=bool(np.max(np.abs(unclipped-output[:6])) > 1e-12),
                   z_before=debug['z_before'], z=debug['z'], u_z=debug['u_z'])
        self.rows.append(row)
        return output, row

    def arrays(self):
        if not self.rows:
            return {"action_index": np.empty(0, int), "geometry_target": np.empty((0, 7)),
                    "solver_time_ms": np.empty(0), "current_clearance_m": np.empty(0),
                    "deficit_m": np.empty(0), "geometry_active": np.empty(0, bool)}
        return {key: np.asarray([row[key] for row in self.rows]) for key in self.rows[0]}

    def metadata(self):
        return dict(algorithm=ALGORITHM, comparator="VLSA-AEGIS paper-algorithm independent action-layer adaptation",
                    official_commit=OFFICIAL_COMMIT, paper="https://arxiv.org/abs/2512.11891v2",
                    action_space="world Cartesian twist + auxiliary z, 9-D QP",
                    physical_qp_weight=np.ones(9).tolist(), cbf_alpha=10., z_ascent_gain=10.,
                    z_initialization="normalized obstacle center minus EEF center",
                    z_integration="z += dt*(I-zzT)*uz, then normalize", action_period_s=self.dt,
                    official_libero_z_dt_s=.05, current_guard=False,
                    measured_servo_offset=False, correction_box_bound=None,
                    joint_target_mapping="DLS correction rho=.05 followed by legal joint range clip; no delta/speed limiter",
                    continuous_safety_guarantee=False,
                    execution_limitation="QP constraint may be violated by DLS, joint clipping, servo tracking, and finite control interval; mapped residual logged",
                    perception="shared GroundingDINO capsule conservatively converted to ellipsoid; no VLM or MVEE",
                    obstacle_center=self.layer.p_obs.tolist(), obstacle_rotation=self.layer.r_obs.tolist(),
                    obstacle_axes=self.layer.q_obs.tolist(), eef_axes=self.layer.eef_axes.tolist(),
                    eef_local_center=self.eef_local_center.tolist(), eef_ellipsoid_scale=1.,
                    eef_enclosure=self.enclosure,
                    geometry_limitation="complete model gripper collision meshes only; excludes visual decorations, carried tool, wrist/forearm/body",
                    protects_carried_tool=False, uses_placement_goal=False, uses_future_policy_actions=False,
                    policy_calls=0, physics_rollouts=0, qp_failures=len(self.failures), failures=self.failures,
                    minimum_h=min((r['h_m'] for r in self.rows), default=None),
                    mapped_constraint_violations=sum(not r['mapped_constraint_satisfied'] for r in self.rows),
                    audited_dependencies=SOURCE_HASHES)


def build_controller(arm, obstacle, initial_tool_height=None, args=None, config=None):
    """Factory compatible with the external benchmark's preview-controller slot."""
    return VLSAIndependent(arm, obstacle, initial_tool_height)
