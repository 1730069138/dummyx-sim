"""OSCBF velocity-objective adaptation, not a PACE innovation or full reproduction.

Source: StanfordASL/oscbf 6082329f8e1d61ee8e875a63a5cd80a14d513e12,
OSCBFVelocityConfig._P and CollisionsVelocityConfig.alpha. We retain their
task/null-space metric and alpha=10, but use shared capsule geometry, MuJoCo
kinematics, OSQP, and an explicitly disclosed absolute-position-servo interface.
Independent external controller; no PACE geometry execution or contact compliance.
The original PACE object is used only as a geometry/kinematics provider.
"""
from dataclasses import dataclass, asdict
import time
import numpy as np
from scipy import sparse
import osqp
from tabletop_tool_proxy import batch_capsule_clearances

OFFICIAL_COMMIT = '6082329f8e1d61ee8e875a63a5cd80a14d513e12'

@dataclass(frozen=True)
class OSCBFAdapterConfig:
    alpha: float = 10.0
    horizon_s: float = .05
    clearance_buffer_m: float = 0.
    metric_regularizer: float = 1e-8
    pinv_rcond: float = 1e-8
    feasibility_tolerance: float = 2e-7

def task_metric(J, regularizer=1e-8, rcond=1e-8):
    """Official unit-weight kinematic metric for a nonredundant full pose task.

    For square nonsingular J, the official dynamically consistent inverse equals
    J^-1, and N=0. At rank loss this adapter uses a disclosed Moore-Penrose
    continuation; it does not reproduce the official undefined matrix inverse.
    """
    J = np.asarray(J, float)
    N = np.eye(J.shape[1]) - np.linalg.pinv(J, rcond=rcond) @ J
    P = N.T @ N + J.T @ J
    return (P + P.T) / 2 + regularizer * max(1., float(np.trace(P) / len(P))) * np.eye(len(P))

def solve_projection(nominal, metric, rows, bounds, lower, upper, tolerance=2e-7):
    """Convex local displacement QP; infeasibility is never nominal fallback."""
    nominal, rows, bounds = map(lambda x: np.asarray(x, float), (nominal, rows, bounds))
    if np.all(rows @ nominal >= bounds) and np.all(nominal >= lower) and np.all(nominal <= upper):
        return nominal.copy(), True, dict(status='nominal_feasible', iterations=0)
    A = sparse.csc_matrix(np.vstack([rows, np.eye(len(nominal))]))
    solver = osqp.OSQP()
    solver.setup(P=sparse.csc_matrix(metric), q=-metric @ nominal, A=A,
                 l=np.r_[bounds, lower], u=np.r_[np.full(len(bounds), np.inf), upper],
                 verbose=False, eps_abs=1e-8, eps_rel=1e-8, max_iter=10000,
                 polishing=True, warm_starting=False)
    result = solver.solve(raise_error=False)
    x = nominal.copy() if result.x is None else np.asarray(result.x, float)
    feasible = bool(result.info.status_val in (1, 2) and np.isfinite(x).all()
                    and np.all(rows @ x >= bounds - tolerance)
                    and np.all(x >= lower - tolerance) and np.all(x <= upper + tolerance))
    return x, feasible, dict(status=result.info.status, status_val=int(result.info.status_val),
                             iterations=int(result.info.iter), primal_residual=float(result.info.prim_res),
                             dual_residual=float(result.info.dual_res))

class OSCBFSharedGeometry:
    def __init__(self, helper, config=None):
        self.helper, self.arm = helper, helper.arm
        self.obstacle = helper.obstacle
        self.config = config or OSCBFAdapterConfig()
        self.current_guard = True
        self.refresh_pending = False
        self.execution_predictor = None
        self.rows, self.failures = [], []
        self.calls = self.active_calls = self.rank_deficient_calls = 0
        self.solver_status_counts = {}

    def capsules(self, data, tool_local=None):
        return self.helper.capsules(data, tool_local)

    def _geometry(self, target):
        h, arm = self.helper, self.arm
        h.prepare(h.current)
        tcp, tool = h.current.site_xpos[h.tcp_id], h.current.xpos[h.tool_id]
        attached = bool(target[6] <= .04 and np.linalg.norm(tcp - tool) < .05)
        local = h.carried_tool_local(h.current) if attached else None
        caps = h.capsules(h.current, local)
        gaps, normals, points = batch_capsule_clearances(caps, h.obstacle)
        bodies = [int(c['body_id']) for c in h.eef_capsules_local]
        bodies += [h.wrist_id] * (len(caps) - len(bodies))
        return gaps, normals, points, bodies, attached

    def process(self, chunk, index, **kwargs):
        target = np.asarray(chunk[index], float).copy()
        target[:6] = np.clip(target[:6], self.arm.limits[:, 0], self.arm.limits[:, 1])
        target[6] = np.clip(target[6], 0., .08)
        gaps, _, _, _, _ = self._geometry(target)
        row = dict(action_index=len(self.rows), current_clearance_m=float(gaps.min()),
                   deficit_m=0., geometry_active=False, geometry_target=target.copy())
        self.rows.append(row)
        return target, row

    def project_current_target(self, target):
        start = time.perf_counter_ns()
        arm, cfg, h = self.arm, self.config, self.helper
        target = np.asarray(target, float).copy()
        target[:6] = np.clip(target[:6], arm.limits[:, 0], arm.limits[:, 1])
        q = arm.data.qpos[arm.qadr].copy()
        gaps, normals, points, bodies, attached = self._geometry(target)
        Jp = np.zeros((3, arm.model.nv)); Jr = np.zeros_like(Jp)
        arm.mj.mj_jacSite(arm.model, h.current, Jp, Jr, h.tcp_id)
        J = np.vstack([Jp[:, arm.dadr], Jr[:, arm.dadr]])
        rows = []
        for normal, point, body in zip(normals, points, bodies):
            pointJ = np.zeros_like(Jp)
            arm.mj.mj_jac(arm.model, h.current, pointJ, None, point, body)
            rows.append(normal @ pointJ[:, arm.dadr])
        rows = np.asarray(rows)
        bounds = -cfg.alpha * cfg.horizon_s * (gaps - cfg.clearance_buffer_m)
        nominal = target[:6] - q
        before = rows @ nominal
        active = bool(np.any(before < bounds))
        rank = int(np.linalg.matrix_rank(J, tol=cfg.pinv_rcond * max(1., np.linalg.norm(J, 2))))
        metric = task_metric(J, cfg.metric_regularizer, cfg.pinv_rcond)
        displacement, feasible, info = solve_projection(
            nominal, metric, rows, bounds, arm.limits[:, 0] - q,
            arm.limits[:, 1] - q, cfg.feasibility_tolerance)
        output = target.copy()
        if feasible and info['status'] != 'nominal_feasible':
            output[:6] = np.clip(q + displacement, arm.limits[:, 0], arm.limits[:, 1])
        if feasible:
            feasible = bool(np.all(rows @ (output[:6] - q) >= bounds - cfg.feasibility_tolerance))
        nearest = int(np.argmin(gaps))
        self.calls += 1; self.active_calls += int(active); self.rank_deficient_calls += int(rank < 6)
        self.solver_status_counts[info['status']] = self.solver_status_counts.get(info['status'], 0) + 1
        if not feasible:
            self.failures.append(dict(time_s=float(arm.data.time), **info))
        debug = dict(post_contact_target=target.copy(), reference_guard_active=active,
                     reference_guard_qualified=True, reference_guard_feasible=feasible,
                     reference_guard_clearance_m=float(gaps[nearest]),
                     reference_guard_bound_m=float(bounds[nearest]),
                     reference_guard_before_m=float(before[nearest]),
                     reference_guard_after_m=float((rows @ (output[:6] - q))[nearest]),
                     reference_guard_joint_delta_rad=output[:6] - target[:6],
                     route_preference_qualified=False, route_projection_active=False,
                     route_near_endpoint_m=0., protected_component_count=len(gaps),
                     constrained_component_count=len(gaps), attached_tool=attached,
                     oscbf_task_jacobian_rank=rank,
                     minimum_constraint_residual_m=float(np.min(rows @ (output[:6]-q)-bounds)),
                     solver_compute_ms=(time.perf_counter_ns()-start)/1e6)
        return output, debug

    def arrays(self):
        return {k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]} if self.rows else {}

    def metadata(self):
        return dict(algorithm='OSCBF_kinematic_shared_geometry_position_servo_adaptation',
                    official_repository='https://github.com/StanfordASL/oscbf',
                    official_commit=OFFICIAL_COMMIT, config=asdict(self.config),
                    objective='unit-weight official task/null-space velocity metric; numerical regularizer disclosed',
                    inputs='same DINO/RGB-D obstacle and component capsules, same simulator-assisted tool pose as PACE',
                    bounds='joint legal ranges converted to virtual-horizon velocity bounds; native URDF speed bounds unavailable, not reproduced',
                    interface='qdot_des=(legal_target-q)/0.05; q_target=q+0.05*qdot_filtered; independent external filter every1ms; PACE G/C both disabled',
                    geometry='actual closest-point arm Jacobian; carried tool treated rigid only under original command/proximity qualification',
                    qualification='robot geometry always protected, no PACE pickup gate',
                    fallback='verified infeasible reference aborts before submission, recorded failure; no nominal fallback',
                    limitations='not native velocity control, no continuous invariance or original benchmark reproduction; no free-tool control; rank loss uses pseudoinverse continuation',
                    calls=self.calls, active_calls=self.active_calls,
                    rank_deficient_calls=self.rank_deficient_calls,
                    solver_status_counts=self.solver_status_counts, failures=self.failures,
                    shared_geometry=dict(proxy_mode=self.helper.proxy_mode, eef_capsules=len(self.helper.eef_capsules_local), controller_execution_called=False),
                    pace_geometry=False, pace_contact=False, independent_external=True)
