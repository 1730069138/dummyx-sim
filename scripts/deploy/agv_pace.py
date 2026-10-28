"""AGV force recording and opt-in admittance; no chunk preview."""

from dataclasses import asdict, dataclass
import numpy as np
from joint_torque_observer import MujocoTorqueObserver, ForceEstimate, arm_body_ids


@dataclass(frozen=True)
class AGVContactConfig:
    gate_above_surface_m: float = .035
    braking_span_m: float = .03
    confirmation_time_s: float = .005
    max_compensation_m_s: float = .05
    max_joint_correction_rad: float = .05
    dls_damping: float = .05

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("Contact configuration must be finite and positive")


class AGVAdmittance:
    """Velocity-unit adaptation of corrected_v2 plus force-confirmed TCP anchor.

    Height is an effective TCP reference, not an estimated physical plane.
    A scene-derived near-surface gate qualifies both feedback and confirmation.
    No prior height floor is inserted before force confirmation.
    """

    def __init__(self, arm, config=None):
        import mujoco
        from pace_novelty import ConfirmedSupportEstimator
        self.arm, self.config = arm, config or AGVContactConfig()
        self.dt = float(arm.model.opt.timestep)
        self.action_dt = self.dt * arm.steps
        count = int(np.ceil(self.config.confirmation_time_s / self.dt - 1e-12))
        self.support = ConfirmedSupportEstimator(confirmation_steps=max(1, count))
        self.scratch = mujoco.MjData(arm.model)
        self.compensation = np.zeros(3)
        self.latest = ForceEstimate(np.zeros(3), np.zeros(6), False, np.zeros(3))
        self.latest_time = None
        self.tcp_id = arm.model.site("tcp_site").id
        self.surfaces = {}
        # These are known static scene references, not force-estimated geometry.
        for name in ("workbench_top", "bin_bottom"):
            geom = arm.model.geom(name)
            rotation = arm.data.geom_xmat[geom.id].reshape(3, 3)
            if geom.type != mujoco.mjtGeom.mjGEOM_BOX or abs(rotation[2, 2]) < 1 - 1e-8:
                raise ValueError("Contact reference requires horizontal box support geometry")
            self.surfaces[name] = dict(center=arm.data.geom_xpos[geom.id].copy(),
                                       rotation=rotation.copy(), half_size=geom.size.copy(),
                                       top=float(arm.data.geom_xpos[geom.id, 2] + geom.size[2]))

    def geometry(self):
        # Private kinematics at CURRENT q: never refresh live dynamics between
        # mj_step and observer sampling; do not use the previous step's Jacobian.
        arm = self.arm
        self.scratch.qpos[:] = arm.data.qpos
        arm.mj.mj_kinematics(arm.model, self.scratch)
        arm.mj.mj_comPos(arm.model, self.scratch)
        J = np.zeros((3, arm.model.nv))
        arm.mj.mj_jacSite(arm.model, self.scratch, J, None, self.tcp_id)
        return self.scratch.site_xpos[self.tcp_id].copy(), J[:, arm.dadr]

    def surface(self, tcp):
        for name in ("bin_bottom", "workbench_top"):
            box = self.surfaces[name]
            local = box["rotation"].T @ (tcp - box["center"])
            if np.all(np.abs(local[:2]) <= box["half_size"][:2]):
                return box
        return None

    def process(self, nominal):
        arm, cfg = self.arm, self.config
        nominal = np.asarray(nominal, dtype=float)
        if nominal.shape != (7,) or not np.isfinite(nominal).all():
            raise ValueError("Expected seven finite nominal targets")
        if np.any(nominal[:6] < arm.limits[:, 0]) or np.any(nominal[:6] > arm.limits[:, 1]):
            raise ValueError("Nominal joint target outside limits")
        tcp, J = self.geometry()
        q = arm.data.qpos[arm.qadr].copy()
        raw = J @ (nominal[:6] - q) / self.action_dt
        surface = self.surface(tcp)
        surface_top = 0. if surface is None else surface["top"]
        gate_height = 0. if surface is None else surface_top + cfg.gate_above_surface_m
        gate = bool(surface is not None and tcp[2] < gate_height)
        height_target = raw.copy()
        z = self.support.z
        if gate and z is not None and height_target[2] < 0 and tcp[2] - z < cfg.braking_span_m:
            height_target[2] *= np.clip((tcp[2] - z) / cfg.braking_span_m, .02, 1.) ** 2
            if tcp[2] < z:
                height_target[2] = 0.
        force = self.latest.force if gate and self.latest.contact else np.zeros(3)
        norm = float(np.linalg.norm(force))
        ideal = np.zeros(3)
        if norm > .5:
            direction = force / norm * [0.1, 0.1, 1.]
            direction /= np.linalg.norm(direction) + 1e-6
            ideal = (norm - .5) * (.001 + .003 * norm) * direction
            ideal *= min(1., cfg.max_compensation_m_s / max(np.linalg.norm(ideal), 1e-12))
        # Preserve the legacy 1 ms filter time constant, not its per-step alpha.
        alpha_1ms = np.clip(.02 + .01 * norm, .02, .1)
        alpha = -np.expm1(np.log1p(-alpha_1ms) * self.dt / .001)
        if gate:
            self.compensation += alpha * (ideal - self.compensation)
        else:
            # Full pass-through outside the qualified region; no stale retreat.
            self.compensation[:] = 0.
        velocity = height_target + self.compensation
        correction = J.T @ np.linalg.solve(J @ J.T + cfg.dls_damping**2 * np.eye(3),
                                           (velocity - raw) * self.action_dt)
        correction = np.clip(correction, -cfg.max_joint_correction_rad, cfg.max_joint_correction_rad)
        output = nominal.copy()
        output[:6] = np.clip(nominal[:6] + correction, arm.limits[:, 0], arm.limits[:, 1])
        debug = dict(nominal_action_target=nominal.copy(), contact_gate=gate,
                     contact_gate_height_m=gate_height, surface_top_m=surface_top,
                     support_before_valid=z is not None, support_before_m=0. if z is None else z,
                     nominal_velocity_m_s=raw.copy(), height_delta_m_s=height_target - raw,
                     force_delta_m_s=self.compensation.copy(),
                     mapped_velocity_m_s=J @ (output[:6] - q) / self.action_dt,
                     feedback_force_N=np.asarray(force).copy(),
                     feedback_sample_time_s=0. if self.latest_time is None else self.latest_time,
                     feedback_sample_valid=self.latest_time is not None,
                     confirmation_tcp_z=float(tcp[2]))
        return output, debug

    def after_step(self, estimate, debug, sample_time):
        # Estimate and TCP/nominal intent belong to the SAME just-completed interval.
        if debug["contact_gate"]:
            self.support.update(estimate, debug["confirmation_tcp_z"], debug["nominal_velocity_m_s"][2])
        elif self.support.z is None:
            self.support.candidate_heights.clear()
        self.latest = estimate
        self.latest_time = float(sample_time)
        debug.update(support_after_valid=self.support.z is not None,
                     support_after_m=0. if self.support.z is None else self.support.z,
                     support_pending_samples=len(self.support.candidate_heights))

    def metadata(self):
        return dict(config=asdict(self.config), action_period_s=self.action_dt,
                    confirmation_steps=self.support.confirmation_steps,
                    confirmation_effective_time_s=self.support.confirmation_steps * self.dt,
                    surfaces={name: value["top"] for name, value in self.surfaces.items()},
                    support_mode="force_confirmed", mapping="corrected_v2_velocity_units",
                    qualification="TCP below local surface top + gate span; upward estimated contact and nominal downward intent",
                    compensation_units="m/s; mapped to absolute joint target using action_period_s",
                    outside_gate="pass-through and clear compensation; retain confirmed anchor until episode reset")


class AGVTorqueObserver(MujocoTorqueObserver):
    """Observe joint1..joint6, keeping the AGV model's integrator unchanged.

    Caller: reset(data), then mj_step(model, data) and sample_after_step(data)
    for EVERY physics substep. HardstopArm.step() calls mj_forward at its end,
    so sampling only after that outer method is invalid.
    """

    def __init__(self, arm, config=None):
        super().__init__(arm.model, arm.model.site("tcp_site").id, config,
                         joint_names=tuple(f"joint{i}" for i in range(1, 7)))
        if tuple(self.dof_indices) != tuple(arm.dadr):
            raise ValueError("Observer and HardstopArm DOF mappings differ")
        self.qpos_indices = arm.qadr.copy()
        self.robot_bodies = arm_body_ids(arm.model, root_name="link0")


class AGVForceRecorder:
    """Synchronized substep records, with no contact-truth read or feedback."""

    SHAPES = {"action_index": (), "substep_index": (), "time_before_s": (),
              "time_after_s": (), "q_before": (6,), "q_after": (6,),
              "qvel_before": (6,), "qvel_after": (6,), "action_target": (7,),
              "tcp_before": (3,), "estimated_force": (3,), "raw_force": (3,),
              "tau_external": (6,), "estimated_contact": ()}

    def __init__(self, arm, config=None):
        self.arm = arm
        self.observer = AGVTorqueObserver(arm, config)
        self.observer.reset(arm.data)
        self.rows = []
        self.action_index = 0

    def before_step(self, action, substep):
        arm = self.arm
        return dict(action_index=self.action_index, substep_index=substep,
                    time_before_s=float(arm.data.time),
                    q_before=arm.data.qpos[arm.qadr].copy(),
                    qvel_before=arm.data.qvel[arm.dadr].copy(),
                    action_target=np.asarray(action).copy(),
                    actuator_ctrl=arm.data.ctrl.copy())

    def after_step(self, row):
        arm = self.arm
        # This MUST precede mj_forward: kinematics and dynamics still refer to
        # time_before_s, while integrated qpos/qvel refer to time_after_s.
        estimate = self.observer.sample_after_step(arm.data)
        # Pure diagnostics: mj_step still holds the just-used actuator dynamics.
        # Current AGV servos use fixed gain / affine bias with unit joint gear.
        m, d = arm.model, arm.data
        ctrl = d.ctrl.copy()
        limited = m.actuator_ctrllimited.astype(bool)
        ctrl[limited] = np.clip(ctrl[limited], m.actuator_ctrlrange[limited, 0],
                                m.actuator_ctrlrange[limited, 1])
        demand = (m.actuator_gainprm[:, 0] * ctrl + m.actuator_biasprm[:, 0]
                  + m.actuator_biasprm[:, 1] * d.actuator_length
                  + m.actuator_biasprm[:, 2] * d.actuator_velocity)
        force_limited = m.actuator_forcelimited.astype(bool)
        saturated = force_limited & ((demand < m.actuator_forcerange[:, 0])
                                     | (demand > m.actuator_forcerange[:, 1]))
        row.update(time_after_s=float(arm.data.time),
                   q_after=arm.data.qpos[arm.qadr].copy(),
                   qvel_after=arm.data.qvel[arm.dadr].copy(),
                   tcp_before=arm.data.site_xpos[self.observer.tcp_id].copy(),
                   estimated_force=estimate.force.copy(), raw_force=estimate.raw_force.copy(),
                   tau_external=estimate.tau_external.copy(), estimated_contact=estimate.contact,
                   actuator_force=d.actuator_force.copy(), actuator_force_demand=demand,
                   actuator_saturated=saturated,
                   joint_target_error_rad=np.asarray(row["action_target"][:6]) - d.qpos[arm.qadr])
        self.rows.append(row)
        return estimate

    def arrays(self):
        shapes = {**self.SHAPES, "actuator_ctrl": (self.arm.model.nu,),
                  "actuator_force": (self.arm.model.nu,),
                  "actuator_force_demand": (self.arm.model.nu,),
                  "actuator_saturated": (self.arm.model.nu,), "joint_target_error_rad": (6,)}
        controller = getattr(self.arm, "contact_controller", None)
        if controller is not None:
            shapes.update(CONTACT_LOG_SHAPES)
        preview = getattr(self.arm, "preview_controller", None)
        if preview is not None and preview.current_guard:
            from agv_pace_preview import GUARD_LOG_SHAPES
            shapes.update(GUARD_LOG_SHAPES)
        result = {}
        for key, shape in shapes.items():
            dtype = bool if key in ("actuator_saturated", "estimated_contact", "contact_gate", "support_before_valid",
                                   "support_after_valid", "feedback_sample_valid", "reference_guard_active",
                                   "reference_guard_qualified", "reference_guard_feasible") else np.int64 if key in (
                "action_index", "substep_index", "support_pending_samples") else float
            result[key] = np.asarray([row[key] for row in self.rows], dtype=dtype).reshape((-1, *shape))
        return result

    def metadata(self):
        controller = getattr(self.arm, "contact_controller", None)
        preview = getattr(self.arm, "preview_controller", None)
        saturation = np.asarray([r["actuator_saturated"] for r in self.rows], dtype=bool).reshape((-1, self.arm.model.nu))
        dt = float(self.arm.model.opt.timestep)
        return dict(schema="agv_contact_substep_v3" if controller else "agv_force_substep_v2",
                    actuator_names=[self.arm.mj.mj_id2name(self.arm.model, self.arm.mj.mjtObj.mjOBJ_ACTUATOR, i)
                                    for i in range(self.arm.model.nu)],
                    saturation_seconds=(saturation.sum(axis=0) * dt).tolist(),
                    saturation_fraction=(saturation.mean(axis=0) if len(saturation) else np.zeros(self.arm.model.nu)).tolist(),
                    saturation_definition="fixed/affine servo demand outside actuator force range; duration is samples*physics_dt; does not diagnose joint-level or solver limits",
                    feedback=controller is not None, contact_controller=controller.metadata() if controller else None,
                    final_reference_guard=bool(preview is not None and preview.current_guard),
                    samples=len(self.rows),
                    completed_actions=self.action_index,
                    physics_timestep_s=float(self.arm.model.opt.timestep),
                    substeps_per_action=self.arm.steps,
                    integrator=int(self.arm.model.opt.integrator),
                    joint_names=list(self.observer.joint_names),
                    qpos_indices=self.observer.qpos_indices.tolist(),
                    dof_indices=self.observer.dof_indices.tolist(),
                    observer_config=asdict(self.observer.observer.config),
                    alignment="action_index is zero-based relative to policy actions, excludes warmup; "
                    "q_before/qvel_before/tcp_before and dynamics at time_before_s; "
                    "q_after/qvel_after at time_after_s; force estimated from that velocity interval; "
                    "actuator_ctrl written before mj_step; partial final actions remain recorded; "
                    "with feedback, action_target is actual substep target and nominal_action_target is the outer target; "
                    "feedback_sample_time_s is the prior interval end, support_after is available only to the next substep")


CONTACT_LOG_SHAPES = {"nominal_action_target": (7,), "contact_gate": (),
    "contact_gate_height_m": (), "surface_top_m": (), "support_before_valid": (),
    "support_before_m": (), "nominal_velocity_m_s": (3,), "height_delta_m_s": (3,),
    "force_delta_m_s": (3,), "mapped_velocity_m_s": (3,), "feedback_force_N": (3,),
    "feedback_sample_time_s": (), "feedback_sample_valid": (), "confirmation_tcp_z": (),
    "support_after_valid": (), "support_after_m": (), "support_pending_samples": ()}
