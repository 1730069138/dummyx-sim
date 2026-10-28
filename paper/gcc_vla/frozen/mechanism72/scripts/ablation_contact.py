"""A: frozen process with the force velocity suppressed before mapping."""
import numpy as np
from tabletop_pace_runtime import TabletopAdmittance
class NoForceVelocity(TabletopAdmittance):
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
        self.compensation[:] = 0.0  # A: suppress force velocity only; anchor feedback retained
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
