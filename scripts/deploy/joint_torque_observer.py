"""Model residual force observer; SI units, world force acting ON the robot.

Core update accepts synchronized robot-model quantities, not contact solver data.
MuJoCo adapter samples immediately AFTER an Euler/implicitfast mj_step: qM, bias, passive,
actuator force and kinematics still describe the pre-integration state, while
qvel is the new velocity. This permits encoder-style acceleration differencing.
Do not call mj_forward between mj_step and sample_after_step.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ObserverConfig:
    damping: float = 0.02       # Jacobian regularization, m/rad
    cutoff_hz: float = 20.0
    deadzone: float = 0.25      # radial soft deadzone, N
    max_force: float = 100.0    # vector norm limit, N
    enter_force: float = 0.8   # thresholds apply AFTER filtering/deadzone
    exit_force: float = 0.4

    def __post_init__(self):
        values = tuple(vars(self).values())
        if not all(np.isfinite(values)) or not (
            self.damping > 0 and self.cutoff_hz > 0 and self.deadzone >= 0
            and 0 <= self.exit_force < self.enter_force <= self.max_force
        ):
            raise ValueError("Invalid force observer parameters")


@dataclass
class ForceEstimate:
    force: np.ndarray
    tau_external: np.ndarray
    contact: bool
    raw_force: np.ndarray = None


class ModelResidualObserver:
    def __init__(self, config=None):
        self.config = config or ObserverConfig()
        self.reset()

    def reset(self):
        self.filtered_force = np.zeros(3)
        self.contact = False

    def update(self, inertia_force, bias, passive, tau_measured, jacobian, dt):
        """inertia_force = full M(q) @ qdd, restricted to measured arm rows.

        M qdd + bias = actuator + passive + external (including unmodeled
        friction/limits). Never subtract qfrc_constraint: it contains contact.
        Minimize ||J.T F - tau_external||^2 + damping^2 ||F||^2.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        inputs = [np.asarray(x, dtype=float) for x in
                  (inertia_force, bias, passive, tau_measured, jacobian)]
        if not all(np.all(np.isfinite(x)) for x in inputs):
            raise ValueError("Non-finite dynamics or joint feedback")
        inertia_force, bias, passive, tau_measured, J = inputs
        tau = inertia_force + bias - passive - tau_measured
        raw = np.linalg.solve(J @ J.T + self.config.damping**2 * np.eye(3), J @ tau)
        alpha = -np.expm1(-2 * np.pi * self.config.cutoff_hz * dt)
        self.filtered_force += alpha * (raw - self.filtered_force)
        norm = np.linalg.norm(self.filtered_force)
        magnitude = min(max(norm - self.config.deadzone, 0.0), self.config.max_force)
        force = self.filtered_force * (magnitude / norm) if norm > 0 else np.zeros(3)
        if self.contact:
            self.contact = magnitude > self.config.exit_force
        else:
            self.contact = magnitude >= self.config.enter_force
        return ForceEstimate(force, tau.copy(), self.contact, raw)


class MujocoTorqueObserver:
    """Encoder residual with integrator correction and explicit arm DOF mapping.

    joint_names=None preserves the original six leading DOFs. For implicitfast,
    (M - h*D) dv/h recovers the pre-integration generalized force; D is the
    smooth-force velocity derivative produced by the integrator, not contact truth.
    Sample exactly once after each mj_step, before mj_forward or another step.
    """
    def __init__(self, model, tcp_id, config=None, *, joint_names=None):
        import mujoco
        if model.opt.integrator not in (mujoco.mjtIntegrator.mjINT_EULER,
                                        mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
            raise ValueError("Timestamp-aligned adapter requires Euler or implicitfast integration")
        if not 0 <= tcp_id < model.nsite:
            raise ValueError("Invalid TCP site")
        self.model, self.tcp_id = model, tcp_id
        self.joint_names = None if joint_names is None else tuple(joint_names)
        if self.joint_names is None:
            if model.nv < 6:
                raise ValueError("Default observer requires six leading DOFs")
            self.dof_indices = np.arange(6)
        else:
            if len(self.joint_names) != 6 or len(set(self.joint_names)) != 6:
                raise ValueError("Expected six unique scalar arm joints")
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                   for n in self.joint_names]
            if any(j < 0 or model.jnt_type[j] not in
                   (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) for j in ids):
                raise ValueError("Arm joint missing or not scalar")
            self.dof_indices = model.jnt_dofadr[ids].copy()
        self.integrator = int(model.opt.integrator)
        self.observer = ModelResidualObserver(config)
        self.previous_velocity = None
        self.previous_time = None
        self.jacobian = np.zeros((3, model.nv))
        self.inertia_force = np.zeros(model.nv)

    def reset(self, data):
        self.previous_velocity = data.qvel.copy()
        self.previous_time = float(data.time)
        self.observer.reset()

    def sample_after_step(self, data):
        import mujoco
        if self.previous_velocity is None:
            raise RuntimeError("reset(data) before stepping")
        dt = self.model.opt.timestep
        if int(self.model.opt.integrator) != self.integrator:
            raise RuntimeError("Integrator changed; recreate observer")
        if not np.isclose(data.time - self.previous_time, dt, rtol=1e-7, atol=1e-12):
            raise RuntimeError("Sample exactly once per physics step; reset after model reset")
        delta_velocity = data.qvel - self.previous_velocity
        acceleration = delta_velocity / dt
        self.previous_velocity[:] = data.qvel
        self.previous_time = float(data.time)
        mujoco.mj_mulM(self.model, data, self.inertia_force, acceleration)
        # Euler integrates joint damping implicitly: (M + dt*D) dv/dt.
        # Restore the pre-step passive-force convention without solver forces.
        if self.integrator == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
            # MuJoCo 3.3.2 gathers the lower triangle of qDeriv and uses a
            # symmetric factorization. Mirror that triangle, not a full unsymmetric D.
            derivative = np.zeros((self.model.nv, self.model.nv))
            for row in range(self.model.nv):
                start = int(data.D_rowadr[row])
                stop = start + int(data.D_rownnz[row])
                derivative[row, data.D_colind[start:stop]] = data.qDeriv[start:stop]
            lower = np.tril(derivative)
            symmetric = lower + np.tril(derivative, -1).T
            self.inertia_force -= symmetric @ delta_velocity
        elif not self.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP):
            self.inertia_force += self.model.dof_damping * delta_velocity
        mujoco.mj_jacSite(self.model, data, self.jacobian, None, self.tcp_id)
        indices = self.dof_indices
        return self.observer.update(
            self.inertia_force[indices], data.qfrc_bias[indices], data.qfrc_passive[indices],
            data.qfrc_actuator[indices], self.jacobian[:, indices], dt)


def select_control_force(source, estimate, legacy_contact_reader):
    """Lazy legacy reader keeps contact solver forces out of torque control."""
    if source == "joint_torque":
        return estimate.force.copy() if estimate.contact else np.zeros(3)
    if source == "contact":
        return legacy_contact_reader()
    raise ValueError(f"Unknown contact force source: {source}")


def robot_contact_truth(model, data, robot_body_ids):
    """Offline net environment force on robot; exclude internal contacts.

    MuJoCo contact wrench acts on geom2; reverse for geom1. No z sign folding.
    This truth excludes table contacts of loose objects not in the robot tree.
    """
    import mujoco
    total = np.zeros(3)
    wrench = np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        first = int(model.geom_bodyid[contact.geom1]) in robot_body_ids
        second = int(model.geom_bodyid[contact.geom2]) in robot_body_ids
        if first == second:
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        total += (1 if second else -1) * (contact.frame.reshape(3, 3).T @ wrench[:3])
    return total


def arm_body_ids(model, root_name="link1_1"):
    import mujoco
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if root < 0:
        raise ValueError(f"Robot root body not found: {root_name}")
    result = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in result:
            result.add(body)
    return result


def add_observer_arguments(parser):
    parser.add_argument("--contact_force_source", choices=("joint_torque", "contact"), default="joint_torque")
    defaults = ObserverConfig()
    help_text = {
        "damping": "TCP Jacobian damping (m/rad)",
        "cutoff_hz": "Force low-pass cutoff (Hz)",
        "deadzone": "Radial soft deadzone (N)",
        "max_force": "Output force norm limit (N)",
        "enter_force": "Contact entry after filtering/deadzone (N)",
        "exit_force": "Contact exit after filtering/deadzone (N)",
    }
    for name, value in vars(defaults).items():
        parser.add_argument(f"--force_{name}", type=float, default=value, help=help_text[name])


def config_from_args(args):
    return ObserverConfig(**{name: getattr(args, f"force_{name}") for name in vars(ObserverConfig())})


class ForceLog:
    """Physics-rate, synchronized observer and evaluation samples for NPZ."""
    def __init__(self):
        self.values = {name: [] for name in (
            "force_time_s", "f_estimated", "f_raw", "f_contact_truth", "tau_external",
            "force_error", "estimated_contact", "truth_contact", "f_control")}

    def append(self, time_s, estimate, truth, control, truth_threshold):
        values = (time_s, estimate.force.copy(),
                  estimate.raw_force.copy() if estimate.raw_force is not None else estimate.force.copy(),
                  truth.copy(), estimate.tau_external.copy(),
                  estimate.force - truth, estimate.contact,
                  np.linalg.norm(truth) >= truth_threshold, control.copy())
        for key, value in zip(self.values, values):
            self.values[key].append(value)

    def arrays(self):
        return {key: np.asarray(value) for key, value in self.values.items()}
