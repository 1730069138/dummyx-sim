"""Model residual force observer; SI units, world force acting ON the robot.

Core update accepts synchronized robot-model quantities, not contact solver data.
MuJoCo adapter samples immediately AFTER an Euler mj_step: qM, bias, passive,
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
    """Uses only model, encoder velocities and qfrc_actuator[:6]."""
    def __init__(self, model, tcp_id, config=None):
        import mujoco
        if model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
            raise ValueError("Timestamp-aligned adapter currently requires Euler integration")
        self.model, self.tcp_id = model, tcp_id
        self.observer = ModelResidualObserver(config)
        self.previous_velocity = None
        self.jacobian = np.zeros((3, model.nv))
        self.inertia_force = np.zeros(model.nv)

    def reset(self, data):
        self.previous_velocity = data.qvel.copy()
        self.observer.reset()

    def sample_after_step(self, data):
        import mujoco
        if self.previous_velocity is None:
            raise RuntimeError("reset(data) before stepping")
        dt = self.model.opt.timestep
        delta_velocity = data.qvel - self.previous_velocity
        acceleration = delta_velocity / dt
        self.previous_velocity[:] = data.qvel
        mujoco.mj_mulM(self.model, data, self.inertia_force, acceleration)
        # Euler integrates joint damping implicitly: (M + dt*D) dv/dt.
        # Restore the pre-step passive-force convention without solver forces.
        if not self.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP):
            self.inertia_force += self.model.dof_damping * delta_velocity
        mujoco.mj_jacSite(self.model, data, self.jacobian, None, self.tcp_id)
        return self.observer.update(
            self.inertia_force[:6], data.qfrc_bias[:6], data.qfrc_passive[:6],
            data.qfrc_actuator[:6], self.jacobian[:, :6], dt)


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


def arm_body_ids(model):
    import mujoco
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link1_1")
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
