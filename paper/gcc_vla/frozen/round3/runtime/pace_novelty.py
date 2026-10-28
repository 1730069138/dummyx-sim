"""PACE revision candidates, contact confound switches and matched baselines.

Candidate A, chunk-preview geometry: the local client executes up to eight actions
before querying pi0 again. The APF/CBF baselines implemented in this repository use
the current action; this candidate also rolls out the remaining executed chunk
through private kinematics to assess future clearance. This is an implementation
comparison, not a claim that APF or CBF methods inherently lack preview.

Candidate B, force-confirmed support: consecutive estimated upward reactions under
a nominal downward command lock a median effective TCP height reference. This
avoids both the fixed 0.212 m reference and the older single-observation anchor's
downward drift. It does not reconstruct the physical support plane. Neither
candidate's novelty or task performance is established by its implementation.

All defaults reproduce the historical controller exactly.
"""
import argparse
import numpy as np
import mujoco
from pregrasp_guard import DEFAULTS as PREGRASP_DEFAULTS, add_arguments as add_pregrasp_arguments, suffix as pregrasp_suffix

SUPPORT_PRIOR_Z = 0.212
SUPPORT_GATE_SPAN = 0.023     # historical 0.235 - 0.212
PILLAR_REGION = ((0.10, 0.15), (-0.14, -0.05))  # x<0.10 can hit the arm base on reset


def _positive_int(value):
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def add_novelty_arguments(parser):
    add_pregrasp_arguments(parser)
    parser.add_argument("--geometry_method", choices=("pace", "pace_preview", "chunk_brake", "apf_classic", "cbf"),
                        default="pace", help="pace: historical reactive branch; pace_preview: action-chunk "
                        "preview candidate; chunk_brake: PACS/RAIL-style baseline that slows along the nominal chunk path when the same preview predicts a clearance deficit; apf_classic / cbf: local single-action baselines on the same capsules")
    parser.add_argument("--preview_slack", type=float, default=0.005,
                        help="Clearance credit (m) per future step of predicted chunk risk in pace_preview")
    parser.add_argument("--preview_max_step", type=float, default=0.03,
                        help="Cap (m per control step) of the pace_preview clearance correction")
    parser.add_argument("--preview_lift", type=float, default=1.0,
                        help="Weight of +z added to the separation normal in pace_preview")
    parser.add_argument("--preview_stop", type=int, default=8,
                        help="Absolute chunk index where the preview stops (8 = re-query stride, 50 = whole pi0 chunk)")
    parser.add_argument("--consistency_sched", choices=("off", "on"), default="off",
                        help="on: schedule target clearance and far-step trust by cross-chunk intent consistency")
    parser.add_argument("--consistency_sigma", type=float, default=0.02,
                        help="TCP disagreement scale (m) mapping to consistency c = exp(-d/sigma)")
    parser.add_argument("--consistency_alpha", type=float, default=0.5, help="Margin gain: target*(1+alpha*(1-c))")
    parser.add_argument("--consistency_beta", type=float, default=1.0, help="Slack gain: slack*(1+beta*(1-c))")
    parser.add_argument("--dir_select", choices=("off", "on"), default="off",
                        help="on: choose the deflection direction among candidates by previewed chunk clearance")
    parser.add_argument("--requery_on_intervene", choices=("off", "on"), default="off",
                        help="on: discard the rest of the chunk and re-query pi0 when a deflection starts")
    parser.add_argument("--cbf_gamma", type=float, default=0.5, help="CBF class-K gain per control step")
    parser.add_argument("--cbf_buffer", type=float, default=0.01, help="CBF safe-set offset (m)")
    parser.add_argument("--apf_gain", type=float, default=2e-5, help="Khatib APF gain (m^3 per step)")
    parser.add_argument("--support_mode", choices=("prior", "force_anchored", "force_confirmed", "none"), default="prior",
                        help="prior: hard-coded 0.212 m floor; force_anchored: historical force anchor; "
                        "force_confirmed: consecutive force observations lock a median TCP height; none: no height floor")
    parser.add_argument("--support_confirmation_steps", type=_positive_int, default=5,
                        help="Consecutive upward-contact observations with a downward command required by force_confirmed")
    parser.add_argument("--contact_feedback", choices=("on", "off"), default="on",
                        help="off forces the compliance increment xi_c to zero (height floor only)")
    parser.add_argument("--contact_mapping", choices=("legacy", "corrected_v2"), default="legacy",
                        help="legacy preserves historical contact mapping; corrected_v2 maps the complete Cartesian change")
    parser.add_argument("--contact_gate", choices=("legacy", "shared_height", "all_heights", "phase"), default="legacy",
                        help="legacy couples force activation to the support mode; shared_height fixes the gate at 0.235 m; "
                        "all_heights enables force feedback at every height across support-reference comparisons")
    parser.add_argument("--joint_filter", choices=("on", "off"), default="on",
                        help="off sets the geometry-path low-pass beta to 1; the 0.25 rad limit is common to all paths")
    parser.add_argument("--pillar_random_seed", type=int, default=None,
                        help="Sample the pillar position per episode inside PILLAR_REGION")


NOVELTY_DEFAULTS = {**PREGRASP_DEFAULTS, "geometry_method": "pace", "preview_slack": 0.005, "preview_max_step": 0.03, "preview_lift": 1.0, "preview_stop": 8, "consistency_sched": "off", "consistency_sigma": 0.02,
                    "consistency_alpha": 0.5, "consistency_beta": 1.0, "dir_select": "off", "requery_on_intervene": "off", "cbf_gamma": 0.5,
                    "cbf_buffer": 0.01, "apf_gain": 2e-5, "support_mode": "prior",
                    "contact_feedback": "on", "contact_mapping": "legacy", "contact_gate": "legacy",
                    "support_confirmation_steps": 5, "joint_filter": "on", "pillar_random_seed": None}


def novelty_suffix(args):
    # Short tags: run folder names must stay under the 255-byte filesystem limit.
    tags = {"geometry_method": "gm", "support_mode": "sm", "contact_feedback": "cf", "joint_filter": "jf",
            "contact_mapping": "cm", "contact_gate": "cg", "support_confirmation_steps": "sc",
            "preview_lift": "pl", "preview_stop": "pst", "consistency_sched": "cs", "dir_select": "ds",
            "requery_on_intervene": "rq"}
    values = {"pace_preview": "prev", "chunk_brake": "brake", "apf_classic": "apf", "force_anchored": "anch",
              "force_confirmed": "conf", "corrected_v2": "cv2", "shared_height": "sh", "all_heights": "ah", "phase": "ph"}
    parts = [f"{tag}-{values.get(getattr(args, key), getattr(args, key))}" for key, tag in tags.items()
             if getattr(args, key, NOVELTY_DEFAULTS[key]) != NOVELTY_DEFAULTS[key]]
    if getattr(args, "fixed_screw_xy", None) is not None:
        parts.append("fs-%g-%g" % tuple(args.fixed_screw_xy))
    if getattr(args, "pillar_random_seed", None) is not None:
        parts.append(f"ps-{args.pillar_random_seed}")
    return "".join("_" + part for part in parts) + pregrasp_suffix(args)


def pillar_positions(seed, count, region=PILLAR_REGION):
    rng = np.random.default_rng(seed)
    return [(float(rng.uniform(*region[0])), float(rng.uniform(*region[1]))) for _ in range(count)]


def smooth_joint_target_beta(current_q, action, correction, previous, beta):
    """pace_diagnostics.smooth_joint_target with a selectable low-pass beta.

    beta=1 disables the low-pass while keeping the common 0.25 rad limit."""
    smooth = beta * (action[:6] - current_q[:6] + correction) + (1 - beta) * previous
    target = np.zeros(8)
    target[:6] = current_q[:6] + np.clip(smooth, -0.25, 0.25)
    target[6:8] = 0.04 if action[6] > 0.02 else 0.0
    return target, smooth


# ---------------- Candidate A: action-chunk preview ----------------
class ChunkPreview:
    """Predicts arm-capsule clearance along the remaining action chunk.

    Uses a private MjData and mj_kinematics only; the simulated state is untouched.
    The obstacle is static within the chunk (same assumption as the cached geometry).
    """
    def __init__(self, model, distance_fn):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.distance_fn = distance_fn

    def predict(self, data, chunk, start, stop, obstacle, slack, capsule_fn):
        """Returns (risk_distance, normal, steps_ahead) of the riskiest step in chunk[start:stop].

        stop is the re-query stride: later actions of the chunk are never executed.
        Risk of step k is d_k + slack*(k-1); additive so penetration (d<0) keeps its order.
        """
        self.scratch.qpos[:] = data.qpos
        best = (np.inf, None, 0)
        for k, action in enumerate(chunk[start:stop], start=1):
            self.scratch.qpos[:6] = action[:6]
            mujoco.mj_kinematics(self.model, self.scratch)
            for cap in capsule_fn(self.scratch):
                dist, normal, _, _ = self.distance_fn(cap["p1"], cap["p2"], obstacle["p1"], obstacle["p2"])
                risk = dist - cap["r"] - obstacle["r"] + slack * (k - 1)
                if risk < best[0]:
                    best = (risk, normal, k)
        return best

    def min_risk_with_offset(self, data, chunk, start, stop, obstacle, slack, capsule_fn, dq):
        """Minimum discounted risk over chunk[start:stop] when every previewed pose is shifted by dq."""
        self.scratch.qpos[:] = data.qpos
        best = np.inf
        for k, action in enumerate(chunk[start:stop], start=1):
            self.scratch.qpos[:6] = np.asarray(action[:6]) + dq
            mujoco.mj_kinematics(self.model, self.scratch)
            for cap in capsule_fn(self.scratch):
                dist, _, _, _ = self.distance_fn(cap["p1"], cap["p2"], obstacle["p1"], obstacle["p2"])
                best = min(best, dist - cap["r"] - obstacle["r"] + slack * (k - 1))
        return best

    def tcp_path(self, data, actions, site_id):
        self.scratch.qpos[:] = data.qpos
        out = []
        for action in actions:
            self.scratch.qpos[:6] = action[:6]
            mujoco.mj_kinematics(self.model, self.scratch)
            out.append(self.scratch.site_xpos[site_id].copy())
        return np.asarray(out)

    def disagreement(self, data, prev_chunk, prev_start, new_chunk, site_id, length=8):
        """Mean TCP distance between the old plan's unexecuted continuation and the new plan's head."""
        n = min(length, len(prev_chunk) - prev_start, len(new_chunk))
        if n <= 0:
            return np.nan
        a = self.tcp_path(data, prev_chunk[prev_start:prev_start + n], site_id)
        b = self.tcp_path(data, new_chunk[:n], site_id)
        return float(np.mean(np.linalg.norm(a - b, axis=1)))


# ---------------- Baselines on identical capsule geometry ----------------
def cbf_correction(v, normal, distance, gamma, buffer):
    """Closed-form minimum-norm solution of the single-constraint CBF-QP:
    min ||v' - v||^2  s.t.  n.v' >= -gamma (d - buffer)."""
    bound = -gamma * (distance - buffer)
    violation = bound - float(np.dot(normal, v))
    return normal * violation if violation > 0 else np.zeros(3)


def apf_correction(normal, distance, margin, gain):
    """Khatib repulsive potential gradient, displacement per control step."""
    d = max(distance, 1e-3)
    if d >= margin:
        return np.zeros(3)
    return gain * (1.0 / d - 1.0 / margin) / d ** 2 * normal


# ---------------- Force-derived TCP support references ----------------
class SupportEstimator:
    """Anchors the support height at the lowest TCP height of an observed support reaction.

    A support reaction is an estimated contact whose force on the robot points mostly up
    while the TCP is commanded downward. Before the first reaction no floor is applied,
    so the first touch is handled by force feedback alone.
    """
    def __init__(self, vertical_ratio=0.7, offset=0.0):
        self.vertical_ratio, self.offset = vertical_ratio, offset
        self.reset()

    def reset(self):
        self.z = None
        self.events = 0

    def update(self, estimate, tcp_z, commanded_vz):
        force = estimate.force
        norm = float(np.linalg.norm(force))
        if estimate.contact and norm > 0 and force[2] >= self.vertical_ratio * norm and commanded_vz <= 0:
            z = tcp_z - self.offset
            self.z = z if self.z is None else min(self.z, z)
            self.events += 1
        return self.z


class ConfirmedSupportEstimator:
    """Lock a TCP support height after consecutive force-and-command evidence.

    Only the estimated robot force, its contact flag, the measured TCP height and
    the downward Cartesian command are inputs. A nonqualifying observation resets
    the pending window. Once confirmed, its median height is held for the episode;
    repeated penetrations cannot move the inferred floor downward. The value is a
    TCP reference height, not a reconstruction of the physical support surface.
    """
    def __init__(self, confirmation_steps=5, vertical_ratio=0.7, offset=0.0):
        if (isinstance(confirmation_steps, (bool, np.bool_))
                or not isinstance(confirmation_steps, (int, np.integer))
                or confirmation_steps <= 0):
            raise ValueError("confirmation_steps must be a positive integer")
        if not np.isfinite(vertical_ratio) or not 0 < vertical_ratio <= 1:
            raise ValueError("vertical_ratio must be finite and in (0, 1]")
        if not np.isfinite(offset):
            raise ValueError("offset must be finite")
        self.confirmation_steps = int(confirmation_steps)
        self.vertical_ratio = float(vertical_ratio)
        self.offset = float(offset)
        self.reset()

    def reset(self):
        self.z = None
        self.events = 0
        self.candidate_heights = []

    def update(self, estimate, tcp_z, commanded_vz):
        if self.z is not None:
            return self.z
        force = np.asarray(estimate.force, dtype=float)
        finite = (force.shape == (3,) and np.all(np.isfinite(force))
                  and np.isfinite(tcp_z) and np.isfinite(commanded_vz))
        norm = float(np.linalg.norm(force)) if finite else 0.0
        qualifies = (finite and estimate.contact and norm > 0
                     and force[2] >= self.vertical_ratio * norm and commanded_vz < 0)
        if not qualifies:
            self.candidate_heights.clear()
            return None
        self.events += 1
        self.candidate_heights.append(float(tcp_z) - self.offset)
        if len(self.candidate_heights) == self.confirmation_steps:
            self.z = float(np.median(self.candidate_heights))
        return self.z


def support_height(mode, estimator):
    if mode == "prior":
        return SUPPORT_PRIOR_Z
    if mode in ("force_anchored", "force_confirmed"):
        return estimator.z
    if mode == "none":
        return None
    raise ValueError(mode)
