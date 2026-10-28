"""Opt-in PACE diagnostics. Never supplies a command or geometry to a controller.

Kinematic evaluations use a private MjData. A free tool has no known arm-to-tool
kinematic constraint: its full point Jacobian is saved, but arm conflict metrics
are unavailable. Linearized command clearances are not dynamic predictions.
"""
from pathlib import Path
import hashlib
import json
import argparse

import mujoco
import numpy as np
from joint_torque_observer import ObserverConfig
from pace_novelty import NOVELTY_DEFAULTS, novelty_suffix


GOAL_REFERENCE = np.array([0.0, -0.1, 0.24])
PILLAR_XY = (0.11, -0.10)
R2_BEHAVIOR = (-0.0611895161, -0.0512742457, 0.2782245065)


def finite_coordinate(value):
    value = float(value)
    if not np.isfinite(value):
        raise argparse.ArgumentTypeError("Coordinates must be finite")
    return value


def scene_configuration(args):
    pillar = np.asarray(getattr(args, "pillar_xy", PILLAR_XY), dtype=float)
    goal = np.asarray(getattr(args, "goal_reference", GOAL_REFERENCE), dtype=float)
    if pillar.shape != (2,) or goal.shape != (3,) or not np.all(np.isfinite(pillar)) or not np.all(np.isfinite(goal)):
        raise ValueError("Expected finite pillar_xy[2] and goal_reference[3]")
    scene = next((name for name, xy in (("S0", (.11, -.10)), ("S1", (.13, .015)), ("S2", (.12, .015)))
                  if np.array_equal(pillar, xy)), "custom")
    reference = ("R0" if np.array_equal(goal, GOAL_REFERENCE) else
                 "R2_behavior_v1" if np.array_equal(goal, R2_BEHAVIOR) else "custom")
    return {"scene_id": scene, "pillar_xy": pillar.tolist(), "pillar_reset_z_m": 0.38,
            "goal_reference": goal.tolist(), "reference_id": reference}


def scene_metadata_arrays(config):
    return {**{key: np.asarray(config[key]) for key in ("scene_id", "pillar_xy", "goal_reference", "reference_id")},
            "source_sha256_json": np.array(json.dumps(config["source_sha256"], sort_keys=True))}


ZERO_TOL = 1e-12  # Diagnostics only; does not gate control.


def add_experiment_arguments(parser):
    parser.add_argument("--pillar_xy", nargs=2, type=finite_coordinate, default=PILLAR_XY, metavar=("X", "Y"))
    parser.add_argument("--goal_reference", nargs=3, type=finite_coordinate,
                        default=tuple(GOAL_REFERENCE), metavar=("X", "Y", "Z"))
    parser.add_argument("--experiment_protocol", choices=("legacy", "controlled"),
                        default="legacy", help="legacy preserves historical reset/filter behavior")
    parser.add_argument("--safety_margin_mode", choices=("dynamic", "fixed_small", "fixed_large"),
                        default="dynamic", help="dynamic, 0.025 m or 0.080 m")
    parser.add_argument("--goal_decay_mode", choices=("legacy", "disabled"), default="legacy",
                        help="Historical goal decay or factor fixed at one")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Save separate 1 kHz diagnostic chunks; default off; adds wall-clock cost")
    parser.add_argument("--geometry_impl", choices=("current", "alias_copy_v1"), default="current",
                        help="Preserve historical array aliasing or explicitly fix the nonnegative-normal branch")
    parser.add_argument("--initial_state_protocol", choices=("historical", "full_v1"), default="historical",
                        help="Historical physical reset or mj_resetData followed by the same scene initialization")


def safety_margin(distance_to_goal, mode="dynamic"):
    if mode == "dynamic":
        return np.clip(0.025 + (distance_to_goal * 0.4), 0.025, 0.08)
    if mode == "fixed_small":
        return 0.025
    if mode == "fixed_large":
        return 0.08
    raise ValueError(f"Unknown safety margin mode: {mode}")


def goal_factor(distance_to_goal, mode="legacy"):
    if mode == "legacy":
        return np.clip((distance_to_goal - 0.035) / 0.065, 0.0, 1.0)
    if mode == "disabled":
        return 1.0
    raise ValueError(f"Unknown goal decay mode: {mode}")


def smooth_joint_target(current_q, action, correction, previous):
    """Historical arithmetic order; common APF-on/off path in controlled mode."""
    nominal = action[:6] - current_q[:6]
    combined = nominal + correction
    beta = 0.6
    smooth = beta * combined + (1 - beta) * previous
    target = np.zeros(8)
    target[:6] = current_q[:6] + np.clip(smooth, -0.25, 0.25)
    target[6:8] = 0.04 if action[6] > 0.02 else 0.0
    return target, smooth


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(json_safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def physics_provenance(model):
    """Read contact/solver settings from the compiled model without modifying it.

    Geom settings are nominal material parameters, before contact-specific mixing.
    Explicit pairs can enable contacts even when both geom bitmasks are zero.
    """
    pair_geom_ids = set(map(int, model.pair_geom1)) | set(map(int, model.pair_geom2))
    geoms = []
    for geom_id in range(model.ngeom):
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]
                or geom_id in pair_geom_ids):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        geoms.append({
            "id": geom_id,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
            "body_id": body_id,
            "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
            "type": int(model.geom_type[geom_id]),
            "type_name": mujoco.mjtGeom(int(model.geom_type[geom_id])).name,
            **{name: getattr(model, "geom_"+name)[geom_id].tolist()
               for name in ("solref", "solimp", "friction")},
            **{name: int(getattr(model, "geom_"+name)[geom_id])
               for name in ("condim", "contype", "conaffinity", "priority")},
            **{name: float(getattr(model, "geom_"+name)[geom_id])
               for name in ("solmix", "margin", "gap")},
        })
    pairs = []
    for pair_id in range(model.npair):
        first, second = int(model.pair_geom1[pair_id]), int(model.pair_geom2[pair_id])
        pairs.append({
            "id": pair_id,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_PAIR, pair_id),
            "geom1_id": first, "geom2_id": second,
            "geom1_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, first),
            "geom2_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, second),
            "condim": int(model.pair_dim[pair_id]),
            **{name: getattr(model, "pair_"+name)[pair_id].tolist()
               for name in ("solref", "solreffriction", "solimp", "friction")},
            **{name: float(getattr(model, "pair_"+name)[pair_id])
               for name in ("margin", "gap")},
        })
    return {
        "mujoco_version": mujoco.__version__,
        "model_counts": {name: int(getattr(model, name))
                         for name in ("njnt", "nq", "nv", "nu", "ngeom", "npair")},
        "options": {
            "integrator": int(model.opt.integrator),
            "integrator_name": mujoco.mjtIntegrator(int(model.opt.integrator)).name,
            "solver": int(model.opt.solver),
            "solver_name": mujoco.mjtSolver(int(model.opt.solver)).name,
            "timestep": float(model.opt.timestep),
            "iterations": int(model.opt.iterations),
            "tolerance": float(model.opt.tolerance),
            "disableflags": int(model.opt.disableflags),
            "enableflags": int(model.opt.enableflags),
            "contact_override": {
                "enabled": bool(model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)),
                "solref": model.opt.o_solref.tolist(), "solimp": model.opt.o_solimp.tolist(),
                "friction": model.opt.o_friction.tolist(), "margin": float(model.opt.o_margin),
            },
        },
        "collision_geoms": geoms, "explicit_pairs": pairs,
        "collision_geom_selection": "contype != 0 or conaffinity != 0 or referenced by an explicit contact pair; body filters and runtime contact generation still apply.",
        "contact_parameter_scope": "Compiled nominal geom and explicit-pair parameters; actual contacts depend on parameter mixing, collision filtering and the optional global contact override. These are not measured penetration or per-contact solver results.",
    }


def legacy_argument_defaults():
    """Conservative reproduction label: every CLI default except instrumentation.

    Matching this label is not proof of a particular remote checkpoint or result.
    The defaults are checked against the actual parser by regression tests.
    """
    return {
        "research_method": "off", "research_shadow": False,
        "research_interface": "auxiliary_v1",
        "arm_protection_eligibility": "pickup",
        "placement_tail_seconds": 0., "research_budget_ledger": None,
        "pillar_xy": PILLAR_XY, "goal_reference": tuple(GOAL_REFERENCE),
        "contact_force_source": "joint_torque",
        **{"force_"+name: value for name, value in vars(ObserverConfig()).items()},
        "experiment_protocol": "legacy", "safety_margin_mode": "dynamic",
        "goal_decay_mode": "legacy", "geometry_impl": "current",
        "initial_state_protocol": "historical", "host": "localhost", "port": 8000,
        "num_episodes": 100, "case": 0, "fixed_eval": False, "eval_seed": None,
        "settle_before_start": False, "obstacle_source": "oracle", "perception_probe": False,
        "groundingdino_config": None, "groundingdino_checkpoint": None,
        "obstacle_text": "red pillar.", "device": "cuda", "box_threshold": 0.35,
        "text_threshold": 0.25, "control_mode": "both",
        **NOVELTY_DEFAULTS,
    }


def capture_physical_state(model, data):
    """Read live state without forward, integration, or observer mutation."""
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    values = {name: getattr(data, name).copy() for name in (
        "qpos", "qvel", "act", "ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied",
        "mocap_pos", "mocap_quat", "userdata", "eq_active", "plugin_state",
        "qacc", "act_dot", "qfrc_constraint", "solver_niter")}
    values.update(time_s=np.array(data.time), integration_state=state,
                  integration_state_spec=np.array(spec))
    return values


def save_initial_state(path, model, data, phase, config):
    """Always record both initialization boundaries, independently of diagnostics."""
    values = capture_physical_state(model, data)
    state_hash = hashlib.sha256()
    for key, value in sorted(values.items()):
        state_hash.update(key.encode())
        state_hash.update(str(value.dtype).encode())
        state_hash.update(str(value.shape).encode())
        state_hash.update(value.tobytes())
    metadata = {
        "schema_version": 1, "phase": phase, "mujoco_version": mujoco.__version__,
        **{key: config[key] for key in ("scene_id", "pillar_xy", "goal_reference", "reference_id", "source_sha256")},
        "experiment_protocol": config["arguments"]["experiment_protocol"],
        "geometry_impl": config["arguments"]["geometry_impl"],
        "initial_state_protocol": config["arguments"]["initial_state_protocol"],
        "research_interface": config["arguments"].get("research_interface", "auxiliary_v1"),
        "arm_protection_eligibility": config["arguments"].get("arm_protection_eligibility", "pickup"),
        "effective_settle_before_start": config["effective_settle_before_start"],
        "run_reference_pillar_xy_m": list(config["pillar_xy"]),
        "pillar_xy_scope": "Requested run reference only; per-episode placement and dynamics can differ. actual_body_poses_world records the current data poses.",
        "actual_body_poses_world": {
            name: {"pos": data.xpos[body_id].tolist(), "quat": data.xquat[body_id].tolist()}
            for name in ("real_screwdriver", "dynamic_pillar")
            if (body_id := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)) >= 0
        },
        "state_sha256": state_hash.hexdigest(),
        "integration_state_note": "Official mjSTATE_INTEGRATION; derived diagnostic arrays are also saved, and may be nonzero after scene mj_forward.",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values,
                            metadata_json=np.array(json.dumps(metadata, ensure_ascii=False, allow_nan=False)))


def run_configuration(args, case, model, root):
    root = Path(root)
    sources = ("scripts/deploy/deploy_screwdriver_client.py", "scripts/deploy/pace_diagnostics.py",
               "scripts/deploy/joint_torque_observer.py", "scripts/deploy/groundingdino_obstacle.py",
               "scripts/deploy/pace_research.py", "scripts/deploy/pace_novelty.py", "scripts/deploy/pregrasp_guard.py",
               "models/dummyx_apf_scene.xml", "models/dummy_apf.xml")
    defaults = legacy_argument_defaults()
    checks = {name: (tuple(getattr(args, name)) if name in ("pillar_xy", "goal_reference") else getattr(args, name, object())) == value for name, value in defaults.items()}
    scene = scene_configuration(args)
    unknown = sorted(set(vars(args))-set(defaults)-{"diagnostics"})
    full_reset = args.initial_state_protocol == "full_v1"
    return {
        "schema_version": 1, **scene, "arguments": vars(args).copy(), "case": case.copy(),
        "legacy_reproduction_settings": all(checks.values()) and not unknown,
        "legacy_reproduction_checks": checks, "unclassified_arguments": unknown,
        "legacy_reproduction_scope": "All CLI defaults except diagnostics; includes workload/initialization/feedback choices. Does not verify remote policy identity.",
        "physics_period_s": float(model.opt.timestep),
        "physics_provenance": physics_provenance(model),
        "contact_evaluation_scope": {
            "legacy_load_metric": {
                "reader": "deploy_screwdriver_client.get_target_table_force",
                "contact_point_world_z_m": {"lower": 0.18, "upper": 0.22, "bounds": "strict"},
                "selected_body_names": ["real_screwdriver", "link7", "link8", "link9"],
                "selection": "Include each contact if either geom's direct body ID is one of the selected bodies and 0.18 < contact.pos[2] < 0.22; no restriction on the other body.",
                "world_force": "For each selected contact, rotate c_force[:3] with contact.frame.reshape(3, 3).T; if world fz < 0 negate the whole vector; then sum the vectors.",
                "impulse": "At each pre-step sample, take the norm of that vector sum and accumulate norm * model.opt.timestep over the episode (N s).",
                "scope_note": "This height-band/body set is not restricted to table contact. Tool-box-bottom and other contacts can contribute when the same selection holds; sign folding is not a net physical force on a fixed body.",
            },
            "robot_contact_truth": {
                "reader": "joint_torque_observer.robot_contact_truth",
                "selection": "Exactly one contact body is in arm_body_ids(model), the descendants of link1_1 including that body; exclude internal robot contacts. No height-band filter.",
                "world_force": "Rotate each contact force to world coordinates, use the geom2/geom1 sign to obtain force on the robot, and sum without fz sign folding.",
                "scope_note": "Independent observer-check truth set; loose-tool contacts without a robot-tree body are excluded. It differs from the historical load metric.",
            },
        },
        "geometry_period_s": float(50 * model.opt.timestep), "physics_substeps_per_action": 50,
        "inference_action_stride": 8, "goal_reference_world_m": scene["goal_reference"],
        "policy_interface": {"prompt": "Pick up the screwdriver and drop it into the box.",
                             "image_cameras": ["rear_cam", "wrist_cam"], "image_size_hw": [256, 256],
                             "checkpoint_training_scope": "Server metadata is saved by the caller; demonstration count, training completion and obstacle exposure are not inferred from source."},
        "effective_settle_before_start": bool(case["obs"] and
                                              (args.settle_before_start or args.obstacle_source == "groundingdino")),
        "reset_scope": {"apf_history": args.experiment_protocol == "controlled",
                        "contact_states": True, "observer_state": True,
                        "free_tool_velocity": full_reset, "mj_resetData": full_reset,
                        "simulation_time": full_reset, "actuator_activation": full_reset,
                        "solver_warmstart": full_reset, "applied_forces": full_reset},
        "initial_state_snapshots": {
            "after_reset": "initial_states/ep_NNN_after_reset.npz",
            "pre_action": "initial_states/ep_NNN_pre_action.npz",
            "note": "Saved even with detailed diagnostics disabled; pre_action follows optional settling, perception and observer reset.",
        },
        "drop_box_body_reference_world_m": [-0.05, 0.0, 0.2],
        "goal_minus_drop_box_body_m": np.asarray(scene["goal_reference"]) - np.array([-0.05, 0.0, 0.2]),
        "controller_capsule_radii_m": {"oracle": 0.025, "groundingdino": float(np.sqrt(2)*0.02+0.01)},
        "evaluation_oracle_enclosing_radius_m": float(np.sqrt(2)*0.02+0.01),
        "geometry_envelopes_harmonized": False,
        "fixed_rules": {"lpf_beta": 0.6, "joint_delta_limit_rad": 0.25,
                        "pickup_height_m": 0.235, "tcp_contact_gate_m": 0.235,
                        "direction_weights": [0.1, 0.1, 1.0], "dls_rho": 0.05},
        "contact_revision_rules": {
            "mapping": args.contact_mapping, "force_gate": args.contact_gate,
            "feedback": args.contact_feedback, "support_mode": args.support_mode,
            "confirmation_steps": args.support_confirmation_steps,
            "confirmation_window_s": args.support_confirmation_steps * float(model.opt.timestep),
            "reference_scope": "Effective TCP height, not physical surface reconstruction; first touch is unprotected by an inferred height; a confirmed reference stays fixed for the episode.",
            "trace_scope": "contact_revision_* are pre-step Cartesian increments in metres; mapped includes DLS attenuation. Trace collection is included in measured contact time; support confirmation computation is also timed.",
            "historical_fixed_rules_scope": "fixed_rules records historical constants; joint_filter and contact_gate arguments select the effective variant."},
        "units": {"arm_target": "rad", "gripper_target": "m", "force": "N",
                  "tau_external": "N m", "cartesian_correction": "m (not m/s)"},
        "statistics_unit": "episode; 1 ms samples are dependent",
        "latency_note": "Diagnostics add wall-clock cost; do not compare these runs as uninstrumented real-time performance.",
        "research_latency_scope": "post_contact_v2 includes guard, its inline research logging and common command checks in safety_layer_time_ms; guard compute_ms separately excludes its JSON emission. auxiliary_v1 retains historical latency scope.",
        "arm_qualification_scope": "pickup: original height gate; latched_v1: per-episode height-ever latch; independent_v1: distal arm risk independent of tool height. None certifies rigid grasp or maps a free tool to arm joints.",
        "source_sha256": {rel: hashlib.sha256((root/rel).read_bytes()).hexdigest() for rel in sources},
    }


def experiment_suffix(args):
    scene = scene_configuration(args)
    coordinates = json.dumps([scene["pillar_xy"], scene["goal_reference"]], separators=(",", ":"))
    tag = hashlib.sha256(coordinates.encode()).hexdigest()[:16]
    revision = (f"_geometry-{args.geometry_impl}_initial-{args.initial_state_protocol}"
                f"_scene-{scene['scene_id']}_reference-{scene['reference_id']}_coords-{tag}")
    if getattr(args, 'research_method', 'off') != 'off' or getattr(args, 'placement_tail_seconds', 0) > 0:
        mode = getattr(args, 'research_method', 'off')
        revision += f"_r-{mode}{'-sh' if getattr(args, 'research_shadow', False) else ''}_tail-{getattr(args, 'placement_tail_seconds', 0):g}"
    if getattr(args, 'research_interface', 'auxiliary_v1') == 'post_contact_v2':
        revision += '_ri-pc2'
    qualification = getattr(args, 'arm_protection_eligibility', 'pickup')
    if qualification != 'pickup':
        revision += '_ae-'+{'latched_v1': 'l1', 'independent_v1': 'i1'}[qualification]
    revision += novelty_suffix(args)
    if (args.experiment_protocol == "legacy" and args.safety_margin_mode == "dynamic"
            and args.goal_decay_mode == "legacy"):
        return revision
    return (f"_{args.experiment_protocol}_margin-{args.safety_margin_mode}"
            f"_goal-{args.goal_decay_mode}"+revision)


def private_kinematics(model, source, target):
    """Synchronize position data without mj_forward on the live observer data."""
    for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "userdata"):
        getattr(target, name)[:] = getattr(source, name)
    target.time = source.time
    mujoco.mj_forward(model, target)


def active_capsules(model, data, gripper_command):
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6_1")
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "real_screwdriver")
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    tcp_pos, tool_pos = data.site_xpos[tcp], data.xpos[tool]
    caps = [{"p1": data.xpos[wrist].copy(), "p2": tcp_pos.copy(), "r": 0.07,
             "body_id": palm, "kind": "distal", "arm_map_available": True}]
    if gripper_command < 0.02 and np.linalg.norm(tcp_pos-tool_pos) < 0.05:
        direction = data.xmat[tool].reshape(3, 3)[:, 1]
        caps.append({"p1": tool_pos-direction*0.12, "p2": tool_pos+direction*0.12,
                     "r": 0.02, "body_id": tool, "kind": "tool_free_body",
                     "arm_map_available": False})
    return caps, bool(tool_pos[2] > 0.235), tcp_pos.copy()


def oracle_capsule(model, data, enclosing=False):
    pillar = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_pillar")
    axis = data.xmat[pillar].reshape(3, 3)[:, 2]*0.08
    center = data.xpos[pillar]
    return {"p1": center-axis, "p2": center+axis,
            "r": float(np.sqrt(2)*0.02+0.01) if enclosing else 0.025}


def nearest_geometry(model, data, capsules, obstacle, distance_fn):
    result = {"distance": np.nan, "normal": np.full(3, np.nan),
              "point_robot": np.full(3, np.nan), "point_obstacle": np.full(3, np.nan),
              "jacobian_full": np.full((3, model.nv), np.nan),
              "arm_map_available": False, "normal_valid": False,
              "kind": "unavailable", "body_id": -1, "jacobian_status": "no_obstacle",
              "obstacle_axis": np.full((2, 3), np.nan), "obstacle_radius": np.nan}
    if obstacle is None:
        return result
    result.update(obstacle_axis=np.array([obstacle["p1"], obstacle["p2"]]),
                  obstacle_radius=float(obstacle["r"]))
    for cap in capsules:
        distance, normal, near, other = distance_fn(cap["p1"], cap["p2"], obstacle["p1"], obstacle["p2"])
        surface = distance-cap["r"]-obstacle["r"]
        if np.isnan(result["distance"]) or surface < result["distance"]:
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, near, cap["body_id"])
            valid_normal = distance > 1e-6
            available = cap["arm_map_available"] and valid_normal
            status = ("distal_point_exact_local" if available else
                      "degenerate_normal" if not valid_normal else
                      "tool_free_body_arm_map_unavailable")
            result.update(distance=float(surface), normal=normal.copy(), point_robot=near.copy(),
                          point_obstacle=other.copy(), jacobian_full=jac,
                          arm_map_available=bool(available), normal_valid=bool(valid_normal),
                          kind=cap["kind"], body_id=cap["body_id"], jacobian_status=status)
    return result


def geometry_snapshot(model, data, gripper, use_obstacle, source, perceived,
                      step, margin_mode, decay_mode, distance_fn, goal_reference=GOAL_REFERENCE):
    caps, pickup, tcp = active_capsules(model, data, gripper)
    # Only the explicitly selected controller geometry enters this branch.
    obstacle = None
    if use_obstacle:
        if source == "groundingdino":
            obstacle = perceived
        elif source == "oracle":
            obstacle = oracle_capsule(model, data)
        else:
            raise ValueError(source)
    control = nearest_geometry(model, data, caps, obstacle, distance_fn)
    arm_control = (control if control["kind"] == "distal" else
                   nearest_geometry(model, data, [c for c in caps if c["kind"] == "distal"],
                                    obstacle, distance_fn))
    # Evaluation is deliberately separate; never returned to the controller.
    evaluation = nearest_geometry(model, data, caps,
                                  oracle_capsule(model, data, enclosing=True) if use_obstacle else None,
                                  distance_fn)
    goal_distance = float(np.linalg.norm(tcp-np.asarray(goal_reference)))
    margin = safety_margin(goal_distance, margin_mode)
    goal = goal_factor(goal_distance, decay_mode)
    fade = np.clip(step/15.0, 0.0, 1.0)
    risk = bool(np.isfinite(control["distance"]) and control["distance"] < margin)
    proximity = ((margin-max(control["distance"], 0.0))/margin)**2 if risk else 0.0
    influence = proximity*goal*fade if risk and pickup else 0.0
    return {"control": control, "arm_control": arm_control, "evaluation": evaluation, "capsules": caps,
            "pickup_gate": pickup, "goal_distance": goal_distance, "safe_margin": margin,
            "goal_factor": goal, "start_factor": fade, "proximity_factor": proximity,
            "influence": influence, "risk_geometric": risk,
            "risk_eligible": risk and pickup,
            "goal_suppressed": bool(risk and pickup and influence <= 0.005
                                    and proximity*fade > 0.005),
            "would_activate": bool(risk and pickup and influence > 0.005)}


def linearized_metrics(geometry, base_delta, contact_delta):
    """Frozen-obstacle, local point linearization. Not a feasibility guarantee."""
    if not geometry["arm_map_available"]:
        return {key: np.nan for key in ("chi", "clearance_base", "clearance_final", "clearance_change")}
    row = geometry["normal"] @ geometry["jacobian_full"][:, :6]
    chi = float(row @ contact_delta)
    base = float(geometry["distance"] + row @ base_delta)
    return {"chi": chi, "clearance_base": base, "clearance_final": base+chi, "clearance_change": chi}


def placement_quality(model, data):
    """Evaluation only: collision mesh vertices in box-local XY, not stable placement."""
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "real_screwdriver")
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drop_box")
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    points = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        mesh = model.geom_dataid[geom]
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Tool containment requires explicit support for non-mesh collision geometry")
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start+count]
        points.append(vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
    if not points:
        raise ValueError("No tool collision mesh")
    world = np.concatenate(points)
    local = (world-data.xpos[box]) @ data.xmat[box].reshape(3, 3)
    # Unmodified box model: inner half-opening is 0.11 m in x and y.
    overhang = np.maximum(np.maximum(local[:, :2].max(0)-.11, -.11-local[:, :2].min(0)), 0)
    return {"tcp_position_world_m": data.site_xpos[tcp].copy(),
            "tcp_rotation_world": data.site_xmat[tcp].reshape(3, 3).copy(),
            "tool_position_world_m": data.xpos[body].copy(), "tool_quat_wxyz": data.xquat[body].copy(),
            "tool_collision_aabb_world_m": np.array([world.min(0), world.max(0)]),
            "inside_box_opening_xy": bool(np.all(overhang <= 0)),
            "overhang_xy_m": overhang, "has_overhang": bool(np.any(overhang > 0)),
            "semantics": "XY containment only; no stable placement or task success claim"}


class EpisodeDiagnostics:
    """Separate, exclusive, chunked NPZ + JSON files; episode is the sampling unit."""
    def __init__(self, directory, model, config, distance_fn, robot_bodies, chunk_size=1000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.model, self.config, self.distance_fn = model, config, distance_fn
        self.robot_bodies = set(robot_bodies)
        self.scratch = mujoco.MjData(model)
        self.chunk_size, self.buffer, self.chunks = chunk_size, [], 0
        self.samples, self.counts = 0, {key: 0 for key in (
            "risk", "height_gate", "observer_contact", "contact_nonzero", "risk_and_contact",
            "held_geo_and_contact", "chi_valid_risk_contact", "toward_obstacle", "goal_suppressed",
            "physical_pillar_contact", "physical_tool_box_contact")}
        self.release_evaluations = []
        self.last_gripper_closed = False
        self.terminal_placement = None
        self.closed = False
        self.held_geo_active = False
        self.held_geo_nonzero = False
        self.action = None
        self.actions = None
        self.events = None
        write_json_exclusive(self.directory/"metadata.json", {
            "schema_version": 1, "configuration": config,
            "status": "opened; episode_summary.json records final completion or error",
            "physics_sampling_period_s": float(model.opt.timestep),
            "definitions": {
                "control_*": "Selected controller capsule, recomputed on private synchronized current qpos; raw clearance before pickup gate.",
                "eval_*": "Evaluation-only oracle capsule with enclosing radius; never supplied to control.",
                "decision_*": "Held 20 Hz decision and immediate correction actually computed by controller.",
                "chi": "n.T @ J_near[:, :6] @ actual_contact_joint_correction [m]; negative means toward obstacle.",
                "jacobian": "mj_jac at capsule nearest point attached to the actual body; tool free-body arm map is unavailable (NaN chi).",
                "linearized_clearance": "Frozen current normal/nearest point and obstacle, first-order command-position estimate, not a dynamics rollout.",
                "placement_quality": "Evaluation-only collision-mesh XY containment in the unmodified box opening; not stable placement or episode_success. Closed-to-open command snapshots and terminal state are stored in episode_summary.json.",
                "physical_contacts": "Live MuJoCo solver contacts after mj_step; dist<=0 and positive normal force; events timestamp is pre-integration time.",
                "post_clearance": "Fresh private mj_forward after mj_step and observer sample; measured kinematic clearance, not caused solely by contact correction.",
                "force_input": "Previous observer sample used by controller; first sample unavailable and zero. force_post is the next-step estimate.",
                "actual_command": "Full 8D value written to data.ctrl before mj_step; not a post-actuator-clamp target.",
                "nonzero": f"norm > {ZERO_TOL}; diagnostics only", "statistics_unit": "episode",
            },
        })
        self.actions = (self.directory/"actions.jsonl").open("x", encoding="utf-8")
        self.events = (self.directory/"physical_contacts.jsonl").open("x", encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close(complete=kind is None, error=None if kind is None else str(value))

    def snapshot(self, data, step):
        private_kinematics(self.model, data, self.scratch)
        a, cfg = self.config["arguments"], self.config["case"]
        return geometry_snapshot(self.model, self.scratch, self.action["base_target"][6],
                                 cfg["obs"], a["obstacle_source"], self.perceived, step,
                                 a["safety_margin_mode"], a["goal_decay_mode"], self.distance_fn,
                                 self.config["goal_reference"])

    def record_action(self, data, action, base_target, decision, step, perceived):
        self.perceived = perceived
        self.held_geo_active = bool(decision.get("geometry_decision_active", False))
        self.held_geo_nonzero = bool(np.linalg.norm(decision.get("delta_q_apf", np.zeros(6))) > ZERO_TOL)
        self.action = {"action_index": step, "time_s": data.time,
                       "nominal_target": action.copy(), "q_current": data.qpos[:8].copy(),
                       "base_target": base_target.copy(),
                       "effective_geo_target_change": base_target[:6]-action[:6],
                       "decision": decision.copy()}
        snap = self.snapshot(data, step)
        self.action["geometry"] = {key: value for key, value in snap.items() if key != "capsules"}
        closed = bool(action[6] <= .02)
        if self.last_gripper_closed and not closed:
            event = {"action_index": step, "time_s": float(data.time),
                     "event": "closed_to_open_command; not proof of physical detachment",
                     "placement": placement_quality(self.model, self.scratch)}
            self.release_evaluations.append(event)
            self.action["release_evaluation"] = event
        self.last_gripper_closed = closed
        self.actions.write(json.dumps(json_safe(self.action), ensure_ascii=False, allow_nan=False)+"\n")
        self.actions.flush()

    def before_step(self, data, estimate, force_input, contact, base_target, final_target, step, substep, contact_target=None, research_state=None):
        snap = self.snapshot(data, step)
        contact_reference = final_target if contact_target is None else contact_target
        correction = contact_reference[:6]-base_target[:6]
        research_delta = final_target[:6]-contact_reference[:6]
        base_delta = base_target[:6]-data.qpos[:6]
        row = {"time_pre_s": float(data.time), "action_index": step, "substep_index": substep,
               "qpos_pre": data.qpos.copy(), "qvel_pre": data.qvel.copy(),
               "nominal_target": self.action["nominal_target"].copy(),
               "geo_target": base_target.copy(), "actual_command": final_target.copy(),
               "geo_immediate_delta": np.asarray(self.action["decision"].get("delta_q_apf", np.zeros(6))),
               "decision_geo_active": self.held_geo_active,
               "decision_geo_nonzero": self.held_geo_nonzero,
               "contact_joint_delta": correction, "research_joint_delta": research_delta,
               "research_nonzero": bool(np.linalg.norm(research_delta)>ZERO_TOL),
               "force_input": force_input.copy(),
               "tau_external_input": estimate.tau_external.copy(), "force_estimated_input": estimate.force.copy(),
               "observer_contact_input": bool(estimate.contact),
               "observer_input_time_s": float(data.time-self.model.opt.timestep) if self.samples else np.nan,
               "observer_input_valid": bool(self.samples),
               "tcp_height_actual": float(self.scratch.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")][2]),
               "contact_mode": self.config["case"]["mode"],
               "geometry_enabled": self.config["case"]["apf"],
               "contact_enabled": self.config["case"]["contact"],
               "tcp_height_gate": bool(contact["height_gate"]),
               "contact_nonzero": bool(np.linalg.norm(correction) > ZERO_TOL),
               "pickup_gate": snap["pickup_gate"], "risk_geometric": snap["risk_geometric"],
               "risk_eligible": snap["risk_eligible"], "goal_suppressed": snap["goal_suppressed"],
               "goal_distance": snap["goal_distance"], "safe_margin": snap["safe_margin"],
               "weight_proximity": snap["proximity_factor"], "weight_goal": snap["goal_factor"],
               "weight_start": snap["start_factor"], "weight_total": snap["influence"]}
        arm = snap['arm_control']
        state = research_state or {}
        row.update(arm_control_distance=arm['distance'], arm_control_normal=arm['normal'],
                   arm_control_jacobian_full=arm['jacobian_full'],
                   arm_control_map_available=arm['arm_map_available'],
                   arm_risk_geometric=bool(np.isfinite(arm['distance']) and arm['distance'] < snap['safe_margin']),
                   research_arm_gate_allowed=bool(state.get('arm_gate_allowed', False)),
                   research_arm_constraint_eligible=bool(state.get('arm_constraint_eligible', False)),
                   research_pickup_height_seen=bool(state.get('pickup_height_seen', False)),
                   research_eligibility_policy=state.get('policy', 'guard_disabled'))
        for key, value in contact.items():
            row["contact_"+key] = value
        for prefix, geometry in (("control", snap["control"]), ("eval", snap["evaluation"])):
            row.update({prefix+"_"+key: value for key, value in geometry.items()})
            contact_metrics = linearized_metrics(geometry, base_delta, correction)
            final_metrics = linearized_metrics(geometry, base_delta, correction+research_delta)
            final_metrics['chi'] = contact_metrics['chi']
            row.update({prefix+"_"+key: value for key, value in final_metrics.items()})
            row[prefix+'_research_normal_delta'] = linearized_metrics(geometry, base_delta, research_delta)['chi']
        row['research_and_contact'] = row['research_nonzero'] and row['contact_nonzero']
        row["risk_and_contact"] = row["risk_geometric"] and row["contact_nonzero"]
        row["eligible_risk_and_contact"] = row["risk_eligible"] and row["contact_nonzero"]
        row["held_geo_and_contact"] = self.held_geo_nonzero and row["contact_nonzero"]
        row["conflict_valid"] = row["risk_and_contact"] and snap["control"]["arm_map_available"]
        row["toward_obstacle"] = bool(row["conflict_valid"] and row["control_chi"] < -ZERO_TOL)
        return row

    def after_step(self, data, row, estimate):
        # Called only after sample_after_step: no live mj_forward is permitted.
        snap = self.snapshot(data, row["action_index"])
        row.update(time_post_s=float(data.time), qpos_post=data.qpos.copy(), qvel_post=data.qvel.copy(),
                   tau_external_post=estimate.tau_external.copy(), force_post=estimate.force.copy(),
                   observer_contact_post=bool(estimate.contact))
        for prefix, geometry in (("control", snap["control"]), ("eval", snap["evaluation"])):
            row[prefix+"_distance_post"] = geometry["distance"]
            row[prefix+"_distance_actual_change"] = geometry["distance"]-row[prefix+"_distance"]
            row[prefix+"_kind_post"] = geometry["kind"]
        pillar = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_pillar")
        tool = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "real_screwdriver")
        events, pillar_contact = [], False
        box = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drop_box")
        tool_box_contact = False
        wrench = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])
            if b1 in self.robot_bodies and b2 in self.robot_bodies:
                continue
            if not ({b1, b2} & (self.robot_bodies | {tool})):
                continue
            mujoco.mj_contactForce(self.model, data, i, wrench)
            if c.dist <= 0 and wrench[0] > ZERO_TOL:
                relevant_pillar = pillar in (b1, b2) and bool({b1, b2} & (self.robot_bodies | {tool}))
                pillar_contact |= relevant_pillar
                tool_box_contact |= set((b1, b2)) == {tool, box}
                events.append({"geom1": int(c.geom1), "geom2": int(c.geom2), "body1": b1, "body2": b2,
                               "distance_m": float(c.dist), "normal_force_N": float(wrench[0]),
                               "position_world_m": c.pos.copy(), "pillar_robot_or_tool": relevant_pillar,
                               "tool_box_contact": set((b1, b2)) == {tool, box}})
        if events:
            self.events.write(json.dumps(json_safe({"time_s": row["time_pre_s"], "contacts": events}), allow_nan=False)+"\n")
        row["physical_tool_box_contact"] = tool_box_contact
        row["physical_pillar_contact"] = pillar_contact
        row["physical_relevant_contact_count"] = len(events)
        row["pillar_upright_cosine"] = float(self.scratch.xmat[pillar].reshape(3, 3)[2, 2])
        row["pillar_position"] = self.scratch.xpos[pillar].copy()
        self.samples += 1
        keys = {"risk": "risk_geometric", "height_gate": "tcp_height_gate", "observer_contact": "observer_contact_input",
                "contact_nonzero": "contact_nonzero", "risk_and_contact": "risk_and_contact",
                "held_geo_and_contact": "held_geo_and_contact", "chi_valid_risk_contact": "conflict_valid",
                "toward_obstacle": "toward_obstacle", "goal_suppressed": "goal_suppressed",
                "physical_pillar_contact": "physical_pillar_contact",
                "physical_tool_box_contact": "physical_tool_box_contact"}
        for key, field in keys.items():
            self.counts[key] += int(row[field])
        self.buffer.append(row)
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        keys = self.buffer[0].keys()
        assert all(row.keys() == keys for row in self.buffer)
        arrays = {key: np.asarray([row[key] for row in self.buffer]) for key in keys}
        if any(value.dtype.hasobject for value in arrays.values()):
            raise ValueError("Diagnostic NPZ must not require pickle")
        with (self.directory/f"physics_{self.chunks:05d}.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        self.buffer.clear()
        self.chunks += 1

    def close(self, complete=True, error=None):
        if self.closed:
            return
        try:
            self.flush()
            if self.samples:
                self.terminal_placement = placement_quality(self.model, self.scratch)
            valid = self.counts["chi_valid_risk_contact"]
            write_json_exclusive(self.directory/"episode_summary.json", {
                "complete": complete, "error": error, "sample_count": self.samples,
                "release_evaluations": self.release_evaluations,
                "terminal_placement": self.terminal_placement,
                "placement_scope": "Diagnostics only; command opening is not physical release; original episode_success unchanged.",
                "completion_semantics": "Clean logger closure, not task success; task outcome remains in the original episode NPZ.",
                "duration_s": self.samples*self.model.opt.timestep, "chunk_count": self.chunks,
                "sample_counts": self.counts,
                "time_fractions": {k: v/self.samples if self.samples else None for k, v in self.counts.items()},
                "toward_fraction_given_valid_risk_contact": self.counts["toward_obstacle"]/valid if valid else None,
                "conflict_coverage_given_risk_contact": valid/self.counts["risk_and_contact"] if self.counts["risk_and_contact"] else None,
                "note": "No empirical safety/innovation claim; aggregate independent episodes, not substeps.",
            })
        finally:
            for stream in (self.actions, self.events):
                if stream is not None:
                    stream.close()
            self.scratch = None
            self.closed = True
