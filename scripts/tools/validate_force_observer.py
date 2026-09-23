"""Headless down/hold/release validation; never imports OpenPI or launches Viewer."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "deploy"))
from joint_torque_observer import (MujocoTorqueObserver, ForceLog, add_observer_arguments,
                                   config_from_args, robot_contact_truth, arm_body_ids)


def run(args):
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models/dummyx_apf_scene.xml"))
    data = mujoco.MjData(model)
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    data.qpos[:6] = [0, 0.4, 0.5, 0, 1.57, 0]
    # Keep loose props away from the downward calibration trajectory.
    for name in ("fj_screwdriver", "pillar_joint"):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = model.jnt_qposadr[joint]
        data.qpos[adr:adr + 3] = [2, 2, 0.3]
    data.ctrl[:6] = data.qpos[:6]
    data.ctrl[6:8] = 0.04
    mujoco.mj_forward(model, data)
    config = config_from_args(args)
    observer = MujocoTorqueObserver(model, tcp, config)
    observer.reset(data)
    bodies = arm_body_ids(model)
    log = ForceLog()
    heights = []
    target = data.qpos[:6].copy()
    J = np.zeros((3, model.nv))
    dt = model.opt.timestep
    for step in range(round(args.duration / dt)):
        t = step * dt
        mujoco.mj_jacSite(model, data, J, None, tcp)
        # Settle, downstroke, hold, then retract. No truth-based trajectory switching.
        speed = 0 if t < 1 else (-0.06 if t < 7 else (0 if t < 8 else 0.04))
        target += J[:, :6].T @ np.linalg.solve(
            J[:, :6] @ J[:, :6].T + 0.05**2 * np.eye(3), [0, 0, speed]) * dt
        data.ctrl[:6] = target
        mujoco.mj_step(model, data)
        estimate = observer.sample_after_step(data)
        truth = robot_contact_truth(model, data, bodies)
        log.append(t, estimate, truth, np.zeros(3), config.enter_force)
        heights.append(data.site_xpos[tcp, 2])
    arrays = log.arrays()
    est, truth = arrays["f_estimated"], arrays["f_contact_truth"]
    actual, detected = arrays["truth_contact"], arrays["estimated_contact"]
    indices = np.flatnonzero(actual)
    first = int(indices[0]) if len(indices) else len(actual)
    before = np.arange(len(actual)) < first
    hits = np.flatnonzero(detected)
    metrics = {
        "samples": len(actual), "dt_s": dt,
        "precontact_raw_bias_xyz_N": arrays["f_raw"][before].mean(axis=0).tolist() if before.any() else None,
        "precontact_raw_rms_norm_N": float(np.sqrt(np.mean(np.sum(arrays["f_raw"][before]**2, axis=1)))) if before.any() else None,
        "precontact_bias_xyz_N": est[before].mean(axis=0).tolist() if before.any() else None,
        "precontact_rms_norm_N": float(np.sqrt(np.mean(np.sum(est[before]**2, axis=1)))) if before.any() else None,
        "precontact_false_positive_samples": int(np.sum(detected & before)),
        "noncontact_false_positive_samples": int(np.sum(detected & ~actual)),
        "noncontact_samples": int(np.sum(~actual)),
        "contact_samples": int(np.sum(actual)),
        "contact_mean_estimated_z_N": float(est[actual, 2].mean()) if actual.any() else None,
        "contact_mean_truth_z_N": float(truth[actual, 2].mean()) if actual.any() else None,
        "contact_z_rmse_N": float(np.sqrt(np.mean((est[actual, 2] - truth[actual, 2])**2))) if actual.any() else None,
        "contact_z_correlation": float(np.corrcoef(est[actual, 2], truth[actual, 2])[0, 1]) if actual.sum() > 1 else None,
        "all_xyz_rmse_N": float(np.sqrt(np.mean((est - truth)**2))),
        "detection_delay_s": float((hits[0] - first) * dt) if len(hits) else None,
        "release_delay_s": float((np.flatnonzero(detected)[-1] - indices[-1]) * dt) if len(indices) and len(hits) else None,
        "normal_positive_fraction": float(np.mean(est[actual, 2] > 0)) if actual.any() else None,
        "contact_missed_samples": int(np.sum(actual & ~detected)),
        "min_tcp_height_m": float(min(heights)),
        "config": vars(config),
    }
    (ROOT / "outputs").mkdir(exist_ok=True)
    output = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix="force_observer_", dir=ROOT / "outputs"))
    output.mkdir(parents=True, exist_ok=True)
    # Refuse overwriting even validation output.
    with (output / "samples.npz").open("xb") as handle:
        np.savez(handle, **arrays, tcp_height=np.asarray(heights))
    with (output / "metrics.json").open("x") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts: {output}")
    if not actual.any():
        raise RuntimeError("No contact in validation trajectory")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_observer_arguments(parser)
    parser.add_argument("--duration", type=float, default=11)
    parser.add_argument("--output", help="New output directory; existing files are never overwritten")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
