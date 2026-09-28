"""Validate recorded images/state/action alignment and optionally replay actions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/collect"))
from collect_hardstop_screwdriver import FPS, HardstopArm, SCENE, STATE_NAMES

LEGACY_SCENE = SCENE.parent / "scene_pre_d415.xml"


def check_episode(folder, arm=None):
    meta = json.loads((folder / "metadata.json").read_text())
    if meta["schema"] not in ("agv_dummyx_screwdriver_v2", "agv_dummyx_screwdriver_v3") or not meta["success"] or not meta["images_saved"]:
        raise ValueError(f"Not a successful image dataset: {folder}")
    if meta["schema"] == "agv_dummyx_screwdriver_v3":
        if (meta["fixed_camera"]["camera_name"] != "overview" or
                meta["fixed_camera"]["raw_resolution_wh"] != [640, 360] or
                meta["fixed_camera"]["saved_resolution_wh"] != [640, 360]):
            raise ValueError("D415 RGB camera convention mismatch")
    if meta["state_names"] != STATE_NAMES or meta["fps"] != FPS:
        raise ValueError("State convention or frequency mismatch")
    with np.load(folder / "joint_data.npz", allow_pickle=False) as data:
        n = meta["frames"]
        for name in ("qpos", "actions", "actions_exact"):
            if data[name].shape != (n, 7) or not np.all(np.isfinite(data[name])):
                raise ValueError(f"Invalid {name} array")
        np.testing.assert_array_equal(data["actions"], data["actions_exact"].astype(np.float32))
        np.testing.assert_allclose(np.diff(data["timestamps"]), 1 / FPS, atol=1e-10)
        if len(data["phase"]) != n or n == 0:
            raise ValueError("Missing phase/frame data")
        if not (folder / "instruction.txt").read_text().strip():
            raise ValueError("Missing language instruction")
        sizes = {"cam_fixed": (360, 640, 3) if meta["schema"] == "agv_dummyx_screwdriver_v3" else (256, 256, 3),
                 "cam_wrist": (256, 256, 3)}
        for camera in ("cam_fixed", "cam_wrist"):
            paths = sorted((folder / camera).glob("*.jpg"))
            if [p.name for p in paths] != [f"{i:05d}.jpg" for i in range(n)]:
                raise ValueError(f"Missing/extra {camera} frames")
            for path in paths:
                image = cv2.imread(str(path))
                if image is None or image.shape != sizes[camera]:
                    raise ValueError(f"Invalid image: {path}")
        result = {"episode": folder.name, "frames": n, "images": 2 * n,
                  "duration_s": n / FPS, "arrays_and_images_valid": True}
        if arm is not None:
            for name, digest in meta["model_files_sha256"].items():
                file = ROOT / name
                if name == "models/hardstop_arm/part_34.msh":
                    file = SCENE.parent / "part_34.msh"
                if name == "models/agv_dummyx/scene.xml" and meta["schema"] == "agv_dummyx_screwdriver_v2":
                    file = LEGACY_SCENE
                content = file.read_bytes()
                if name == "models/agv_dummyx/scene.xml" and digest != hashlib.sha256(content).hexdigest():
                    # Older recordings used the same mesh through its former path.
                    content = content.replace(b'file="part_34.msh"',
                                              b'file="../hardstop_arm/part_34.msh"').rstrip(b"\n")
                if digest != hashlib.sha256(content).hexdigest():
                    raise ValueError(f"Model changed since recording: {file}")
            arm.reset(meta["initial_xy"], meta["initial_yaw"])
            max_q_error, max_object_error = 0., 0.
            for i, action in enumerate(data["actions_exact"]):
                max_q_error = max(max_q_error, float(np.max(np.abs(arm.data.qpos - data["sim_qpos"][i]))))
                max_object_error = max(max_object_error, float(np.linalg.norm(
                    arm.data.body("real_screwdriver").xpos - data["object_xyz"][i])))
                arm.step(action)
            np.testing.assert_allclose(arm.data.qpos, data["final_sim_qpos"], atol=1e-9, rtol=0)
            if max_q_error > 1e-9 or max_object_error > 1e-9:
                raise ValueError("Recorded actions do not reproduce recorded physics states")
            result.update(replay_max_qpos_error=max_q_error,
                          replay_max_object_position_error_m=max_object_error,
                          replay_final_metrics=arm.placement_metrics())
        return result


def preview(folder, output):
    data = np.load(folder / "joint_data.npz", allow_pickle=False)
    phases = ["approach", "grasp_hold", "lift", "above_bin", "release", "verify"]
    rows = []
    for camera in ("cam_fixed", "cam_wrist"):
        row = []
        for phase in phases:
            index = int(np.flatnonzero(data["phase"] == phase)[-1])
            image = cv2.imread(str(folder / camera / f"{index:05d}.jpg"))
            if image.shape[:2] == (360, 640):
                image = cv2.copyMakeBorder(cv2.resize(image, (256, 144), interpolation=cv2.INTER_AREA),
                                           56, 56, 0, 0, cv2.BORDER_CONSTANT, value=(22, 20, 18))
            cv2.rectangle(image, (0, 0), (255, 24), (245, 245, 245), -1)
            cv2.putText(image, f"{camera}: {phase}", (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (20, 20, 20), 1, cv2.LINE_AA)
            row.append(image)
        rows.append(np.concatenate(row, axis=1))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.concatenate(rows, axis=0)):
        raise OSError("Could not save preview")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/hardstop_dataset_check")
    args = parser.parse_args()
    folders = sorted(args.run.glob("ep_*"))
    if not folders:
        parser.error("No successful episodes found")
    arms = {}
    report = []
    for folder in folders:
        schema = json.loads((folder / "metadata.json").read_text())["schema"]
        if args.replay and schema not in arms:
            arms[schema] = HardstopArm(scene=LEGACY_SCENE if schema == "agv_dummyx_screwdriver_v2" else SCENE)
        report.append(check_episode(folder, arms.get(schema)))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    preview(folders[0], args.output / "preview.jpg")
    print(json.dumps(report, indent=2))
    print(f"Preview: {args.output / 'preview.jpg'}")


if __name__ == "__main__":
    main()
