"""Convert zip-tie sponge-sweeping episodes to the LeRobot format used by OpenPI."""

import argparse
import gc
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image


def episodes(data_dir):
    result = sorted(data_dir.glob("ep_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not result:
        raise ValueError(f"No ep_* directories in {data_dir}")
    return result


def check_episode(folder):
    metadata = json.loads((folder / "metadata.json").read_text())
    if metadata.get("schema") != "agv_dummyx_workbench_cleanup_zip_tie_short_table_v7" or metadata.get("fps") != 50:
        raise ValueError(f"Unsupported episode schema or FPS: {folder}")
    with np.load(folder / "joint_data.npz") as data:
        states, actions = data["qpos"], data["actions"]
    if states.ndim != 2 or states.shape[1] != 7 or actions.shape != states.shape:
        raise ValueError(f"Expected matching (N, 7) state/action arrays: {folder}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"Nonfinite state/action: {folder}")
    for camera, size in (("cam_fixed", (640, 360)), ("cam_wrist", (256, 256))):
        images = sorted((folder / camera).glob("*.jpg"))
        if len(images) != len(states) or [p.name for p in images] != [f"{i:05d}.jpg" for i in range(len(states))]:
            raise ValueError(f"Image count or numbering mismatch: {folder / camera}")
        with Image.open(images[0]) as image:
            if image.size != size:
                raise ValueError(f"Unexpected {camera} resolution: {image.size}")
    task = (folder / "instruction.txt").read_text().strip()
    if not task:
        raise ValueError(f"Empty instruction: {folder}")
    return states, actions, task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/agv_dummyx_workbench_cleanup_zip_ties_short_table"))
    parser.add_argument("--repo-id", default="local/agv_dummyx_zip_tie_sweep")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue an existing dataset from its recorded episode count")
    args = parser.parse_args()
    source = episodes(args.data_dir)
    checked = [(folder, *check_episode(folder)) for folder in source]
    total = sum(len(states) for _, states, _, _ in checked)
    print(f"Validated {len(checked)} episodes, {total} frames; fixed RGB 640x360, wrist RGB 256x256, state/action 7D, 50 FPS")
    if args.check_only:
        return

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output = (Path.home() / ".cache/huggingface/lerobot" / args.repo_id).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {output}")
    if args.resume:
        if not output.exists():
            raise FileNotFoundError(f"No dataset to resume: {output}")
        info = json.loads((output / "meta/info.json").read_text())
        completed = info["total_episodes"]
        if completed < 1 or completed > len(checked):
            raise ValueError(f"Invalid completed episode count: {completed}")
        if info["fps"] != 50 or info["robot_type"] != "agv_dummyx":
            raise ValueError(f"Incompatible existing dataset: {output}")
        if any(folder.name != f"ep_{i:04d}" for i, (folder, *_rest) in enumerate(checked)):
            raise ValueError("Source episode indices must be contiguous from ep_0000")
        orphan = output / info["data_path"].format(
            episode_chunk=completed // info["chunks_size"], episode_index=completed)
        if orphan.exists():
            orphan.unlink()
            print(f"Removed uncommitted parquet: {orphan}")
        dataset = LeRobotDataset(args.repo_id, root=output, episodes=[completed - 1])
        if dataset.meta.total_episodes != completed:
            raise ValueError("Existing dataset episode count changed during loading")
        dataset.episodes = None
        dataset.hf_dataset = dataset.create_hf_dataset()
        dataset.episode_buffer = dataset.create_episode_buffer()
        dataset.start_image_writer(num_threads=4)
        print(f"Resuming at ep_{completed:04d}; {completed} episodes already saved")
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=output,
            robot_type="agv_dummyx",
            fps=50,
            features={
                "image": {"dtype": "image", "shape": (360, 640, 3), "names": ["height", "width", "channel"]},
                "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            },
            image_writer_threads=4,
        )
        completed = 0
    task_is_argument = "task" in inspect.signature(dataset.add_frame).parameters
    for folder, states, actions, task in checked[completed:]:
        for i in range(len(states)):
            with Image.open(folder / "cam_fixed" / f"{i:05d}.jpg") as image:
                fixed = np.asarray(image.convert("RGB"))
            with Image.open(folder / "cam_wrist" / f"{i:05d}.jpg") as image:
                wrist = np.asarray(image.convert("RGB"))
            frame = {"image": fixed, "wrist_image": wrist,
                     "state": states[i], "actions": actions[i]}
            if task_is_argument:
                dataset.add_frame(frame, task=task)
            else:
                dataset.add_frame({**frame, "task": task})
        dataset.save_episode()
        dataset.hf_dataset = dataset.create_hf_dataset()
        gc.collect()
        print(f"Saved {folder.name}: {len(states)} frames")
    print(f"LeRobot dataset: {output}")


if __name__ == "__main__":
    main()
