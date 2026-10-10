"""Replay a recorded pi0 episode and export overview and front-view H.264 MP4s."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/collect"))
from collect_hardstop_screwdriver import HardstopArm, model_manifest


def export_video(episode, output=None, view="both"):
    if view not in ("overview", "front", "both"):
        raise ValueError("view must be overview, front or both")
    episode = Path(episode)
    os.environ["MUJOCO_GL"] = "egl"
    config = json.loads((episode.parent / "run_config.json").read_text())
    scene = ROOT / config["scene"]
    sweep = config.get("task") == "zip_tie_sweep"
    if sweep:
        from collect_zip_tie_sweep import CleanupArm, model_manifest as sweep_model_manifest
        current_manifest = sweep_model_manifest()
    else:
        current_manifest = model_manifest()
    dependencies = set()
    pending = [scene]
    while pending:
        path = pending.pop().resolve()
        if path in dependencies:
            continue
        dependencies.add(path)
        current_manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".xml":
            for element in ET.parse(path).iter():
                if element.get("file"):
                    pending.append(path.parent / element.get("file"))
    recorded_manifest = dict(config["model_manifest"])
    # The default manifest also tracks obstacle.xml for collection safety, even without an include.
    obstacle = ROOT / "models/agv_dummyx/obstacle.xml"
    if obstacle.resolve() not in dependencies:
        key = str(obstacle.relative_to(ROOT))
        current_manifest.pop(key, None)
        recorded_manifest.pop(key, None)
    if recorded_manifest != current_manifest:
        raise RuntimeError("Scene/model files changed since recording; restore the recorded model before replay")
    arm = CleanupArm() if sweep else HardstopArm(scene=scene)
    if config["mujoco_version"] != arm.mj.__version__:
        raise RuntimeError("Replay requires the recorded MuJoCo version")
    with np.load(episode / "trace.npz", allow_pickle=False) as trace:
        frames = trace["simulation_states"]
        fps = int(trace["fps"])
        spec = arm.mj.mjtState(int(trace["simulation_state_spec"]))
    if frames.ndim != 2 or frames.shape[1] != arm.mj.mj_stateSize(arm.model, spec):
        raise ValueError("Recorded simulation state dimensions do not match the model")
    output = Path(output) if output is not None else episode / "replay.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    option = arm.mj.MjvOption()
    option.geomgroup[3:] = 0
    width, height = 1280, 720
    front_camera = arm.mj.MjvCamera()
    front_camera.type = arm.mj.mjtCamera.mjCAMERA_FREE
    front_camera.lookat[:] = [.55, .10, .90] if sweep else [.47, 0., .86]
    front_camera.distance = 1.5
    front_camera.azimuth = 180.
    front_camera.elevation = -30.
    views = []
    if view in ("overview", "both"):
        views.append((output, "overview"))
    if view in ("front", "both"):
        views.append((output.with_name(f"{output.stem}_front{output.suffix}"), front_camera))
    # Recorded state playback preserves object/contact poses rather than rerunning physics.
    with arm.mj.Renderer(arm.model, height=height, width=width) as renderer:
        for view_output, camera in views:
            process = subprocess.Popen([
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an",
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(view_output)], stdin=subprocess.PIPE)
            try:
                for state in frames:
                    arm.mj.mj_setState(arm.model, arm.data, state, spec)
                    arm.mj.mj_forward(arm.model, arm.data)
                    renderer.update_scene(arm.data, camera=camera, scene_option=option)
                    process.stdin.write(renderer.render().tobytes())
            finally:
                process.stdin.close()
                code = process.wait()
            if code:
                raise RuntimeError(f"ffmpeg failed with exit code {code}")
            print(f"Video: {view_output} ({len(frames)} frames, {fps} FPS)", flush=True)
    return tuple(path for path, _ in views)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, nargs="?",
                        help="Recorded ep_XXXX directory; defaults to the moved 100-episode run's ep_0000")
    parser.add_argument("--comparison_run", type=Path,
                        help="Comparison directory containing schedule.json; select one completed case with --round and --mode")
    parser.add_argument("--round", type=int, help="Comparison round number, starting at 1")
    parser.add_argument("--mode", choices=("no_obstacle", "obstacle", "apf", "vlsa"),
                        help="no_obstacle: 无障碍; obstacle: 有障碍; apf: 有障碍＋GroundingDINO APF")
    parser.add_argument("--view", choices=("overview", "front", "both"), default="both",
                        help="Camera view: overview (全景), front (正面), or both (默认双视角)")
    parser.add_argument("--output", type=Path, help="Overview output; defaults to episode/replay.mp4, with a second *_front.mp4")
    args = parser.parse_args()
    if args.comparison_run is not None:
        if args.episode is not None or args.round is None or args.mode is None:
            parser.error("--comparison_run requires --round and --mode, without an episode path")
        if args.round < 1:
            parser.error("--round starts at 1")
        try:
            schedule = json.loads((args.comparison_run / "schedule.json").read_text())
            matches = [entry for entry in schedule["episodes"]
                       if entry["round"] == args.round - 1 and entry["evaluation_mode"] == args.mode]
            if len(matches) != 1:
                parser.error("The selected round and mode must identify one completed episode")
            args.episode = Path(matches[0]["run_path"]) / "ep_0000"
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
    elif args.round is not None or args.mode is not None:
        parser.error("--round and --mode require --comparison_run")
    elif args.episode is None:
        args.episode = ROOT / "inference_results/pi0_screwdriver_100ep_seed42_20261002_193214_038376/ep_0000"
    export_video(args.episode, args.output, args.view)


if __name__ == "__main__":
    main()
