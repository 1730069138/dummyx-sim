#!/usr/bin/env python3
"""Compare matched screwdriver runs, including shared-perception VLSA."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiment_recording import TIMINGS, TIME_DEFINITION, TIMING_DEFINITION, aggregate_timings


MODES = ("no_obstacle", "obstacle", "apf")
LABELS = ("无障碍\n原始策略", "有障碍\n原始策略", "有障碍\n人工势场避障")
MODE_LABELS = dict(zip(MODES, LABELS), vlsa="有障碍\nVLSA（共享感知）")


def read_json(path):
    return json.loads(path.read_text())


def parse_cell(value):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def load_run(path, mode):
    path = path.resolve()
    directory = path / "experiment"
    config = read_json(path / "run_config.json")
    experiment_config = read_json(directory / "run_config.json")
    summary_path = directory / "summary.json"
    if config.get("defer_plots") is True and not summary_path.exists():
        summary = None
        episodes = [read_json(metric_path) for metric_path in sorted(directory.glob("ep_*/metrics.json"))]
    else:
        summary = read_json(summary_path)
        with (directory / "episodes.csv").open(newline="") as stream:
            episodes = [{key: parse_cell(value) for key, value in row.items()}
                        for row in csv.DictReader(stream)]
    if not episodes:
        raise ValueError(f"{path}: no recorded episodes")
    if any(episode.get("evaluation_mode") != mode for episode in episodes):
        raise ValueError(f"{path}: episode evaluation_mode does not match {mode}")
    if summary is not None and summary.get("evaluation_mode") != mode:
        raise ValueError(f"{path}: summary evaluation_mode does not match {mode}")
    if config.get("evaluation_mode") != mode:
        raise ValueError(f"{path}: run configuration evaluation_mode does not match {mode}")
    for episode in episodes:
        expected_obstacle = mode != "no_obstacle"
        if episode.get("obstacle_present") is not expected_obstacle:
            raise ValueError(f"{path}: obstacle presence conflicts with {mode}")
        if episode.get("task") != "screwdriver":
            raise ValueError(f"{path}: comparison currently requires the screwdriver task")
        if episode.get("task_prompt") != config.get("task_prompt"):
            raise ValueError(f"{path}: episode task prompt conflicts with run configuration")
    return {"path": path, "mode": mode, "config": config,
            "experiment_config": experiment_config, "summary": summary, "episodes": episodes}


def require_matching(runs, key):
    values = [run["config"].get(key) for run in runs]
    if any(value is None for value in values) or any(value != values[0] for value in values[1:]):
        raise ValueError(f"Runs lack matching {key}: {values}")
    return values[0]


def validate_matching(runs):
    matching = {key: require_matching(runs, key) for key in
                ("seed", "chunk_steps", "task_prompt", "fixed_eval", "max_steps", "fps",
                 "success_criteria", "collision_criteria", "comparison_scene_signature")}
    reference = runs[0]
    scenes = {run["config"].get("scene") for run in runs}
    poses = []
    reference_ids = [episode["episode"] for episode in reference["episodes"]]
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("Duplicate episode identifiers")
    for run in runs:
        if [episode["episode"] for episode in run["episodes"]] != reference_ids:
            raise ValueError("Runs must contain matching episode identifiers in the same order")
        for actual, expected in zip(run["episodes"], reference["episodes"]):
            xy, expected_xy = actual.get("initial_xy"), expected.get("initial_xy")
            yaw, expected_yaw = actual.get("initial_yaw"), expected.get("initial_yaw")
            if (xy is None or expected_xy is None or np.shape(xy) != (2,) or np.shape(expected_xy) != (2,)
                    or not np.allclose(xy, expected_xy, rtol=0, atol=1e-9)
                    or yaw is None or expected_yaw is None or not np.isclose(yaw, expected_yaw, rtol=0, atol=1e-9)):
                raise ValueError(f"Episode {actual['episode']}: unmatched initial object pose")
        raw_manifest = run["config"].get("model_manifest")
        if not isinstance(raw_manifest, dict) or not raw_manifest:
            raise ValueError(f"{run['path']}: missing model manifest")
        robot_manifest = {key: value for key, value in raw_manifest.items() if key not in scenes}
        reference_manifest = {key: value for key, value in reference["config"]["model_manifest"].items()
                              if key not in scenes}
        if robot_manifest != reference_manifest:
            raise ValueError("Robot or shared model assets differ between runs")
    obstacles = [run for run in runs if run["mode"] != "no_obstacle"]
    if any(run["config"].get("scene") != obstacles[0]["config"].get("scene")
           or run["config"]["model_manifest"] != obstacles[0]["config"]["model_manifest"]
           for run in obstacles[1:]):
        raise ValueError("Obstacle, APF and VLSA runs use different scene models")
    metadata = [run["experiment_config"].get("server_metadata") for run in runs]
    if metadata[1:] != metadata[:-1]:
        raise ValueError("Policy-server metadata differs between runs")
    for episode in reference["episodes"]:
        poses.append({"episode": episode["episode"], "initial_xy": episode["initial_xy"],
                      "initial_yaw": episode["initial_yaw"]})
    matching["initial_object_poses"] = poses
    matching["policy_server_metadata"] = metadata[0]
    return matching


def summarize(run, *, check_summary=True):
    episodes = run["episodes"]
    evaluated = [episode for episode in episodes if episode.get("reason") != "Viewer closed"]
    successes = sum(episode.get("success") is True for episode in evaluated)
    contacts = [episode for episode in evaluated if isinstance(episode.get("contact_collision"), bool)]
    if run["mode"] == "no_obstacle":
        contacts = []
    tilts = [episode for episode in evaluated if episode.get("obstacle_present") is True
             and isinstance(episode.get("pillar_tilted"), bool)]
    if run["mode"] == "no_obstacle":
        tilts = []
    durations = [float(episode["successful_completion_time_s"]) for episode in evaluated
                 if episode.get("success") is True and episode.get("successful_completion_time_s") is not None]
    contact_count = sum(episode["contact_collision"] for episode in contacts)
    tilt_count = sum(episode["pillar_tilted"] for episode in tilts)
    safe_count = sum(episode.get("success") is True and not episode["contact_collision"] for episode in contacts)
    row = {"evaluation_mode": run["mode"], "run_path": str(run["path"]),
           "recorded_episodes": len(episodes), "evaluated_episodes": len(evaluated),
           "interrupted_episodes": len(episodes) - len(evaluated),
           "successes": successes, "failures": len(evaluated) - successes,
           "success_rate": successes / len(evaluated) if evaluated else None,
           "collision_evaluated_episodes": len(contacts), "contact_collision_episodes": contact_count,
           "contact_collision_rate": contact_count / len(contacts) if contacts else None,
           "pillar_tilt_evaluated_episodes": len(tilts),
           "pillar_tilted_episodes": tilt_count,
           "pillar_tilt_rate": tilt_count / len(tilts) if tilts else None,
           "safe_task_success_rate": safe_count / len(contacts) if contacts else None,
           "successful_completion_time_samples": len(durations),
           "successful_completion_time_mean_s": float(np.mean(durations)) if durations else None,
           "successful_completion_time_median_s": float(np.median(durations)) if durations else None,
           "minimum_obstacle_clearance_m": None}
    qp_counts = [episode["vlsa_qp_failures"] for episode in evaluated
                 if episode.get("vlsa_qp_failures") is not None]
    row["vlsa_qp_failures"] = sum(qp_counts) if qp_counts else None
    if check_summary and run["summary"] is not None:
        for key in ("recorded_episodes", "evaluated_episodes", "interrupted_episodes", "successes", "failures", "success_rate"):
            if run["summary"].get(key) != row[key]:
                raise ValueError(f"{run['path']}: episodes.csv and summary.json disagree on {key}")
    row.update(aggregate_timings(run["path"], evaluated))
    return row


def load_comparison_rounds(path, vlsa_path=None):
    path = path.resolve()
    schedule = read_json(path / "schedule.json")
    modes = tuple(schedule.get("mode_order", MODES))
    entries = schedule["episodes"]
    if vlsa_path is not None:
        vlsa_schedule = read_json(vlsa_path.resolve() / "schedule.json")
        if set(modes) != set(MODES) or set(vlsa_schedule.get("mode_order", ())) != {"vlsa"}:
            raise ValueError("Combine a three-mode comparison with a VLSA-only comparison")
        modes = (*modes, "vlsa")
        entries = entries + vlsa_schedule["episodes"]
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODE_LABELS for mode in modes):
        raise ValueError("Unrecognized comparison mode order")
    # Keep the original report order while allowing one or four methods.
    modes = tuple(mode for mode in (*MODES, "vlsa") if mode in modes)
    rounds = {}
    for entry in entries:
        round_index, mode = entry["round"], entry["evaluation_mode"]
        if mode not in modes:
            raise ValueError(f"Round {round_index}: unexpected evaluation mode {mode}")
        group = rounds.setdefault(round_index, {})
        if mode in group:
            raise ValueError(f"Round {round_index}: duplicate {mode} run")
        group[mode] = Path(entry["run_path"])
    if not rounds:
        raise ValueError(f"{path}: no completed comparison rounds")
    combined = [{"path": path, "mode": mode, "episodes": []} for mode in modes]
    matched_rounds = []
    reference_runs = None
    common = None
    for round_index, group in sorted(rounds.items()):
        if set(group) != set(modes):
            raise ValueError(f"Round {round_index}: all requested evaluation modes are required")
        runs = [load_run(group[mode], mode) for mode in modes]
        matched = validate_matching(runs)
        settings = {key: value for key, value in matched.items()
                    if key not in ("seed", "initial_object_poses")}
        if common is None:
            common, reference_runs = settings, runs
        elif settings != common:
            raise ValueError(f"Round {round_index}: task, control, scene, or policy-server settings differ")
        for run, reference, target in zip(runs, reference_runs, combined):
            if (run["config"].get("scene") != reference["config"].get("scene")
                    or run["config"]["model_manifest"] != reference["config"]["model_manifest"]):
                raise ValueError(f"Round {round_index}: {run['mode']} model assets differ between rounds")
            summarize(run)
            target["episodes"].extend({**episode, "source_run_path": str(run["path"])}
                                      for episode in run["episodes"])
        matched_rounds.append({"round": round_index,
                               "run_paths": {run["mode"]: str(run["path"]) for run in runs},
                               "matching_validation": matched})
    return combined, {"common_settings": common, "matched_rounds": len(matched_rounds),
                      "rounds": matched_rounds}


def plot_comparison(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "WenQuanYi Micro Hei", "axes.unicode_minus": False,
                         "pdf.fonttype": 42})

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    labels = [MODE_LABELS[row["evaluation_mode"]] for row in rows]
    if len(rows) == 4:
        labels = [label.replace("人工势场避障", "人工势场\n避障").replace("VLSA（共享感知）", "VLSA\n（共享感知）")
                  for label in labels]
    positions = range(len(rows))
    xlim = (-.6, len(rows) - .4)
    figure, axes = plt.subplots(2, 4, figsize=(17, 9), constrained_layout=True)

    def bars(axis, key, title, ylabel, denominator, scale=1.):
        for index, row in enumerate(rows):
            value = row[key]
            if value is None:
                axis.text(index, .03, "不适用", ha="center", transform=axis.get_xaxis_transform())
            else:
                axis.bar(index, scale * value, color=colors[index])
                axis.annotate(f"{scale * value:.1f}\n样本数={row[denominator]}", (index, scale * value),
                              xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
        axis.set(xticks=positions, xticklabels=labels, title=title, ylabel=ylabel, xlim=xlim)
        axis.grid(axis="y", alpha=.25)
        valid = [scale * row[key] for row in rows if row[key] is not None]
        axis.set_ylim(0, 125 if scale == 100 else max(valid + [1]) * 1.3)

    bars(axes[0, 0], "success_rate", "任务成功率", "比例（%）", "evaluated_episodes", 100)
    bars(axes[0, 1], "contact_collision_rate", "障碍接触碰撞率", "比例（%）", "collision_evaluated_episodes", 100)
    bars(axes[0, 2], "safe_task_success_rate", "安全任务成功率", "比例（%）", "collision_evaluated_episodes", 100)
    bars(axes[0, 3], "successful_completion_time_mean_s", "成功回合平均完成时间", "控制仿真时间（秒）", "successful_completion_time_samples")
    timing_labels = {"policy_inference_time": "策略请求（网络与服务端）",
                     "safety_layer_time": "完整安全层处理",
                     "solver_time": "实际优化求解（VLSA：二次规划）"}
    for axis, key in zip(axes[1, :3], TIMINGS):
        for index, row in enumerate(rows):
            mean, p95 = row[f"{key}_mean_ms"], row[f"{key}_p95_ms"]
            if mean is None:
                axis.text(index, .03, "不适用", ha="center", transform=axis.get_xaxis_transform())
            else:
                axis.bar(index - .16, mean, width=.32, color=colors[index], label="均值" if index == 0 else None)
                if p95 is not None:
                    axis.bar(index + .16, p95, width=.32, color=colors[index], alpha=.45,
                             label="第95百分位" if index == 0 else None)
                maximum = max(mean, p95 if p95 is not None else mean)
                axis.annotate(f"{mean:.2f} / {p95:.2f}\n样本数={row[f'{key}_samples']}" if p95 is not None
                              else f"{mean:.2f} / 不适用\n样本数={row[f'{key}_samples']}", (index, maximum),
                              xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
        values = [row[f"{key}_{stat}_ms"] for row in rows for stat in ("mean", "p95")
                  if row[f"{key}_{stat}_ms"] is not None]
        axis.set(xticks=positions, xticklabels=labels, title=timing_labels[key], ylabel="耗时（毫秒）", xlim=xlim,
                 ylim=(0, max(values + [1]) * 1.4))
        axis.text(.02, .97, "均值／第95百分位", transform=axis.transAxes, va="top", fontsize=9)
        axis.grid(axis="y", alpha=.25)
    axes[1, 3].axis("off")
    note = ("至少一种模式仅有1个或更少有效回合。\n仅展示本次结果，不能得出统计性能结论。"
            if min(row["evaluated_episodes"] for row in rows) <= 1
            else "比例仅描述已记录回合。\n未计算置信区间或作显著性判断。")
    axes[1, 3].text(0, .98, "统计口径\n\n" + note + "\n\n"
                   "未测量或不适用：标为“不适用”。\n手动关闭窗口的回合不参与统计。\n运行异常计为任务失败。\n"
                   "完成时间：控制步数除以控制频率。\n策略请求：客户端至服务端完整往返。\n安全层：完整APF或VLSA处理。\n"
                   "求解：仅统计实际优化调用。\nVLSA：共享感知、末端速度安全层。\n第95百分位：合并有效计时样本计算。\n最小安全间距：尚未记录。",
                   va="top", fontsize=10, linespacing=1.5, wrap=True)
    figure.suptitle("统一设置下的螺丝刀任务对比", fontsize=17)
    figure.savefig(output / "comparison.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "comparison.pdf", bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("no_obstacle_run", type=Path, nargs="?")
    parser.add_argument("obstacle_run", type=Path, nargs="?")
    parser.add_argument("apf_run", type=Path, nargs="?")
    parser.add_argument("--comparison_run", type=Path,
                        help="Completed round-robin run; aggregate all recorded rounds")
    parser.add_argument("--vlsa_comparison_run", type=Path,
                        help="Matched VLSA-only rounds to combine with an existing three-mode --comparison_run")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    positional_runs = (args.no_obstacle_run, args.obstacle_run, args.apf_run)
    if args.vlsa_comparison_run is not None and args.comparison_run is None:
        parser.error("--vlsa_comparison_run requires --comparison_run")
    if args.comparison_run is not None:
        if any(path is not None for path in positional_runs):
            parser.error("--comparison_run cannot be combined with individual run directories")
        if args.output is None:
            args.output = (args.vlsa_comparison_run / "四组对比" if args.vlsa_comparison_run is not None
                           else args.comparison_run / "comparison")
    elif any(path is None for path in positional_runs) or args.output is None:
        parser.error("Provide all three run directories and --output, or use --comparison_run")
    try:
        if args.comparison_run is not None:
            runs, matched = load_comparison_rounds(args.comparison_run, args.vlsa_comparison_run)
            rows = [summarize(run, check_summary=False) for run in runs]
        else:
            runs = [load_run(path, mode) for path, mode in zip(positional_runs, MODES)]
            matched = validate_matching(runs)
            rows = [summarize(run) for run in runs]
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    note = ("Demonstration samples only (N <= 1 in at least one mode); no statistical performance conclusion."
            if min(row["evaluated_episodes"] for row in rows) <= 1
            else "Observed rates describe these recorded episodes; no confidence interval or significance claim.")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "comparison.json").write_text(json.dumps(
        {"comparison_note": note, "matching_validation": matched, "time_definition": TIME_DEFINITION,
         "timing_definition": TIMING_DEFINITION, "modes": rows}, indent=2, allow_nan=False) + "\n")
    plot_comparison(args.output, rows)
    print(f"Comparison written to {args.output.resolve()}")
    print(note)


if __name__ == "__main__":
    main()
