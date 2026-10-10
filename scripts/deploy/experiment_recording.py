"""Export optional experiment tables and figures from existing inference traces."""

import csv
import json
from pathlib import Path

import numpy as np


TIME_DEFINITION = "Nominal control time = executed control steps / fps; not wall-clock time."
SAFETY_DEFINITION = "Obstacle contact requires an obstacle, executed actions, and recorded contact monitoring; manual viewer closure is excluded. Robot-table contact is reported separately."
TIMINGS = {
    "policy_inference_time": "Policy request (network + server)",
    "safety_layer_time": "Full safety layer",
    "solver_time": "Actual optimization solves",
}
TIMING_DEFINITION = "Policy request includes network and server processing. Full safety-layer time is measured per executed APF or VLSA step. Solver time includes only actual optimization calls (VLSA: CVXPY problem.solve, including canonicalization). Unmeasured or uncalled components are N/A."


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n")


def export_run_config(output, task, config, fps, server_metadata):
    directory = Path(output) / "experiment"
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "run_config.json", {
        "task": task,
        "fps": fps,
        "cli_config": config,
        "server_metadata": server_metadata,
        "state_columns": [f"joint{i}_rad" for i in range(1, 7)] + ["gripper_width_m"],
        "action_columns": [f"joint{i}_target_rad" for i in range(1, 7)] + ["gripper_target_m"],
        "trajectory_alignment": "All trajectory rows are measured after the corresponding executed action.",
        "time_s_definition": "Row i uses (i + 1) / fps. " + TIME_DEFINITION,
        "completion_time_definition": "Control duration of successful episodes only. " + TIME_DEFINITION,
        "safety_definition": SAFETY_DEFINITION,
        "timing_definition": TIMING_DEFINITION,
        "unavailable_metrics": "Unrecorded metrics (including minimum obstacle clearance) are null in JSON, empty in CSV, N/A in figures.",
        "original_result_pattern": "../ep_{episode:04d}/result.json",
        "original_trace_pattern": "../ep_{episode:04d}/trace.npz",
        "original_run_config": "../run_config.json (when present, includes the original model manifest)",
    })


def _save_figure(figure, directory, name):
    figure.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(directory / f"{name}.pdf", bbox_inches="tight")


def _plot_episode(directory, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    interrupted = metrics.get("reason") == "Viewer closed"
    outcome = "INTERRUPTED" if interrupted else "SUCCESS" if metrics.get("success") else "FAILURE"
    completion = metrics["successful_completion_time_s"]
    def status(value):
        return "N/A" if value is None else "Yes" if value else "No"
    rows = [
        ["Evaluation mode", metrics.get("evaluation_mode", "N/A")],
        ["Task outcome", outcome],
        ["Completion control time (successful only)", f"{completion:.2f} s" if completion is not None else "N/A"],
        ["Executed control steps", str(metrics["control_steps"])],
        ["Obstacle contact", status(metrics["contact_collision"])],
        ["Safe task success (no obstacle contact)", status(metrics["safe_task_success"])],
        ["Robot-table contact", status(metrics["robot_table_contact"])],
        ["Minimum obstacle clearance", "N/A - not recorded"],
    ]
    if "collected_count" in metrics:
        rows.insert(3, ["Collected zip ties", str(metrics["collected_count"])])
    figure, (axis, timing_axis) = plt.subplots(2, 1, figsize=(11, 8.3),
                                              gridspec_kw={"height_ratios": [3, 1.3]})
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=["Metric", "Result"], colWidths=[.65, .35],
                       cellLoc="left", bbox=[0, .12, 1, .82])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    axis.set_title(f"{metrics['task']}: episode {metrics['episode']}", fontsize=15, pad=15)
    axis.text(0, .02, "N/A = not evaluated or not recorded.\nControl time is nominal simulation time, not wall-clock time.",
              transform=axis.transAxes, fontsize=10, color="dimgray")
    timing_axis.axis("off")
    _timing_table(timing_axis, metrics)
    _save_figure(figure, directory, "task_result")
    plt.close(figure)


def _timing_table(axis, metrics):
    def value(key):
        item = metrics.get(key)
        return f"{item:.3f}" if item is not None else "N/A"
    rows = [[label, value(f"{key}_mean_ms"), value(f"{key}_p95_ms"),
             str(metrics.get(f"{key}_samples", 0))] for key, label in TIMINGS.items()]
    table = axis.table(cellText=rows, colLabels=["Measured component", "Mean (ms)", "P95 (ms)", "Samples"],
                       colWidths=[.52, .17, .17, .14], cellLoc="left", bbox=[0, .35, 1, .65])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    axis.text(0, .05, "Policy: client request including network and server. Safety: full APF or VLSA processing.\n"
              "Solver: actual optimization calls only. P95 pools recorded samples; unavailable values are N/A.",
              transform=axis.transAxes, fontsize=9, color="dimgray")


def export_episode(output, episode, task, result, actions, measured_states,
                   object_trajectories, fps, *, plots=True):
    actions = np.asarray(actions)
    measured = np.asarray(measured_states)
    objects = {name: np.asarray(xyz) for name, xyz in object_trajectories.items()}
    count = len(actions)
    if actions.shape != (count, 7) or measured.shape != (count, 7):
        raise ValueError("Actions and post-action measured states must have matching (N, 7) shapes")
    if any(xyz.shape != (count, 3) for xyz in objects.values()):
        raise ValueError("Post-action object trajectories must have matching (N, 3) shapes")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    directory = Path(output) / "experiment" / f"ep_{episode:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    times = (np.arange(count) + 1) / fps
    fields = ["time_s"]
    for joint in range(1, 7):
        fields.extend([f"joint{joint}_actual_rad", f"joint{joint}_target_rad"])
    fields.extend(["gripper_actual_m", "gripper_target_m"])
    for name in objects:
        fields.extend([f"{name}_{axis}_m" for axis in "xyz"])
    with (directory / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for index, time_s in enumerate(times):
            row = [time_s]
            for joint in range(6):
                row.extend([measured[index, joint], actions[index, joint]])
            row.extend([measured[index, 6], actions[index, 6]])
            for xyz in objects.values():
                row.extend(xyz[index])
            writer.writerow(_json_value(row))
    metrics = {key: _json_value(value) for key, value in result.items()
               if value is None or isinstance(value, (str, bool, int, float, np.generic))}
    if "initial_xy" in result:
        metrics["initial_xy"] = _json_value(result["initial_xy"])
    metrics.update({
        "episode": episode,
        "task": task,
        "control_steps": count,
        "control_duration_s": count / fps,
        "control_time_definition": TIME_DEFINITION,
        "successful_completion_time_s": count / fps if metrics.get("success") and metrics.get("reason") != "Viewer closed" else None,
    })
    metrics.setdefault("minimum_obstacle_clearance_m", None)
    contact_recorded = (metrics.get("obstacle_present") is True and count > 0
                        and metrics.get("reason") != "Viewer closed"
                        and isinstance(metrics.get("contact_collision"), bool))
    metrics["contact_collision"] = metrics.get("contact_collision") if contact_recorded else None
    metrics["pillar_tilted"] = metrics.get("pillar_tilted") if contact_recorded else None
    metrics["safe_task_success"] = bool(metrics.get("success")) and not metrics["contact_collision"] if contact_recorded else None
    metrics["robot_table_contact"] = metrics.get("robot_table_contact") if count and metrics.get("reason") != "Viewer closed" else None
    metrics["safety_definition"] = SAFETY_DEFINITION
    metrics["timing_definition"] = TIMING_DEFINITION
    metrics = _json_value(metrics)
    _write_json(directory / "metrics.json", metrics)
    if plots:
        _plot_episode(directory, metrics)
    return metrics


def _plot_summary(directory, metrics, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    outcome_axis, safety_axis, time_axis, info_axis = axes.flat
    counts = [summary["successes"], summary["failures"], summary["interrupted_episodes"]]
    colors = ["tab:green", "tab:red", "tab:gray"]
    bars = outcome_axis.bar(["Success", "Failure", "Interrupted"], counts, color=colors)
    outcome_axis.bar_label(bars, padding=3)
    outcome_axis.set(ylabel="Episodes", ylim=(0, max(counts + [1]) * 1.2))
    rate = summary["success_rate"]
    outcome_axis.set_title(f"Task success rate: {rate:.1%} (N = {summary['evaluated_episodes']})" if rate is not None else "Task success rate: N/A (N = 0)")
    safety_rows = []
    for label, key, denominator in [
            ("Obstacle\ncontact", "contact_collision_rate", "collision_evaluated_episodes"),
            ("Safe task\nsuccess", "safe_task_success_rate", "collision_evaluated_episodes"),
            ("Robot-table\ncontact", "robot_table_contact_rate", "robot_table_contact_evaluated_episodes")]:
        if summary[key] is not None:
            safety_rows.append((label, summary[key], summary[denominator]))
    if safety_rows:
        bars = safety_axis.bar([row[0] for row in safety_rows], [100 * row[1] for row in safety_rows], color="tab:blue")
        safety_axis.bar_label(bars, labels=[f"{100 * row[1]:.1f}%\nN = {row[2]}" for row in safety_rows], padding=3)
        safety_axis.set(ylabel="Rate (%)", ylim=(0, 125))
        safety_axis.grid(axis="y", alpha=.3)
    else:
        safety_axis.text(.5, .5, "N/A\nNo evaluated contact monitoring", ha="center", va="center", transform=safety_axis.transAxes)
        safety_axis.set_xticks([])
        safety_axis.set_yticks([])
    safety_axis.set_title("Measured safety results")
    successful = [metric for metric in metrics if metric["successful_completion_time_s"] is not None]
    if successful:
        time_axis.scatter([metric["episode"] for metric in successful], [metric["successful_completion_time_s"] for metric in successful], color="tab:green")
        time_axis.axhline(summary["successful_completion_time_mean_s"], color="dimgray", linestyle="--", label=f"Mean = {summary['successful_completion_time_mean_s']:.2f} s")
        time_axis.legend()
        if len(successful) <= 20:
            time_axis.set_xticks([metric["episode"] for metric in successful])
    else:
        time_axis.text(.5, .5, "N/A\nNo successful episodes", ha="center", va="center", transform=time_axis.transAxes)
    time_axis.set(xlabel="Episode", ylabel="Completion control time (s)", title="Successful episodes only")
    time_axis.grid(alpha=.3)
    info_axis.axis("off")
    info_axis.text(0, .95, "Reporting scope", fontsize=13, weight="bold", va="top")
    info_axis.text(0, .82,
                   f"Successful time samples: {len(successful)}\n"
                   f"Obstacle-contact samples: {summary['collision_evaluated_episodes']}\n"
                   f"Robot-table-contact samples: {summary['robot_table_contact_evaluated_episodes']}\n\n"
                   "Minimum obstacle clearance: N/A\n\n"
                   "N/A means not evaluated or not recorded.\n"
                   "Control time is steps / FPS, not wall-clock.\n"
                   "Manual viewer closure is excluded from rates.\n"
                   "Runtime errors count as task failures.",
                   fontsize=10, va="top", linespacing=1.7)
    modes = sorted({metric.get("evaluation_mode", "unknown") for metric in metrics})
    figure.suptitle(f"Mode: {', '.join(modes)}; recorded N = {summary['recorded_episodes']}; evaluated N = {summary['evaluated_episodes']}")
    _save_figure(figure, directory, "overview")
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(11, 3.2))
    axis.axis("off")
    _timing_table(axis, summary)
    axis.set_title("Measured computation time: pooled evaluated episodes")
    _save_figure(figure, directory, "computation_time")
    plt.close(figure)


def aggregate_timings(output, evaluated):
    """Pool trace samples; use sample-weighted episode means if traces are absent."""
    totals = {key: {"sum": 0., "samples": 0, "arrays": [], "complete": True} for key in TIMINGS}
    for metric in evaluated:
        trace_path = Path(metric.get("source_run_path", output)) / f"ep_{int(metric['episode']):04d}" / "trace.npz"
        trace = np.load(trace_path, allow_pickle=False) if trace_path.exists() else None
        try:
            for key, total in totals.items():
                array_key = f"{key}_ms"
                if trace is not None and array_key in trace:
                    samples = np.asarray(trace[array_key], dtype=float).reshape(-1)
                    samples = samples[np.isfinite(samples)]
                    total["arrays"].append(samples)
                    total["sum"] += float(np.sum(samples))
                    total["samples"] += len(samples)
                else:
                    count = int(metric.get(f"{key}_samples") or 0)
                    mean = metric.get(f"{key}_mean_ms")
                    if count and mean is not None:
                        total["sum"] += count * float(mean)
                        total["samples"] += count
                        total["complete"] = False
        finally:
            if trace is not None:
                trace.close()
    summary = {}
    for key, total in totals.items():
        count = total["samples"]
        summary[f"{key}_samples"] = count
        summary[f"{key}_mean_ms"] = total["sum"] / count if count else None
        summary[f"{key}_p95_ms"] = (float(np.percentile(np.concatenate(total["arrays"]), 95))
                                             if count and total["complete"] else None)
    return summary


def export_summary(output, episode_metrics, requested_episodes, *, plots=True):
    directory = Path(output) / "experiment"
    directory.mkdir(parents=True, exist_ok=True)
    metrics = _json_value(episode_metrics)
    first_fields = ["episode", "task", "evaluation_mode", "success", "reason", "control_steps", "control_duration_s"]
    fields = first_fields + sorted({key for metric in metrics for key in metric} - set(first_fields))
    with (directory / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                          for key, value in metric.items()} for metric in metrics)
    evaluated = [metric for metric in metrics if metric.get("reason") != "Viewer closed"]
    successes = sum(bool(metric.get("success", False)) for metric in evaluated)
    count = len(evaluated)
    durations = [metric["successful_completion_time_s"] for metric in evaluated
                 if metric["successful_completion_time_s"] is not None]
    obstacle_runs = [metric for metric in evaluated if isinstance(metric.get("contact_collision"), bool)]
    table_runs = [metric for metric in evaluated if isinstance(metric.get("robot_table_contact"), bool)]
    collisions = sum(metric["contact_collision"] for metric in obstacle_runs)
    safe_successes = sum(metric["safe_task_success"] for metric in obstacle_runs)
    table_contacts = sum(metric["robot_table_contact"] for metric in table_runs)
    summary = {
        "requested_episodes": requested_episodes,
        "recorded_episodes": len(metrics),
        "evaluated_episodes": count,
        "successes": successes,
        "failures": count - successes,
        "interrupted_episodes": len(metrics) - count,
        "success_rate": successes / count if count else None,
        "successful_completion_time_samples": len(durations),
        "successful_completion_time_mean_s": float(np.mean(durations)) if durations else None,
        "successful_completion_time_median_s": float(np.median(durations)) if durations else None,
        "collision_evaluated_episodes": len(obstacle_runs),
        "contact_collision_episodes": collisions,
        "contact_collision_rate": collisions / len(obstacle_runs) if obstacle_runs else None,
        "safe_successes": safe_successes,
        "safe_task_success_rate": safe_successes / len(obstacle_runs) if obstacle_runs else None,
        "robot_table_contact_evaluated_episodes": len(table_runs),
        "robot_table_contact_episodes": table_contacts,
        "robot_table_contact_rate": table_contacts / len(table_runs) if table_runs else None,
        "minimum_obstacle_clearance_m": None,
        "timing_definition": TIMING_DEFINITION,
        "safety_definition": SAFETY_DEFINITION,
        "denominator": "All recorded episodes except manual viewer closure; runtime errors count as failures.",
        "total_control_duration_s": sum(metric.get("control_duration_s", 0) for metric in metrics),
        "control_time_definition": TIME_DEFINITION,
    }
    modes = sorted({metric.get("evaluation_mode", "unknown") for metric in metrics})
    summary["evaluation_mode"] = modes[0] if len(modes) == 1 else modes
    summary.update(aggregate_timings(output, evaluated))
    _write_json(directory / "summary.json", summary)
    if plots:
        _plot_summary(directory, metrics, summary)
