#!/usr/bin/env python3
"""Create PPT-ready APF versus VLSA tables and figures from matched runs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_TIMING_KEY = "safety_layer_time_ms"


def _scalar(data, key, default=np.nan):
    if key not in data:
        return default
    value = np.asarray(data[key])
    return value.item() if value.size == 1 else default


def _episode_row(path, algorithm):
    episode_index = int(path.stem.rsplit("_", 1)[-1])
    with np.load(path, allow_pickle=False) as data:
        if REQUIRED_TIMING_KEY not in data:
            raise ValueError(
                f"{path} lacks {REQUIRED_TIMING_KEY}; rerun with the instrumented script. "
                "VLSA qp_time_ms alone is not the same timing boundary."
            )
        latency = np.asarray(data[REQUIRED_TIMING_KEY], dtype=float).reshape(-1)
        latency = latency[np.isfinite(latency)]
        if not len(latency):
            raise ValueError(f"{path} contains no finite safety-layer timings")
        period = float(_scalar(data, "control_period_s"))
        if not np.isfinite(period) or period <= 0:
            raise ValueError(f"{path} lacks a valid control_period_s")
        tcp = np.asarray(data["tcp_pos"], dtype=float) if "tcp_pos" in data else np.empty((0, 3))
        tcp_path = float(np.linalg.norm(np.diff(tcp, axis=0), axis=1).sum()) if len(tcp) > 1 else 0.0
        intervention = np.asarray(data["intervention_norm"], dtype=float).reshape(-1)
        intervention = intervention[np.isfinite(intervention)]
        success = bool(_scalar(data, "episode_success", path.parent.name == "success"))
        collision = bool(_scalar(data, "episode_knockdown", path.parent.name == "fail"))
        initial_xy = (
            np.asarray(data["episode_initial_target_xy"], dtype=float).reshape(2)
            if "episode_initial_target_xy" in data else np.full(2, np.nan)
        )
        row = {
            "algorithm": algorithm,
            "episode": episode_index,
            "file": str(path),
            "success": int(success),
            "collision": int(collision),
            "initial_target_x": float(initial_xy[0]),
            "initial_target_y": float(initial_xy[1]),
            "steps": int(len(tcp)),
            "completion_sim_s": len(tcp) * period,
            "tcp_path_m": tcp_path,
            "latency_mean_ms": float(np.mean(latency)),
            "latency_p50_ms": float(np.percentile(latency, 50)),
            "latency_p95_ms": float(np.percentile(latency, 95)),
            "latency_p99_ms": float(np.percentile(latency, 99)),
            "latency_max_ms": float(np.max(latency)),
            "deadline_miss_pct": float(100 * np.mean(latency > period * 1000.0)),
            "peak_contact_force_n": float(_scalar(data, "episode_peak_contact_force")),
            "contact_impulse_ns": float(_scalar(data, "episode_contact_impulse")),
            "peak_joint_torque_nm": float(_scalar(data, "episode_peak_joint_torque")),
            "apf_compute_mean_ms": (
                float(np.nanmean(data["apf_time_ms"])) if "apf_time_ms" in data else np.nan
            ),
            "contact_compute_mean_ms": (
                float(np.nanmean(data["contact_controller_time_ms"]))
                if "contact_controller_time_ms" in data else np.nan
            ),
            "qp_compute_mean_ms": (
                float(np.nanmean(data["qp_time_ms"])) if "qp_time_ms" in data else np.nan
            ),
            "intervention_active_pct": float(100 * np.mean(intervention > 1e-6)) if len(intervention) else np.nan,
            "intervention_mean": float(np.mean(intervention)) if len(intervention) else np.nan,
            "intervention_max": float(np.max(intervention)) if len(intervention) else np.nan,
        }
        return row, latency, period * 1000.0


def load_run(run_dir, algorithm):
    paths = sorted(
        Path(run_dir).glob("**/data_ep_*.npz"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if not paths:
        raise ValueError(f"No data_ep_*.npz found under {run_dir}")
    indices = [int(path.stem.rsplit("_", 1)[-1]) for path in paths]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate episode indices under {run_dir}: {indices}")
    rows, latency_parts, deadlines = [], [], []
    for path in paths:
        row, latency, deadline = _episode_row(path, algorithm)
        rows.append(row)
        latency_parts.append(latency)
        deadlines.append(deadline)
    if not np.allclose(deadlines, deadlines[0]):
        raise ValueError(f"Inconsistent control periods under {run_dir}: {deadlines}")
    return rows, np.concatenate(latency_parts), deadlines[0]


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return 100 * (center - half), 100 * (center + half)


def latency_stats(name, values, deadline_ms):
    return {
        "algorithm": name,
        "samples": len(values),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
        "deadline_ms": deadline_ms,
        "deadline_miss_pct": float(100 * np.mean(values > deadline_ms)),
    }


def validate_paired_runs(proposed_rows, vlsa_rows):
    if len(proposed_rows) != len(vlsa_rows):
        raise ValueError(
            f"Paired comparison requires equal episode counts, got "
            f"{len(proposed_rows)} and {len(vlsa_rows)}"
        )
    if [row["episode"] for row in proposed_rows] != [row["episode"] for row in vlsa_rows]:
        raise ValueError("Episode indices do not match between paired runs")
    proposed_xy = np.asarray([[row["initial_target_x"], row["initial_target_y"]]
                              for row in proposed_rows])
    vlsa_xy = np.asarray([[row["initial_target_x"], row["initial_target_y"]]
                          for row in vlsa_rows])
    if not np.isfinite(proposed_xy).all() or not np.isfinite(vlsa_xy).all():
        raise ValueError(
            "Runs lack episode_initial_target_xy; recollect both runs with matching --eval_seed"
        )
    if not np.allclose(proposed_xy, vlsa_xy, atol=1e-12, rtol=0):
        raise ValueError("Initial target positions are not paired episode by episode")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def method_summary(rows, latency_stat):
    successful = [row for row in rows if row["success"]]

    def mean_of(key, sample):
        values = np.asarray([row[key] for row in sample], dtype=float)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if len(values) else np.nan

    successes = sum(row["success"] for row in rows)
    collisions = sum(row["collision"] for row in rows)
    sr_low, sr_high = wilson_interval(successes, len(rows))
    cr_low, cr_high = wilson_interval(collisions, len(rows))
    summary = {
        "algorithm": rows[0]["algorithm"],
        "episodes": len(rows),
        "successes": successes,
        "success_rate_pct": 100 * successes / len(rows),
        "success_ci95_low_pct": sr_low,
        "success_ci95_high_pct": sr_high,
        "collisions": collisions,
        "collision_rate_pct": 100 * collisions / len(rows),
        "collision_ci95_low_pct": cr_low,
        "collision_ci95_high_pct": cr_high,
    }
    for key in (
        "completion_sim_s",
        "tcp_path_m",
        "peak_contact_force_n",
        "contact_impulse_ns",
        "peak_joint_torque_nm",
    ):
        summary[f"{key}_all_mean"] = mean_of(key, rows)
        summary[f"{key}_success_mean"] = mean_of(key, successful)
    success_impulses = np.asarray(
        [row["contact_impulse_ns"] for row in successful], dtype=float
    )
    success_impulses = success_impulses[np.isfinite(success_impulses)]
    summary["contact_impulse_success_std"] = (
        float(np.std(success_impulses)) if len(success_impulses) else np.nan
    )
    for key in (
        "mean_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms",
        "deadline_ms", "deadline_miss_pct",
    ):
        summary[f"latency_{key}"] = latency_stat[key]
    return summary


def plot_latency(stats, values, output_dir):
    names = [item["algorithm"] for item in stats]
    percentiles = ["p50_ms", "p95_ms", "p99_ms"]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.23
    for index, key in enumerate(percentiles):
        vals = [item[key] for item in stats]
        bars = axes[0].bar(x + (index - 1) * width, vals, width, label=key[:3].upper())
        axes[0].bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Safety-layer latency (ms)")
    axes[0].set_title("Matched end-to-end safety-layer latency")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    for name, sample in zip(names, values):
        ordered = np.sort(sample)
        axes[1].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=name, linewidth=2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Latency (ms, log scale)")
    axes[1].set_ylabel("Empirical CDF")
    axes[1].set_title("Latency distribution and tail")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "latency_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_task_metrics(rows, output_dir):
    algorithms = list(dict.fromkeys(row["algorithm"] for row in rows))
    groups = [[row for row in rows if row["algorithm"] == name] for name in algorithms]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    x = np.arange(len(algorithms))
    width = 0.32
    for offset, (key, label) in zip((-width / 2, width / 2), (("success", "Success"), ("collision", "Collision"))):
        rates, low_err, high_err = [], [], []
        for group in groups:
            count = sum(row[key] for row in group)
            rate = 100 * count / len(group)
            low, high = wilson_interval(count, len(group))
            rates.append(rate)
            low_err.append(rate - low)
            high_err.append(high - rate)
        axes[0, 0].bar(x + offset, rates, width, label=label,
                       yerr=np.array([low_err, high_err]), capsize=4)
    axes[0, 0].set_xticks(x, algorithms)
    axes[0, 0].set_ylim(0, 110)
    axes[0, 0].set_ylabel("Rate (%)")
    axes[0, 0].set_title("Outcome rate with 95% Wilson interval")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.25)

    metrics = [
        ("completion_sim_s", "Successful completion time (sim s)"),
        ("tcp_path_m", "Successful TCP path length (m)"),
        ("contact_impulse_ns", "Successful contact impulse (N·s)"),
    ]
    for axis, (key, title) in zip(axes.flat[1:], metrics):
        samples = [np.asarray([row[key] for row in group if row["success"] and np.isfinite(row[key])])
                   for group in groups]
        if all(len(sample) for sample in samples):
            axis.boxplot(samples, tick_labels=algorithms, showmeans=True)
        else:
            missing = ", ".join(name for name, sample in zip(algorithms, samples) if not len(sample))
            axis.text(0.5, 0.5, f"No successful samples: {missing}",
                      ha="center", va="center", transform=axis.transAxes)
            axis.set_xticks([])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "task_metrics_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_average_metrics(summaries, output_dir):
    names = [row["algorithm"] for row in summaries]
    metrics = [
        ("peak_contact_force_n", "Peak contact force (N)"),
        ("contact_impulse_ns", "Contact impulse (N·s)"),
        ("peak_joint_torque_nm", "Peak joint torque (N·m)"),
    ]
    x = np.arange(len(names))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, (key, title) in zip(axes, metrics):
        all_values = [row[f"{key}_all_mean"] for row in summaries]
        success_values = [row[f"{key}_success_mean"] for row in summaries]
        bars_all = axis.bar(x - width / 2, all_values, width, label="All episodes")
        bars_success = axis.bar(x + width / 2, success_values, width, label="Successful only")
        axis.bar_label(bars_all, fmt="%.2f", fontsize=8, padding=2)
        axis.bar_label(bars_success, fmt="%.2f", fontsize=8, padding=2)
        axis.set_xticks(x, names)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "average_metrics_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_compute_breakdown(rows, stats, output_dir):
    names = [item["algorithm"] for item in stats]
    groups = [[row for row in rows if row["algorithm"] == name] for name in names]

    def component_mean(group, key):
        values = np.asarray([row[key] for row in group], dtype=float)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if len(values) else 0.0

    apf = np.asarray([component_mean(group, "apf_compute_mean_ms") for group in groups])
    contact = np.asarray([component_mean(group, "contact_compute_mean_ms") for group in groups])
    qp = np.asarray([component_mean(group, "qp_compute_mean_ms") for group in groups])
    totals = np.asarray([item["mean_ms"] for item in stats])
    other = np.maximum(totals - apf - contact - qp, 0.0)
    components = {
        "APF": apf,
        "Contact protection": contact,
        "CBF/QP solve": qp,
        "Kinematics + adaptation": other,
    }
    fig, axis = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(names))
    for label, values in components.items():
        values = np.asarray(values)
        if np.any(values > 1e-9):
            axis.bar(names, values, bottom=bottom, label=label)
            bottom += values
    axis.set_ylabel("Mean compute time per control period (ms)")
    axis.set_title("Safety computation breakdown")
    axis.bar_label(axis.containers[-1], labels=[f"{value:.3f}" for value in bottom], padding=3)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "compute_breakdown.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposed-run", "--apf-run", dest="proposed_run", type=Path)
    parser.add_argument("--case8-run", type=Path, help="Paper ablation group B8")
    parser.add_argument("--case11-run", type=Path, help="Paper ablation group B11 (form I)")
    parser.add_argument("--case12-run", type=Path, help="Paper ablation group B12")
    parser.add_argument("--vlsa-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.case8_run or args.case11_run or args.case12_run:
        if not args.case8_run or not args.case11_run or not args.case12_run or args.proposed_run:
            parser.error(
                "Use --case8-run, --case11-run and --case12-run together, without --proposed-run"
            )
        run_specs = [
            ("B8", args.case8_run),
            ("B11", args.case11_run),
            ("B12", args.case12_run),
        ]
    elif args.proposed_run:
        run_specs = [("Proposed", args.proposed_run)]
    else:
        parser.error(
            "Specify --proposed-run or all of --case8-run, --case11-run and --case12-run"
        )
    run_specs.append(("VLSA", args.vlsa_run))

    loaded = [(name, *load_run(path, name)) for name, path in run_specs]
    reference_rows = loaded[0][1]
    for _, method_rows, _, _ in loaded[1:]:
        validate_paired_runs(reference_rows, method_rows)
    rows = [row for _, method_rows, _, _ in loaded for row in method_rows]
    stats = [latency_stats(name, latency, deadline)
             for name, _, latency, deadline in loaded]
    summaries = [method_summary(method_rows, stat)
                 for (_, method_rows, _, _), stat in zip(loaded, stats)]
    write_csv(args.output_dir / "episode_metrics.csv", rows)
    write_csv(args.output_dir / "latency_summary.csv", stats)
    write_csv(args.output_dir / "method_summary.csv", summaries)
    plot_latency(stats, [latency for _, _, latency, _ in loaded], args.output_dir)
    plot_task_metrics(rows, args.output_dir)
    plot_average_metrics(summaries, args.output_dir)
    plot_compute_breakdown(rows, stats, args.output_dir)
    print(f"Saved comparison tables and figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
