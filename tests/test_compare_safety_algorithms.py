import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_safety_algorithms",
    ROOT / "scripts" / "tools" / "compare_safety_algorithms.py",
)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def write_episode(self, root, timing=True):
        folder = root / "success"
        folder.mkdir(parents=True)
        fields = {
            "control_period_s": np.array(0.05),
            "episode_success": np.array(True),
            "episode_knockdown": np.array(False),
            "episode_initial_target_xy": np.array([0.28, -0.10]),
            "episode_peak_contact_force": np.array(2.0),
            "episode_contact_impulse": np.array(0.3),
            "episode_peak_joint_torque": np.array(1.5),
            "tcp_pos": np.array([[0, 0, 0], [0.03, 0.04, 0]]),
            "intervention_norm": np.array([0.0, 0.2]),
            "apf_time_ms": np.array([0.1, 0.2]),
            "contact_controller_time_ms": np.array([0.8, 0.9]),
            "qp_time_ms": np.array([0.7, 0.8]),
        }
        if timing:
            fields["safety_layer_time_ms"] = np.array([1.0, 2.0, 60.0])
        np.savez(folder / "data_ep_001.npz", **fields)

    def test_load_and_latency_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            self.write_episode(run)
            rows, latency, deadline = comparison.load_run(run, "APF")
            self.assertEqual(rows[0]["success"], 1)
            self.assertAlmostEqual(rows[0]["tcp_path_m"], 0.05)
            self.assertAlmostEqual(rows[0]["intervention_active_pct"], 50.0)
            stats = comparison.latency_stats("APF", latency, deadline)
            self.assertAlmostEqual(stats["deadline_miss_pct"], 100 / 3)
            summary = comparison.method_summary(rows, stats)
            self.assertEqual(summary["episodes"], 1)
            self.assertEqual(summary["success_rate_pct"], 100.0)
            output = Path(temp) / "plots"
            output.mkdir()
            comparison.plot_latency([stats, stats], [latency, latency], output)
            comparison.plot_task_metrics(rows + [{**rows[0], "algorithm": "VLSA"}], output)
            comparison.plot_average_metrics(
                [summary, {**summary, "algorithm": "VLSA"}], output
            )
            plot_rows = [{**rows[0], "algorithm": "Proposed"}, {**rows[0], "algorithm": "VLSA"}]
            comparison.plot_compute_breakdown(plot_rows, [stats, stats], output)
            self.assertTrue((output / "latency_comparison.png").is_file())
            self.assertTrue((output / "task_metrics_comparison.png").is_file())
            self.assertTrue((output / "average_metrics_comparison.png").is_file())
            self.assertTrue((output / "compute_breakdown.png").is_file())
            comparison.validate_paired_runs(rows, [{**rows[0], "algorithm": "VLSA"}])

    def test_old_incomparable_timing_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            self.write_episode(run, timing=False)
            with self.assertRaisesRegex(ValueError, "rerun"):
                comparison.load_run(run, "VLSA")


if __name__ == "__main__":
    unittest.main()
