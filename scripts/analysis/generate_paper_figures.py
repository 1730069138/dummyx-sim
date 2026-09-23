"""Generate the paper figures that can be reproduced from saved experiment logs."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = PROJECT_ROOT / "docs" / "paper"
DEFAULT_OUTPUT = PAPER_DIR / "assets" / "figures"
ACTION_DT = 0.05
BOX_CENTER = np.array([0.0, -0.1, 0.24])

DEFAULT_LOGS = {
    "baseline_contact": PROJECT_ROOT
    / "recordings/run_20260913_235341_none_Case1_no_obs_no_apf_no_cont/success/data_ep_001.npz",
    "contact_i": PROJECT_ROOT
    / "recordings/run_20260914_111437_admittance_Case5_no_obs_no_apf_cont/success/data_ep_001.npz",
    "contact_ii": PROJECT_ROOT
    / "recordings/run_20260914_111437_impedance_Case5_no_obs_no_apf_cont/success/data_ep_001.npz",
    "pi0": PROJECT_ROOT
    / "recordings/run_20260914_110311_none_Case3_obs_no_apf_no_cont/success/data_ep_001.npz",
    "pace": PROJECT_ROOT
    / "recordings/run_20260914_110333_impedance_Case8_obs_apf_cont/success/data_ep_001.npz",
    "vlsa": PROJECT_ROOT
    / "recordings/run_20260914_111531_vlsa_oracle_obs1/fail/data_ep_001.npz",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Noto Sans CJK SC", "WenQuanYi Micro Hei"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def load_npz(path: Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment log: {path}")
    return np.load(path, allow_pickle=True)


def extract_original_framework(docx: Path, output: Path) -> None:
    """Restore the original embedded Figure 1 without modification."""
    with zipfile.ZipFile(docx) as archive:
        output.write_bytes(archive.read("word/media/image1.png"))
    print(output)


def force_norm(data: np.lib.npyio.NpzFile) -> np.ndarray:
    return np.linalg.norm(
        np.column_stack([data["f_real_x"], data["f_real_y"], data["f_real_z"]]),
        axis=1,
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def compose_method_comparison(output: Path) -> None:
    """Compose Figure 3 from the nine original simulator screenshots."""
    rows = [
        (
            "π0 无约束基线",
            [
                "fig3_pi0_start_stable.png",
                "fig3_pi0_collision_stable.png",
                "fig3_pi0_end_stable.png",
            ],
        ),
        (
            "VLSA/AEGIS",
            [
                "fig3_vlsa_start_stable.png",
                "fig3_vlsa_constraint_stable.png",
                "fig3_vlsa_end_stable.png",
            ],
        ),
        (
            "PACE（形式 II）",
            [
                "fig3_pace_start_stable.png",
                "fig3_pace_avoid_stable.png",
                "fig3_pace_end_stable.png",
            ],
        ),
    ]
    source_dir = output.parent
    cell_size = 512
    canvas = Image.new("RGB", (3 * cell_size, 3 * cell_size), "white")
    for row_idx, (_, filenames) in enumerate(rows):
        for col_idx, filename in enumerate(filenames):
            with Image.open(source_dir / filename) as source:
                panel = source.convert("RGB").resize(
                    (cell_size, cell_size), Image.LANCZOS
                )
            canvas.paste(panel, (col_idx * cell_size, row_idx * cell_size))

    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.truetype(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 28
    )
    header_height = 54
    for row_idx, (label, _) in enumerate(rows):
        y0 = row_idx * cell_size
        draw.rectangle((0, y0, canvas.width, y0 + header_height), fill=(7, 24, 34, 225))
        draw.text((18, y0 + 10), label, font=font, fill=(255, 255, 255, 255))

    canvas.save(output)
    print(output)


def plot_activation_timing(data_path: Path, output: Path) -> None:
    data = load_npz(data_path)
    steps = np.asarray(data["steps"], dtype=float)
    time_s = steps * ACTION_DT
    tcp = np.asarray(data["tcp_pos"])
    distance_to_box = np.linalg.norm(tcp - BOX_CENTER, axis=1)
    safety_margin = np.clip(0.025 + 0.4 * distance_to_box, 0.025, 0.08)
    contact_force = force_norm(data)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.5), sharex=True)
    axes[0].plot(time_s, data["min_dist"], color="#2563a5", lw=1.7, label=r"$d_{\min}$")
    axes[0].plot(time_s, safety_margin, color="#d97706", lw=1.4, ls="--", label=r"$\delta$")
    axes[0].axhline(0, color="#444444", lw=0.8)
    axes[0].set_ylabel("距离 / m")
    axes[0].legend(ncol=2, loc="upper right")

    axes[1].plot(time_s, data["influence"], color="#17843b", lw=1.7, label=r"避障权重 $\mu$")
    axes[1].fill_between(
        time_s,
        0,
        np.asarray(data["is_apf_active"], dtype=float),
        color="#86c995",
        alpha=0.25,
        label="几何避障激活",
    )
    axes[1].set_ylabel("权重 / 状态")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(ncol=2, loc="upper right")

    axes[2].plot(time_s, contact_force, color="#b4232c", lw=1.5, label=r"$\Vert f_c\Vert$")
    contact_active = np.asarray(data["is_contact"], dtype=float)
    ymax = max(float(np.nanmax(contact_force)) * 1.08, 1.0)
    axes[2].fill_between(
        time_s,
        0,
        contact_active * ymax,
        color="#ef9a9a",
        alpha=0.22,
        label="接触调节激活",
    )
    axes[2].set_ylabel("接触力 / N")
    axes[2].set_xlabel("时间 / s")
    axes[2].set_ylim(0, ymax)
    axes[2].legend(ncol=2, loc="upper right")

    fig.suptitle("PACE 两类执行时修正的典型激活时序", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output)


def plot_contact_force(paths: dict[str, Path], output: Path) -> None:
    series = [
        ("B1  无接触调节", load_npz(paths["baseline_contact"]), "#555555", "-"),
        ("B3  形式 I", load_npz(paths["contact_i"]), "#2563a5", "-"),
        ("B4  形式 II", load_npz(paths["contact_ii"]), "#b4232c", "--"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for label, data, color, linestyle in series:
        time_s = np.asarray(data["steps"], dtype=float) * ACTION_DT
        ax.plot(time_s, force_norm(data), color=color, ls=linestyle, lw=1.35, label=label)
    ax.axhline(0.5, color="#d97706", ls=":", lw=1.1, label="0.5 N 力死区")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel(r"合成接触力 $\Vert f_c\Vert$ / N")
    ax.set_title("不同接触调节形式的代表性回合对比", fontweight="bold")
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output)


def plot_trajectories(paths: dict[str, Path], output: Path) -> None:
    pi0 = load_npz(paths["pi0"])
    pace = load_npz(paths["pace"])
    vlsa = load_npz(paths["vlsa"])
    tracks = [
        (r"$\pi_0$ 无约束", np.asarray(pi0["tcp_pos"]), "#555555", ":"),
        ("VLSA/AEGIS", np.asarray(vlsa["tcp_pos"]), "#d97706", "--"),
        ("PACE（形式 II）", np.asarray(pace["tcp_pos"]), "#2563a5", "-"),
    ]
    obstacle = np.asarray(vlsa["obstacle_center"])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for label, xyz, color, linestyle in tracks:
        axes[0].plot(xyz[:, 0], xyz[:, 1], color=color, ls=linestyle, lw=1.7, label=label)
        axes[0].scatter(xyz[0, 0], xyz[0, 1], color=color, marker="o", s=20)
        axes[0].scatter(xyz[-1, 0], xyz[-1, 1], color=color, marker="x", s=35)
        axes[1].plot(xyz[:, 0], xyz[:, 2], color=color, ls=linestyle, lw=1.7, label=label)
        axes[1].scatter(xyz[0, 0], xyz[0, 2], color=color, marker="o", s=20)
        axes[1].scatter(xyz[-1, 0], xyz[-1, 2], color=color, marker="x", s=35)

    axes[0].add_patch(Circle(obstacle[:2], 0.025, color="#c62828", alpha=0.25, label="障碍物"))
    axes[0].add_patch(Rectangle((-0.08, -0.18), 0.16, 0.16, color="#6b8e23", alpha=0.16, label="放置区域"))
    axes[0].set_xlabel("x / m")
    axes[0].set_ylabel("y / m")
    axes[0].set_title("俯视轨迹")
    axes[0].axis("equal")

    axes[1].add_patch(
        Rectangle(
            (obstacle[0] - 0.025, obstacle[2] - 0.08),
            0.05,
            0.16,
            color="#c62828",
            alpha=0.25,
            label="障碍物",
        )
    )
    axes[1].axhspan(0.20, 0.28, color="#6b8e23", alpha=0.12, label="放置高度带")
    axes[1].set_xlabel("x / m")
    axes[1].set_ylabel("z / m")
    axes[1].set_title("侧视轨迹")
    axes[1].legend(loc="best")
    fig.suptitle(r"$\pi_0$、VLSA/AEGIS 与 PACE 的 TCP 轨迹", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--docx", type=Path, default=PAPER_DIR / "9.6小论文.docx")
    args = parser.parse_args()

    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extract_original_framework(args.docx, args.output_dir / "fig1_pace_framework.png")
    plot_activation_timing(DEFAULT_LOGS["pace"], args.output_dir / "fig2_activation_timing.png")
    compose_method_comparison(args.output_dir / "fig3_method_comparison_form_ii.png")
    plot_contact_force(DEFAULT_LOGS, args.output_dir / "fig4_contact_force.png")
    plot_trajectories(DEFAULT_LOGS, args.output_dir / "fig5_tcp_trajectories.png")


if __name__ == "__main__":
    main()
