"""Render a visual preview of equation (10) with one left brace."""

from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT = Path(__file__).with_name("equation10_left_brace_preview.png")

fig, ax = plt.subplots(figsize=(10.5, 4.1), dpi=220)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

lines = [
    r"$\mu=\mu_d\mu_g\mu_s,$",
    r"$\mu_d=\left[\dfrac{\delta-\max(d_{\min},0)}{\delta}\right]^2,$",
    r"$\mu_g=\operatorname{clip}\!\left(\dfrac{\rho-\rho_0}{\Delta\rho},0,1\right),$",
    r"$\mu_s=\operatorname{clip}\!\left(\dfrac{k}{K_0},0,1\right),\quad d_{\min}<\delta.$",
]

for y, line in zip((0.80, 0.61, 0.40, 0.19), lines):
    ax.text(0.48, y, line, ha="center", va="center", fontsize=22, color="black")

ax.text(0.155, 0.505, "{", ha="center", va="center", fontsize=160,
        fontfamily="STIXGeneral", fontweight="normal", color="black")
ax.text(0.925, 0.50, "(10)", ha="center", va="center", fontsize=21,
        fontfamily="serif", color="black")

fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.18, facecolor="white")
plt.close(fig)
print(OUTPUT)
