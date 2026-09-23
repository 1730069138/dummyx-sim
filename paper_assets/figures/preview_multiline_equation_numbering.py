"""Generate a preview of per-line subequation numbering for Equation (10)."""

import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(8.6, 3.7), dpi=180)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

lines = [
    (r"$\mu=\mu_d\mu_g\mu_s,$", "(10a)"),
    (r"$\mu_d=\left[\dfrac{\delta-\max(d_{\min},0)}{\delta}\right]^2,$", "(10b)"),
    (r"$\mu_g=\mathrm{clip}\!\left(\dfrac{\rho-\rho_0}{\Delta\rho},0,1\right),$", "(10c)"),
    (r"$\mu_s=\mathrm{clip}\!\left(\dfrac{k}{K_0},0,1\right),\quad d_{\min}<\delta.$", "(10d)"),
]

for y, (formula, number) in zip((0.82, 0.62, 0.40, 0.18), lines):
    ax.text(0.48, y, formula, ha="center", va="center", fontsize=18, color="black")
    ax.text(0.94, y, number, ha="right", va="center", fontsize=15, color="black")

fig.savefig(
    "/home/jun/dummyx-sim/paper_assets/figures/preview_multiline_equation_numbering.png",
    bbox_inches="tight",
    pad_inches=0.18,
    facecolor="white",
)
plt.close(fig)
