#!/usr/bin/env python3
"""
Controller-architecture figures.

  (a) nested donut : inner ring = architecture, outer ring = brand
  (b) pie          : broader robotics, External vs Integrated
  table            : per-architecture characteristics, full width

Outputs PDF (vector, Type-42 fonts) + PNG to OUT_DIR.
Run:  python3 robot_arch_pies.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "vis"

# ============================================================================
# 1. CONFIG  —  edit everything here
# ============================================================================

# ---- colours ---------------------------------------------------------------
C_EXT = "#F2B705"       # External controller   (yellow)
C_INT = "#2E6FB7"       # Integrated controller (blue)

# outer-ring brand colours; keep the External ones inside the yellow family
C_BRAND = {
    "Franka":     "#D99A00",
    "ABB":        "#E8AC10",
    "UR":         "#F2B705",
    "KUKA":       "#F6C93F",
    "Fanuc":      "#F7D163",
    "RAVEN II":   "#F8D98A",
    "Others (E)": "#FBE0A0",
    "Others (I)": C_INT,     # the only Integrated slice
}

# ---- fonts (points) --------------------------------------------------------
FONT_FAMILY = "serif"
FONT_NAME   = "DejaVu Serif"     # e.g. "Times New Roman", "Nimbus Roman"
SCALE       = 3.0                # global multiplier: all sizes below x SCALE

F_LABEL   = 8.5 * SCALE          # brand labels around the donut
F_PCT     = 8.0 * SCALE          # percentages inside the wedges
F_PCT_SM  = 0.70                 # shrink factor for the narrow inner wedge
F_TITLE   = 9.5 * SCALE          # panel titles (a) / (b)
F_LEGEND  = 8.5 * SCALE          # legend (standalone (a) only)
F_TABLE   = 48                   # table; auto-shrinks if a row overflows

# ---- geometry --------------------------------------------------------------
START      = 90                  # first wedge starts at 12 o'clock
R_OUT, W_OUT = 1.00, 0.30        # outer (brand) ring: radius, thickness
R_IN,  W_IN  = 0.66, 0.30        # inner (architecture) ring
R_PIE        = 1.00              # panel (b) pie radius -> equals donut size
LABEL_X      = 1.42              # x of the leader-line labels
LABEL_GAP    = 0.46              # min vertical gap between stacked labels
LABEL_LIM    = 1.34              # labels never leave +/- this y
EDGE_LW      = 2.0               # white separator between wedges

# ---- data ------------------------------------------------------------------
N_TOTAL = 83
INNER = [("External controller", 84.6, C_EXT),
         ("Integrated controller", 15.4, C_INT)]
BRAND = [("Franka", 26.5), ("ABB", 12.0), ("UR", 12.0), ("KUKA", 4.8),
         ("Fanuc", 4.8), ("RAVEN II", 4.8), ("Others (E)", 19.6),
         ("Others (I)", 15.4)]
PIE_B = [("External", 66.96, C_EXT), ("Integrated", 33.04, C_INT)]

COLS = ["Architecture", "Cost", "Payload", "Repeatability", "Communication",
        "Brand "]
ROWS = [
    ("Integrated", C_INT, ["< \\$10k", "< 5 kg", "0.01\u20130.05 mm", "CAN / TCP/IP",
                           "AgileX, ARX, YAM, ViperX,\nxArm Lite6, RealMan"]),
    ("External",   C_EXT, ["> \\$10k", "> 3 kg", "< 0.01 mm", "TCP/IP",
                           "Franka, Kinova, xArm, UR,\nABB, FANUC"]),
]

TITLE_A = "(a) This SoK"
TITLE_B = "(b) Broader robotics"

# ============================================================================
# 2. RENDERING
# ============================================================================

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.serif": [FONT_NAME],
    "font.size": 9 * SCALE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

XLIM_A, YLIM = 2.55, 1.55        # (a) data window; (b) shares YLIM -> equal circles


def _repel(items, gap):
    items.sort(key=lambda t: -t[0])
    for i in range(1, len(items)):
        if items[i - 1][0] - items[i][0] < gap:
            items[i][0] = items[i - 1][0] - gap
    return items


def draw_A(ax):
    """Nested donut: brand ring outside, architecture ring inside."""
    wedges, _ = ax.pie([v for _, v in BRAND], radius=R_OUT,
                       colors=[C_BRAND[n] for n, _ in BRAND],
                       startangle=START, counterclock=False,
                       wedgeprops=dict(width=W_OUT, edgecolor="white",
                                       linewidth=EDGE_LW))
    inner, _ = ax.pie([v for _, v, _ in INNER], radius=R_IN,
                      colors=[c for _, _, c in INNER],
                      startangle=START, counterclock=False,
                      wedgeprops=dict(width=W_IN, edgecolor="white",
                                      linewidth=EDGE_LW))

    for (_, val, _), w in zip(INNER, inner):
        a = np.deg2rad((w.theta1 + w.theta2) / 2)
        small = val <= 50
        r = (R_IN - W_IN / 2) * (0.93 if small else 1.0)
        ax.text(r * np.cos(a), r * np.sin(a), f"{val:.1f}%",
                ha="center", va="center", color="white", weight="bold",
                fontsize=F_PCT * (F_PCT_SM if small else 1.0))

    right, left, ang = [], [], {}
    for i, w in enumerate(wedges):
        a = np.deg2rad((w.theta1 + w.theta2) / 2)
        ang[i] = a
        (right if np.cos(a) >= 0 else left).append([np.sin(a) * 1.05, i])

    placed = {}
    for side in (right, left):
        pts = _repel(side, LABEL_GAP)
        ys = [p[0] for p in pts]
        shift = 0.0
        if min(ys) < -LABEL_LIM:
            shift = -LABEL_LIM - min(ys)
        if max(ys) + shift > LABEL_LIM:
            shift -= max(ys) + shift - LABEL_LIM
        for y, i in pts:
            placed[i] = y + shift

    for i, (name, val) in enumerate(BRAND):
        a = ang[i]
        xs = 1.0 if np.cos(a) >= 0 else -1.0
        ax.annotate(f"{name}\n{val:.1f}%",
                    xy=(np.cos(a) * (R_OUT - W_OUT / 2),
                        np.sin(a) * (R_OUT - W_OUT / 2)),
                    xytext=(xs * LABEL_X, placed[i]),
                    ha="left" if xs > 0 else "right", va="center",
                    fontsize=F_LABEL, linespacing=1.15,
                    arrowprops=dict(arrowstyle="-", color="0.45", lw=1.2,
                                    connectionstyle="arc3,rad=0.05"))

    ax.text(0, 0, f"n = {N_TOTAL}", ha="center", va="center",
            fontsize=F_LABEL, color="0.25")
    ax.set_aspect("equal")
    ax.set_xlim(-XLIM_A, XLIM_A)
    ax.set_ylim(-YLIM, YLIM)


def draw_B(ax, span=1.28):
    """Flat pie; span only pads the x window, the circle size follows YLIM."""
    ax.pie([v for _, v, _ in PIE_B], radius=R_PIE,
           colors=[c for _, _, c in PIE_B], startangle=START, counterclock=False,
           wedgeprops=dict(edgecolor="white", linewidth=EDGE_LW),
           autopct=lambda p: f"{p:.2f}%", pctdistance=0.60,
           textprops=dict(color="white", fontsize=F_PCT, weight="bold"))
    ax.set_aspect("equal")
    ax.set_xlim(-span, span)
    ax.set_ylim(-YLIM, YLIM)


def draw_table(ax, fs=F_TABLE):
    """Booktabs-style block; column widths measured from the rendered text."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    axw = ax.get_window_extent().width

    cells = [[r[0] for r in ROWS]] + [[r[2][j] for r in ROWS] for j in range(5)]
    heads = ["Architecture"] + COLS[1:5] + ["Representative"]

    def w_of(txt, bold, size):
        t = ax.text(0, -5, txt, fontsize=size, weight="bold" if bold else "normal")
        w = t.get_window_extent(rend).width
        t.remove()
        return w

    for _ in range(20):
        widths = []
        for j in range(6):
            w = w_of(heads[j], True, fs)
            for c in cells[j]:
                for line in str(c).split("\n"):
                    w = max(w, w_of(line, False, fs))
            widths.append(w / axw + 0.014)
        widths[0] += 0.032                       # colour swatch
        if sum(widths) <= 1.0:
            break
        fs *= 0.96

    x, acc = [], 0.0
    for w in widths:
        x.append(acc)
        acc += w
    slack = (1.0 - acc) / 5.0 if acc < 1.0 else 0.0
    x = [xi + i * slack for i, xi in enumerate(x)]

    y_head, y_rows = 0.90, [0.64, 0.42]
    rule = dict(color="0.15", lw=1.8, solid_capstyle="butt")
    ax.plot([0, 1], [0.99, 0.99], **rule)
    for j, c in enumerate(COLS):
        ax.text(x[j], y_head,
                "Brand" if j == 5 else c,
                ha="left", va="center", fontsize=fs, weight="bold",
                linespacing=1.15)
    ax.plot([0, 1], [0.8, 0.8], color="0.15", lw=1.1)
    for (name, col, cs), yr in zip(ROWS, y_rows):
        ax.add_patch(Rectangle((0.0, yr - 0.05), 0.026, 0.10,
                               facecolor=col, edgecolor="none", clip_on=False))
        ax.text(0.038, yr, name, ha="left", va="center", fontsize=fs)
        for j, cell in enumerate(cs, start=1):
            ax.text(x[j], yr, cell, ha="left", va="center", fontsize=fs,
                    linespacing=1.25)
    ax.plot([0, 1], [0.27, 0.27], **rule)


LEG = [Patch(facecolor=C_EXT, edgecolor="white", label="External controller"),
       Patch(facecolor=C_INT, edgecolor="white", label="Integrated controller")]


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- combined: (a) + (b) on top, table across both columns ------------------
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0], height_ratios=[1.68, 1.0],
                      wspace=0.0, hspace=0.02, left=0.01, right=0.99,
                      top=0.94, bottom=0.02)
axA, axB, axT = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                 fig.add_subplot(gs[1, :]))
draw_A(axA)
draw_B(axB)
draw_table(axT)
fig.text(0.33, 0.965, TITLE_A, ha="center", va="center", fontsize=F_TITLE)
fig.text(0.83, 0.965, TITLE_B, ha="center", va="center", fontsize=F_TITLE)
save(fig, "pie_AB_combined")

# ---- standalone (a) --------------------------------------------------------
figA = plt.figure(figsize=(10, 7))
draw_A(figA.add_axes([0.01, 0.10, 0.98, 0.88]))
figA.legend(handles=LEG, loc="lower center", ncol=2, frameon=False,
            fontsize=F_LEGEND, handlelength=1.3, columnspacing=2.5,
            bbox_to_anchor=(0.5, 0.0))
save(figA, "pie_A_brand_architecture")

# ---- standalone (b) + table ------------------------------------------------
figB = plt.figure(figsize=(12, 8))
gsb = figB.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.02,
                        left=0.01, right=0.99, top=0.98, bottom=0.02)
draw_B(figB.add_subplot(gsb[0]), span=2.4)
draw_table(figB.add_subplot(gsb[1]))
save(figB, "pie_B_broader_robotics")

print(f"written to {OUT_DIR}")