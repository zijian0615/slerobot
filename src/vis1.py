#!/usr/bin/env python3
"""
Attack / Defense x Implicit / Explicit publication counts, 2016-2026.

Design: implicit series = solid + markers; explicit series = dashed + light fill.
Values were read off the supplied raster figure -- verify DATA below.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "vis"

# ============================================================================
# 1. CONFIG
# ============================================================================
YEARS = list(range(2016, 2027))

DATA = {                       # (label, values)
    "attack_implicit":  [0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 12],
    "attack_explicit":  [1, 2, 5, 3, 1, 0, 1, 2, 4, 9, 0],
    "defense_implicit": [0, 0, 0, 0, 0, 0, 0, 1, 0, 12, 1],
    "defense_explicit": [0, 1, 1, 3, 3, 0, 1, 0, 3, 4, 0],
}

# implicit series only start once the term is used explicitly; earlier years hidden
IMPLICIT_START = {"attack_implicit": 2023, "defense_implicit": 2022}

C_ATK = "#C0355C"              # Attack   (crimson)
C_DEF = "#2E6FB7"             # Defense  (blue)
FILL_ALPHA = 0.13              # explicit-series area fill
GRID_COLOR = "#D9D9D9"

FONT_FAMILY, FONT_NAME = "serif", "DejaVu Serif"
SCALE = 3.0
F_TICK   = 9.5 * SCALE
F_AXLBL  = 10.5 * SCALE
F_LEGEND = 11.0 * SCALE
F_ANNOT  = 7.0 * SCALE

LW_SOLID, LW_DASH = 4.0, 3.4
MS = 17                        # marker size
DASH = (0, (6, 3))

YLABEL = "Number of papers"
XLABEL = None                  # e.g. "Year"
YMAX, YSTEP = 13, 2
PARTIAL_YEAR = 2026            # shade + annotate; set None to disable
FIGSIZE = (15, 7.6)

# ============================================================================
# 2. RENDER
# ============================================================================
plt.rcParams.update({
    "font.family": FONT_FAMILY, "font.serif": [FONT_NAME],
    "font.size": 9 * SCALE, "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=FIGSIZE)
x = np.array(YEARS)

# explicit series: dashed + fill
for key, col in (("attack_explicit", C_ATK), ("defense_explicit", C_DEF)):
    y = np.array(DATA[key])
    ax.fill_between(x, 0, y, color=col, alpha=FILL_ALPHA, lw=0, zorder=1)
    ax.plot(x, y, color=col, lw=LW_DASH, ls=DASH, zorder=3,
            solid_capstyle="round", dash_capstyle="round")

# implicit series: solid + markers
for key, col, mk in (("attack_implicit", C_ATK, "o"),
                     ("defense_implicit", C_DEF, "s")):
    y = np.array(DATA[key], dtype=float)
    y[x < IMPLICIT_START.get(key, YEARS[0])] = np.nan
    ax.plot(x, y, color=col, lw=LW_SOLID, marker=mk, ms=MS,
            mfc="white", mew=3.0, mec=col, zorder=4, clip_on=False)

if PARTIAL_YEAR:
    ax.axvspan(PARTIAL_YEAR - 0.5, PARTIAL_YEAR + 0.45, color="0.5", alpha=0.09,
               lw=0, zorder=0)
    ax.text(PARTIAL_YEAR - 0.40, YMAX - 0.3, "partial year", ha="right", va="top",
            fontsize=F_ANNOT, color="0.45", rotation=90)

ax.set_xlim(YEARS[0] - 0.35, YEARS[-1] + 0.45)
ax.set_ylim(0, YMAX)
ax.set_yticks(range(0, YMAX, YSTEP))
ax.set_xticks(YEARS)
ax.set_xticklabels([str(y) for y in YEARS])
ax.tick_params(labelsize=F_TICK, length=6, width=1.4, pad=8)
ax.set_ylabel(YLABEL, fontsize=F_AXLBL, labelpad=14)
if XLABEL:
    ax.set_xlabel(XLABEL, fontsize=F_AXLBL, labelpad=12)

ax.yaxis.grid(True, color=GRID_COLOR, lw=1.2, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_linewidth(1.4)
    ax.spines[side].set_color("0.25")

handles = [
    Line2D([], [], color=C_ATK, lw=LW_SOLID, marker="o", ms=MS, mfc="white",
           mew=3.0, mec=C_ATK, label="Attack (implicit)"),
    Line2D([], [], color=C_ATK, lw=LW_DASH, ls=DASH, label="Attack (explicit)"),
    Line2D([], [], color=C_DEF, lw=LW_SOLID, marker="s", ms=MS, mfc="white",
           mew=3.0, mec=C_DEF, label="Defense (implicit)"),
    Line2D([], [], color=C_DEF, lw=LW_DASH, ls=DASH, label="Defense (explicit)"),
]
ax.legend(handles=handles, loc="upper left", ncol=1, frameon=False,
          fontsize=F_LEGEND, handlelength=2.6, columnspacing=2.0,
          labelspacing=0.5, borderaxespad=0.4)

fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "trend_attack_defense.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "trend_attack_defense.png", dpi=200, bbox_inches="tight")
print(f"written to {OUT_DIR}")