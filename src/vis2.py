#!/usr/bin/env python3
"""
Reset-failure taxonomy for two platforms, as side-by-side cascading breakdowns.

Each level renormalises the drilled segment to the panel width, so rare leaves
(1/185) stay legible; grey wedges carry the parent-child lineage.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "vis"

# ============================================================================
# 1. CONFIG
# ============================================================================
# colours: the drilled branch darkens level by level; terminal leaves get a
# contrasting hue (grasp failures) or a light tint (stochastic slippage)
C_OK = "#E8A33D"
C_FAIL = "#4A87C9"
C_OOW = "#2E6FB7"
C_GRASP = "#C0355C"
C_CONS = "#17457A"
C_SLIP = "#A9C6E4"
T_L, T_D = "white", "#0F3055"


def cascade(ok, fail, oow, grasp, cons, slip):
    """(name, count, fill, text colour) per level; last item = drilled index."""
    return [
        ([("Successful resets", ok, C_OK, T_L),
          ("Failed resets", fail, C_FAIL, T_L)], 1),
        ([("Out-of-workspace", oow, C_OOW, T_L),
          ("Grasp failures", grasp, C_GRASP, T_L)], 0),
        ([("Constraint violation", cons, C_CONS, T_L),
          ("Stochastic slippage", slip, C_SLIP, T_D)], None),
    ]


PANELS = [
    ("SO-100", cascade(158, 42, 29, 13, 24, 5)),
    ("xArm6",  cascade(193, 7, 5, 2, 4, 1)),
]

C_WEDGE, WEDGE_ALPHA = "#B0B0B0", 0.20
C_HEAD = "#1A1A1A"

FONT_FAMILY, FONT_NAME = "serif", "DejaVu Serif"
SHOW_HEADERS = False        # True prints "SO-100 (n = 200)" above each panel

W = 10.0                    # canvas width (inches == data units)
ASPECT = 2              # width : height of the whole figure
COL_GAP = 0.55              # space between the two panels
HEAD_H = 1.35 if SHOW_HEADERS else 0.60   # header band / sliver-label room
GAP_FRAC = 0.38             # wedge height as a fraction of the bar height
PAD = 0.035                 # gap between adjacent segments
MIN_INLINE = 0.45           # below this width (in), the label moves outside

# fonts scale with the bar height, in points per inch of bar
K_NAME, K_NUM, K_PCT, K_HEAD = 14.0, 21.5, 12.5, 17.5

# ============================================================================
# 2. RENDER
# ============================================================================
plt.rcParams.update({
    "font.family": FONT_FAMILY, "font.serif": [FONT_NAME],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

n_lv = len(PANELS[0][1])
H = W / ASPECT
BAR_H = (H - HEAD_H) / (n_lv + (n_lv - 1) * GAP_FRAC)
GAP = GAP_FRAC * BAR_H
PANEL_W = (W - COL_GAP) / len(PANELS)
F_NAME, F_NUM = K_NAME * BAR_H, K_NUM * BAR_H
F_PCT, F_HEAD = K_PCT * BAR_H, K_HEAD * BAR_H

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
ax.set_xlim(0, W)
ax.set_ylim(0, H)
fig.canvas.draw()
rend = fig.canvas.get_renderer()
PPI = fig.get_dpi()


def text_w(txt, fs, bold=False):
    t = ax.text(0, -9, txt, fontsize=fs, weight="bold" if bold else "normal")
    w = t.get_window_extent(rend).width / PPI
    t.remove()
    return w


def fit_name(name, avail, fs):
    """Wrap into at most two lines, then shrink until it fits."""
    words = name.split()
    cands = [name] + [" ".join(words[:k]) + "\n" + " ".join(words[k:])
                      for k in range(1, len(words))]
    for c in cands:
        if max(text_w(l, fs, True) for l in c.split("\n")) <= avail:
            return c, fs
    best = min(cands, key=lambda c: max(text_w(l, fs, True) for l in c.split("\n")))
    while fs > 8 and max(text_w(l, fs, True) for l in best.split("\n")) > avail:
        fs -= 0.5
    return best, fs


def draw_panel(x0, head, levels):
    if SHOW_HEADERS:
        total0 = sum(c for _, c, _, _ in levels[0][0])
        ax.text(x0 + PANEL_W / 2, H - HEAD_H * 0.26,
                f"{head}   (n = {total0})", ha="center", va="center",
                fontsize=F_HEAD, color=C_HEAD, weight="bold")

    y = H - HEAD_H
    for segs, drill in levels:
        total = sum(c for _, c, _, _ in segs)
        x, row = x0, []
        for name, cnt, col, tcol in segs:
            w = PANEL_W * cnt / total
            ax.add_patch(FancyBboxPatch((x + PAD, y - BAR_H), w - 2 * PAD,
                                        BAR_H, fc=col, ec="none", zorder=3,
                                        boxstyle="round,pad=0,rounding_size=0.06"))
            cx = x + w / 2
            avail = w - 4 * PAD
            num, pct = f"{cnt}", f"{100 * cnt / total:.0f}%"

            if avail < MIN_INLINE:          # sliver: label it below the bar
                lx = min(max(cx, x0 + 0.4), x0 + PANEL_W)
                ha = "right" if lx > x0 + PANEL_W - 1.6 else "center"
                ax.plot([cx, cx], [y - BAR_H, y - BAR_H - 0.16], color=col, lw=1.6, zorder=4)
                ax.text(lx, y - BAR_H - 0.22, f"{name}  {num} ({pct})", ha=ha,
                        va="top", fontsize=F_NAME * 0.85, color=col,
                        weight="bold", zorder=4)
                row.append((x, x + w))
                x += w
                continue

            label, fs = fit_name(name, avail, F_NAME)
            ax.text(cx, y - 0.33 * BAR_H, label, ha="center", va="center",
                    fontsize=fs, color=tcol, weight="bold", zorder=4, linespacing=1.05)
            fn, fp = F_NUM, F_PCT
            while fn > 10 and (text_w(num, fn, True) + 0.10
                               + text_w(pct, fp, True)) > avail:
                fn, fp = fn * 0.94, fp * 0.94
            w_num, w_pct = text_w(num, fn, True), text_w(pct, fp, True)
            xl = cx - (w_num + 0.10 + w_pct) / 2
            ax.text(xl, y - 0.72 * BAR_H, num, ha="left", va="center",
                    fontsize=fn, color=tcol, weight="bold", zorder=4)
            ax.text(xl + w_num + 0.10, y - 0.745 * BAR_H, pct, ha="left",
                    va="center", fontsize=fp, color=tcol, weight="bold",
                    alpha=0.9, zorder=4)
            row.append((x, x + w))
            x += w

        if drill is None:
            break
        a, b = row[drill]
        y2 = y - BAR_H - GAP
        ax.add_patch(Polygon([(a + PAD, y - BAR_H), (b - PAD, y - BAR_H),
                              (x0 + PANEL_W, y2), (x0, y2)], closed=True,
                             fc=C_WEDGE, alpha=WEDGE_ALPHA, ec="none", zorder=1))
        y = y2


for i, (head, levels) in enumerate(PANELS):
    draw_panel(i * (PANEL_W + COL_GAP), head, levels)

OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "reset_failure_breakdown.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "reset_failure_breakdown.png", dpi=200,
            bbox_inches="tight")
print(f"written to {OUT_DIR}")