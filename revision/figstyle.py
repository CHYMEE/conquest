"""Shared figure style for the CONQUEST revision (reference: Fig. 6 of the manuscript).

Bold condensed sans-serif text (Arial Narrow Bold), bold "(a) Title" panel titles, thick
lines with outlined markers and a soft drop shadow, glossy 3-D bars and boxes (cylindrical
light-to-dark shading), glossy "ball" scatter markers, vivid colours, light dashed grid,
boxed legends, and 600-dpi output. Only appearance is controlled here; plotted data are
never modified.

Usage in a plotting script:
    import figstyle
    figstyle.apply()
    ... plot ...
    figstyle.glossy_bars(ax, bar_container)       # bars
    figstyle.glossy_boxes(ax, bp["boxes"])        # boxplot boxes (patch_artist=True)
    figstyle.balls(ax, x, y, color)               # 3-D scatter markers
    figstyle.shadow(line_artists)                 # drop shadow on lines
    figstyle.finish(ax)                           # spines, ticks, dashed grid
"""
import os

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import numpy as np
from cycler import cycler
from matplotlib import patheffects as pe

BRIGHT = ["#1565ff", "#ff7f00", "#18b33c", "#e8112d", "#8e2ce0", "#00a7a7", "#ff3d9a", "#c79a00", "#5a3e1b"]
INK = "#111111"
DPI = 600


def _register_narrow():
    """Arial Narrow registers as family 'Arial' with a condensed stretch."""
    found = False
    for f in ("ARIALNB.TTF", "ARIALN.TTF"):
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", f)
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            found = True
    return "condensed" if found else "normal"


def apply(base_size: float = 13.5):
    stretch = _register_narrow()
    mpl.rcParams.update({
        "font.family": "Arial", "font.stretch": stretch, "font.weight": "bold", "font.size": base_size,
        "axes.titleweight": "bold", "axes.titlesize": base_size + 2.5,
        "axes.labelweight": "bold", "axes.labelsize": base_size + 1,
        "axes.linewidth": 1.8, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "axes.prop_cycle": cycler(color=BRIGHT), "axes.axisbelow": True,
        "xtick.labelsize": base_size, "ytick.labelsize": base_size,
        "xtick.major.width": 1.6, "ytick.major.width": 1.6, "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.color": INK, "ytick.color": INK,
        "lines.linewidth": 3.0, "lines.markersize": 9, "lines.markeredgewidth": 1.6,
        "lines.markeredgecolor": INK,
        "patch.linewidth": 1.5, "patch.edgecolor": INK,
        "legend.fontsize": base_size - 1, "legend.frameon": True, "legend.framealpha": 0.95,
        "legend.edgecolor": "0.35", "legend.fancybox": False,
        "grid.linestyle": "--", "grid.linewidth": 0.9, "grid.alpha": 0.45, "grid.color": "0.6",
        "errorbar.capsize": 5,
        "boxplot.boxprops.linewidth": 2.0, "boxplot.whiskerprops.linewidth": 2.0,
        "boxplot.capprops.linewidth": 2.0, "boxplot.medianprops.linewidth": 3.0,
        "boxplot.medianprops.color": INK, "boxplot.flierprops.markersize": 6,
        "mathtext.default": "bf", "mathtext.fontset": "dejavusans",
        "savefig.dpi": DPI, "savefig.bbox": "tight", "savefig.facecolor": "white", "figure.dpi": 110,
    })


def shadow(artists, offset=(1.6, -1.6), alpha=0.35):
    """Soft drop shadow behind lines (3-D lift)."""
    for a in np.atleast_1d(artists):
        a.set_path_effects([pe.SimpleLineShadow(offset=offset, alpha=alpha, shadow_color="0.25"), pe.Normal()])


def text_shadow(texts, alpha=0.25):
    for t in np.atleast_1d(texts):
        t.set_path_effects([pe.withSimplePatchShadow(offset=(0.8, -0.8), alpha=alpha, shadow_rgbFace="0.4")])


def _gradient(color, n=64):
    """Cylindrical shading across the width: dark edge -> highlight (~35%) -> dark edge."""
    rgb = np.array(mcolors.to_rgb(color))
    x = np.linspace(0, 1, n)
    light = np.clip(np.exp(-((x - 0.35) / 0.28) ** 2), 0, 1)
    shade = 0.80 + 0.20 * (1 - np.abs(x - 0.35) / 0.65)
    rows = rgb[None, :] * shade[:, None]
    rows = rows + (1 - rows) * (0.60 * light)[:, None]
    return np.clip(rows, 0, 1)[None, :, :]


def _shade_patch(ax, patch, color, horizontal=False):
    bb = patch.get_extents().transformed(ax.transData.inverted())
    x0, x1, y0, y1 = bb.x0, bb.x1, bb.y0, bb.y1
    # keep the shading inside the visible axes (bars may extend below a non-zero y-limit)
    (xa, xb), (ya, yb) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    x0, x1, y0, y1 = max(x0, xa), min(x1, xb), max(y0, ya), min(y1, yb)
    if x1 <= x0 or y1 <= y0:
        return
    img = _gradient(color)
    if horizontal:
        img = np.transpose(img, (1, 0, 2))[::-1]
    im = ax.imshow(img, extent=(x0, x1, y0, y1), aspect="auto", origin="lower",
                   zorder=patch.get_zorder() + 0.01, interpolation="bicubic")
    im.set_clip_path(patch)
    im.set_clip_box(ax.bbox)
    patch.set_facecolor("none")
    patch.set_edgecolor(INK)
    patch.set_zorder(patch.get_zorder() + 0.02)
    patch.set_path_effects([pe.SimplePatchShadow(offset=(2.2, -2.2), alpha=0.3, shadow_rgbFace="0.3"), pe.Normal()])


def _locked(ax, fn):
    """Run fn() without letting the shading images change the axis limits."""
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.set_autoscale_on(False)
    fn()
    ax.set_xlim(xl)
    ax.set_ylim(yl)


def glossy_bars(ax, bars, colors=None, horizontal=False):
    """3-D glossy bars (call after the axis limits are final)."""
    ax.figure.canvas.draw()

    def go():
        for i, b in enumerate(bars):
            c = colors[i % len(colors)] if colors is not None else b.get_facecolor()
            _shade_patch(ax, b, c, horizontal=horizontal)
    _locked(ax, go)


def glossy_boxes(ax, boxes, colors):
    ax.figure.canvas.draw()

    def go():
        for b, c in zip(boxes, colors):
            _shade_patch(ax, b, c)
    _locked(ax, go)


def balls(ax, x, y, color, s=90, zorder=4, label=None, marker="o", alpha=1.0):
    """Glossy sphere markers: coloured disc with dark rim plus an offset highlight."""
    x, y = np.asarray(x), np.asarray(y)
    h = ax.scatter(x, y, s=s, color=color, edgecolor=INK, linewidth=1.3, zorder=zorder, label=label,
                   marker=marker, alpha=alpha)
    h.set_path_effects([pe.SimplePatchShadow(offset=(1.5, -1.5), alpha=0.35, shadow_rgbFace="0.3"), pe.Normal()])
    if marker == "o":
        hl = ax.scatter(x, y, s=s * 0.22, color="white", alpha=0.75 * alpha, edgecolor="none",
                        zorder=zorder + 0.1, marker="o", linewidths=0)
        # shift the highlight up-left in display space (offset ~ 0.2 marker radius)
        d = 0.18 * np.sqrt(s) / 72.0
        hl.set_offset_transform(ax.transData + mpl.transforms.ScaledTranslation(-d, d, ax.figure.dpi_scale_trans))
    return h


def panel_title(ax, text, **kw):
    t = ax.set_title(text, pad=8, **kw)
    return t


def finish(ax, grid_axis="both"):
    for sp in ax.spines.values():
        sp.set_linewidth(1.8)
        sp.set_color(INK)
    ax.tick_params(width=1.6, length=5)
    ax.grid(True, axis=grid_axis)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")


def bright_cmap(name="viridis"):
    return mpl.colormaps[name]


def bright_seq(hi="#1565ff", lo="#fffbe6", name="bright_seq"):
    """Bright sequential colormap (cream -> vivid colour), used for matrices and heatmaps."""
    return mcolors.LinearSegmentedColormap.from_list(name, [lo, "#7fc4ff", hi])
