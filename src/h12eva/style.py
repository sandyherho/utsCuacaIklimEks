"""Figure style following AGU artwork guidelines.

Widths follow AGU column sizes (95 mm single, 190 mm full page). Text
uses an Arial-compatible sans-serif at 7-8 pt, fonts are embedded as
TrueType, panels carry lowercase labels in parentheses, and colors come
from the Okabe-Ito palette, which remains distinguishable under the
common forms of color-vision deficiency.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

from h12eva import config as cfg

MM = 1.0 / 25.4
WIDTH_SINGLE = 95.0 * MM
WIDTH_FULL = 190.0 * MM

# Okabe-Ito palette.
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#F0E442"
BLUE = "#0072B2"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#7F7F7F"
LIGHT_GREY = "#D9D9D9"

DRY = VERMILION
EVENT = VERMILION
EPOCH_COLORS = (BLUE, VERMILION, GREEN)


def apply():
    """Set global matplotlib parameters."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                            "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 0.9,
        "lines.markersize": 4,
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def panel_label(ax, letter, x=-0.12, y=1.03):
    """Add an AGU-style panel label such as (a)."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


def plain_log_axis(ax, axis="x"):
    """Label a log axis with plain numbers (1,000 instead of 10^3)."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}")
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(fmt)
    target.set_minor_formatter(NullFormatter())


def save(fig, name):
    """Save a figure as vector PDF and 300 dpi PNG."""
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIG_DIR / f"{name}.pdf")
    fig.savefig(cfg.FIG_DIR / f"{name}.png", dpi=300)
    plt.close(fig)


ANOMALY_LABEL = r"$\delta^{18}$O anomaly (‰ VPDB)"
DRY_DOWN = r"drier $\downarrow$"


def dry_down(ax):
    """Invert the anomaly axis so that drought plots downward.

    Higher d18O means less rain, so the axis is flipped to follow the
    usual convention of speleothem plots: wet up, dry down.
    """
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


YEAR_LABEL = "Year (CE; negative = BCE)"
