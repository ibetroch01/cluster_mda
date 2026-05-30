"""Small plotting helpers for report-quality figures."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLUSTER_COLORS = {
    # Shared colors keep report figures, maps and dashboard interpretation aligned.
    "regular": "#3B6EA8",
    "seasonal": "#E6862E",
    "mixed": "#6BB6A8",
    "commuter": "#4C9A5F",
    "grey": "#6E7781",
}

K2_CLUSTER_LABELS = {
    0: "Broad mixed / regular-use",
    1: "Strongly seasonal / recreational-like",
}

K3_CLUSTER_LABELS = {
    0: "Mixed daytime / regular-use",
    1: "Strongly seasonal / recreational-like",
    2: "Commuter-like / peak-oriented",
}

FEATURE_LABELS = {
    "log_weekend_weekday_ratio": "Weekend/\nweekday\nbalance",
    "weekday_commute_peak_share": "Weekday\ncommute\npeaks",
    "weekday_midday_share": "Weekday\nmidday",
    "weekend_midday_afternoon_share": "Weekend\nmidday-\nafternoon",
    "log_summer_winter_ratio": "Summer/\nwinter\nseasonality",
}

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def setup_matplotlib():
    """Import matplotlib and apply one clean academic style."""
    import os

    # Keep matplotlib cache inside the project so figure scripts work in local
    # environments without writing hidden files elsewhere.
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10.5,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": "#D9DDE3",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
        }
    )
    return plt, PercentFormatter


def save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def clean_axes(ax, grid_axis="y") -> None:
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)


def add_note(fig, text: str) -> None:
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=8.5, color="#4A4A4A")
