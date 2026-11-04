from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

def plot_counts(
    df: pd.DataFrame,
    column: str,
    hue: str | None = None,
) -> Axes:
    """Plot the number of observations in each category.

    Args:
        df: Dataset containing the columns to plot.
        column: Categorical column to count.
        hue: Optional column defining comparison groups.

    Returns:
        The Axes containing the count plot.
    """
    _, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(
        data=df,
        x=column,
        hue=hue,
        ax=ax,
    )

    ax.set_title(f"Passenger counts by {column}")
    ax.set_ylabel("Number of passengers")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-.", linewidth=0.5)

    return ax

def plot_distribution(
    df: pd.DataFrame,
    column: str,
    hue: str | None = None,
    bins: int | str = "auto",
    palette: list[str] | None = None,
) -> Axes:
    """Plot a histogram, optionally grouped and coloured.

    Args:
        df: Dataset to plot.
        column: Column whose distribution is displayed.
        hue: Optional column defining comparison groups.
        bins: Histogram bin count or binning method.
        palette: Optional colours for the comparison groups.

    Returns:
        The Axes containing the histogram.
    """
    _, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        data=df,
        x=column,
        hue=hue,
        bins=bins,
        palette=palette,
        multiple="layer",
        alpha=0.5,
        ax=ax,
    )

    ax.set_title(f"Number of passengers / {column}")

    return ax

# --- Figures for the feature-selection report -------------------------------

PRIMARY = "#2a78d6"
SECONDARY = "#eb6834"
MUTED = "#6b6b6b"


def save_figure(ax: Axes, filename: str, directory: str | Path = "figures") -> Path:
    """Save the figure that contains ``ax`` as a PNG in ``directory``.

    Args:
        ax: Any Axes of the figure to save.
        filename: File name without extension, e.g. "01_missing_values".
        directory: Target folder; created if it does not exist.

    Returns:
        The path of the saved file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{filename}.png"
    ax.figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_missing(df: pd.DataFrame) -> Axes:
    """Plot the share of missing values for every column with gaps.

    Args:
        df: Dataset to inspect.

    Returns:
        The Axes containing the horizontal bar chart.
    """
    share = df.isna().mean().mul(100)
    share = share[share > 0].sort_values()

    _, ax = plt.subplots(figsize=(7, 2.5))
    ax.barh(share.index, share.values, color=PRIMARY, height=0.6)
    for y, value in enumerate(share.values):
        ax.text(value + 1, y, f"{value:.1f}%", va="center", fontsize=9)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Missing values (% of passengers)")
    ax.set_title("Missing values per column (training data)")
    _tidy(ax, grid_axis="x")
    return ax


def plot_survival_rate(
    df: pd.DataFrame,
    column: str,
    hue: str | None = None,
    order: list | None = None,
    target: str = "Survived",
) -> Axes:
    """Plot the survival rate per category, with 95% confidence intervals.

    A dashed line marks the overall survival rate, so every bar can be read
    as above or below average. Small groups get wide intervals, which warns
    against over-reading them.

    Args:
        df: Training data containing ``column`` and ``target``.
        column: Categorical feature on the x-axis.
        hue: Optional second feature, shown as coloured bars.
        order: Optional order of the categories on the x-axis.
        target: Binary outcome column.

    Returns:
        The Axes containing the bar chart.
    """
    _, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=df,
        x=column,
        y=target,
        hue=hue,
        order=order,
        errorbar=("ci", 95),
        palette=[PRIMARY, SECONDARY] if hue else None,
        color=None if hue else PRIMARY,
        err_kws={"linewidth": 1, "color": MUTED},
        capsize=0.1,
        saturation=1,
        ax=ax,
    )

    overall = df[target].mean()
    ax.axhline(overall, color=MUTED, linestyle="--", linewidth=1)
    ax.text(
        ax.get_xlim()[1], overall, f" overall {overall:.0%}",
        va="center", ha="left", fontsize=9, color=MUTED,
    )

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.set_ylabel("Survival rate")
    ax.set_title(f"Survival rate by {column}" + (f" and {hue}" if hue else ""))
    _tidy(ax, grid_axis="y")
    return ax


def plot_feature_importance(importances: pd.Series) -> Axes:
    """Plot feature importances as a sorted horizontal bar chart.

    Args:
        importances: Importance per feature, indexed by feature name.

    Returns:
        The Axes containing the bar chart.
    """
    importances = importances.sort_values()

    _, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(importances.index, importances.values, color=PRIMARY, height=0.6)
    ax.set_xlabel("Drop in validation accuracy when the feature is shuffled")
    ax.set_title("Permutation importance (random forest, validation set)")
    _tidy(ax, grid_axis="x")
    return ax


def _tidy(ax: Axes, grid_axis: str) -> None:
    """Apply a light, consistent style: no top/right spines, faint grid."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, linewidth=0.5, alpha=0.4)
