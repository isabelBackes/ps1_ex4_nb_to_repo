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