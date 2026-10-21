import pandas as pd
from IPython.display import display

def load_data(path):
    """ Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def inspect_data(df: pd.DataFrame) -> None:
    """Display a quick overview without modifying the dataset."""
    
    print("Preview")
    display(df.head())

    print("\nColumn information")
    df.info()

    print("\nNumeric summary")
    display(df.describe())


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing values and column data types.

    Args:
        df: Dataset to inspect.

    Returns:
        A table containing missing-value counts, percentages,
        and data types for each column.
    """
    total = df.isnull().sum()
    percent = df.isnull().sum() / df.isnull().count() * 100

    summary = pd.concat(
        [total, percent],
        axis=1,
        keys=["Total", "Percent"],
    )

    types = []
    for col in df.columns:
        types.append(str(df[col].dtype))

    summary["Types"] = types
    return summary.T

def most_frequent_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise the most frequent non-missing value in each column.

    Args:
        df: Dataset to inspect.

    Returns:
        A table containing non-missing counts, most frequent values,
        their frequencies, and percentages of non-missing values.
        Columns with no observed values have missing summary values.
    """
    total = df.count()
    summary = total.to_frame(name="Total")

    items = []
    frequencies = []

    for col in df.columns:
        counts = df[col].value_counts()

        if counts.empty:
            items.append(pd.NA)
            frequencies.append(0)
        else:
            items.append(counts.index[0])
            frequencies.append(counts.iloc[0])

    summary["Most frequent item"] = items
    summary["Frequence"] = frequencies
    summary["Percent from total"] = (
        summary["Frequence"] / summary["Total"] * 100
    ).round(3)

    return summary.T


def unique_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise non-missing counts and unique values per column.

    Args:
        df: Dataset to inspect.

    Returns:
        A table with non-missing counts and distinct non-missing
        value counts for each column.
    """
    total = df.count()
    summary = total.to_frame(name="Total")

    uniques = []
    for col in df.columns:
        uniques.append(df[col].nunique())

    summary["Uniques"] = uniques
    return summary.T

def combine_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine training and test data for exploratory analysis.

    Args:
        train_df: Training dataset.
        test_df: Test dataset.

    Returns:
        A combined table with a 'set' column identifying each
        row's source. The input datasets are not modified.
    """
    train = train_df.assign(set="train")
    test = test_df.assign(set="test")

    return pd.concat([train, test], axis=0)
