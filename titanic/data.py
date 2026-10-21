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
