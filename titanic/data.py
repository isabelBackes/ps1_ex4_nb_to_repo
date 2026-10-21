import pandas as pd

def load_data(path):
    """ Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)

