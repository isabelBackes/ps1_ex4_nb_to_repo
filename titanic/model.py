from titanic import feature, model
"""Functions for the Titanic baseline model."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def split_data(df, valid_size=0.2):
    """Split data into training and validation sets."""
    return train_test_split(
        df,
        test_size=valid_size,
        random_state=42,
        shuffle=True,
    )


def get_features_target(df, predictors, target="Survived"):
    """Extract predictor columns and target values."""
    return df[predictors], df[target].values


def train_model(X, y):
    """Train and return a random forest classifier."""
    model = RandomForestClassifier(
        n_jobs=-1,
        random_state=42,
        criterion="gini",
        n_estimators=100,
        verbose=False,
    )
    model.fit(X, y)
    return model


def evaluate_model(y_true, y_pred):
    """Return a classification report."""
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Not Survived", "Survived"],
        zero_division=0,
    )