"""Feature engineering for the Titanic analysis."""

import pandas as pd


def add_family_size(df):
    """Add family size, including the passenger."""
    df = df.copy()
    df["Family Size"] = df["SibSp"] + df["Parch"] + 1
    return df


def add_age_interval(df):
    """Group ages at 16, 32, 48 and 64; missing ages stay in group 0."""
    df = df.copy()
    df["Age Interval"] = 0.0
    df.loc[(df["Age"] > 16) & (df["Age"] <= 32), "Age Interval"] = 1
    df.loc[(df["Age"] > 32) & (df["Age"] <= 48), "Age Interval"] = 2
    df.loc[(df["Age"] > 48) & (df["Age"] <= 64), "Age Interval"] = 3
    df.loc[df["Age"] > 64, "Age Interval"] = 4
    return df


def add_fare_interval(df):
    """Group fares at 7.91, 14.454 and 31; missing fares stay in group 0."""
    df = df.copy()
    df["Fare Interval"] = 0.0
    df.loc[
        (df["Fare"] > 7.91) & (df["Fare"] <= 14.454),
        "Fare Interval",
    ] = 1
    df.loc[
        (df["Fare"] > 14.454) & (df["Fare"] <= 31),
        "Fare Interval",
    ] = 2
    df.loc[df["Fare"] > 31, "Fare Interval"] = 3
    return df


def add_sex_pclass(df):
    """Combine sex and passenger class into labels such as F_C1."""
    df = df.copy()
    df["Sex_Pclass"] = (
        df["Sex"].str[0].str.upper()
        + "_C"
        + df["Pclass"].astype(str)
    )
    return df


def parse_names(row):
    """Extract family name, title, given name and maiden name."""
    family_name, rest = row["Name"].split(",", 1)
    title, rest = rest.split(".", 1)

    if "(" in rest:
        given_name, maiden_name = rest.split("(", 1)
        maiden_name = maiden_name.rstrip(")")
    else:
        given_name = rest
        maiden_name = None

    return pd.Series(
        [family_name, title.strip() + ".", given_name, maiden_name]
    )


def add_name_features(df):
    """Add the four name components to the dataframe."""
    df = df.copy()
    columns = ["Family Name", "Title", "Given Name", "Maiden Name"]
    df[columns] = df.apply(parse_names, axis=1)
    return df


def add_family_type(df):
    """Label families as Single (1), Small (2–4) or Large (5+)."""
    df = df.copy()
    size = df["Family Size"]

    df["Family Type"] = size.astype(object)
    df.loc[size == 1, "Family Type"] = "Single"
    df.loc[(size > 1) & (size < 5), "Family Type"] = "Small"
    df.loc[size >= 5, "Family Type"] = "Large"
    return df


def add_grouped_titles(df):
    """Standardise Miss and Mrs titles and combine the listed rare titles."""
    df = df.copy()
    rare_titles = [
        "Lady.", "the Countess.", "Capt.", "Col.", "Don.",
        "Dr.", "Major.", "Rev.", "Sir.", "Jonkheer.", "Dona.",
    ]

    df["Titles"] = df["Title"].replace(
        {"Mlle.": "Miss.", "Ms.": "Miss.", "Mme.": "Mrs."}
    )
    df["Titles"] = df["Titles"].replace(rare_titles, "Rare")
    return df


def survival_by_title_and_sex(df):
    """Return mean survival for each title and sex group."""
    return df.groupby(["Titles", "Sex"], as_index=False)["Survived"].mean()


def encode_sex(df):
    """Encode sex as 0/1, keeping already encoded values."""
    df = df.copy()
    df["Sex"] = df["Sex"].map(
        {"female": 1, "male": 0, 1: 1, 0: 0}
    ).astype(int)
    return df