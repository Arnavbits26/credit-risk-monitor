"""Load the German Credit dataset and prepare model-ready frames."""
import pandas as pd
from sklearn.model_selection import train_test_split

# Excluded from the model on fairness grounds: sex-linked and nationality-linked fields.
PROTECTED = ["personal_status_sex", "foreign_worker"]
# Age is kept out of the model too; it is used only for monitoring (age-band concentration).
MONITOR_ONLY = ["age"]
RANDOM_STATE = 42


def load(path="data/german_credit.csv"):
    df = pd.read_csv(path)
    # Source coding: credit_risk 1 = good, 0 = bad. We model default (bad) as the event.
    df["default"] = (df["credit_risk"] == 0).astype(int)
    return df.drop(columns=["credit_risk"])


def model_columns(df):
    drop = set(PROTECTED + MONITOR_ONLY + ["default"])
    return [c for c in df.columns if c not in drop]


def split(df, test_size=0.3):
    """Stratified development / holdout split. Holdout stands in for 'new business'."""
    return train_test_split(
        df, test_size=test_size, stratify=df["default"], random_state=RANDOM_STATE
    )
