"""Champion (logistic regression) and challenger (gradient boosting) default models."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import RANDOM_STATE


def _preprocess(df, cols):
    num = [c for c in cols if df[c].dtype != object and str(df[c].dtype) != "str"]
    cat = [c for c in cols if c not in num]
    return ColumnTransformer(
        [("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]
    )


def build(kind, df, cols):
    est = (
        LogisticRegression(max_iter=2000, class_weight="balanced")
        if kind == "champion"
        else GradientBoostingClassifier(random_state=RANDOM_STATE)
    )
    return Pipeline([("prep", _preprocess(df, cols)), ("model", est)])


def ks_statistic(y_true, score):
    """Max separation between the score CDFs of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def evaluate(y_true, score):
    auc = roc_auc_score(y_true, score)
    return {"auc": float(auc), "gini": float(2 * auc - 1), "ks": ks_statistic(y_true, score)}
