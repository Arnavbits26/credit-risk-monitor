"""Portfolio monitoring: drift (PSI), concentration (HHI), KRIs with RAG status, exceptions."""
import numpy as np
import pandas as pd

# Illustrative thresholds for this project. A real policy would set its own.
PSI_AMBER, PSI_RED = 0.10, 0.25
HHI_AMBER, HHI_RED = 0.15, 0.25
BAD_RATE_RATIO_AMBER, BAD_RATE_RATIO_RED = 1.25, 1.5
EXCEPTION_MIN_SHARE = 0.05  # segment must hold at least 5% of exposure to be flagged


def psi(expected, actual, bins=10):
    """Population Stability Index of a numeric series, bins from the expected sample's quantiles."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected, actual):
    cats = sorted(set(expected) | set(actual))
    e = pd.Series(expected).value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=1e-4)
    a = pd.Series(actual).value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=1e-4)
    return float(((a - e) * np.log(a / e)).sum())


def hhi(shares, normalise=True):
    """Herfindahl-Hirschman index of exposure shares.

    Raw HHI ranges from 1/n (even split) to 1 (one bucket), so it is not comparable across
    dimensions with different bucket counts. The normalised form rescales to 0 (even) .. 1 (single bucket).
    """
    s = np.asarray(shares, float)
    s = s / s.sum()
    raw = float((s**2).sum())
    n = len(s)
    if not normalise or n == 1:
        return raw
    return (raw - 1 / n) / (1 - 1 / n)


def rag(value, amber, red):
    return "RED" if value >= red else "AMBER" if value >= amber else "GREEN"


def drift_table(dev, new, cols):
    rows = []
    for c in cols:
        num = pd.api.types.is_numeric_dtype(dev[c])
        v = psi(dev[c], new[c]) if num else psi_categorical(dev[c], new[c])
        rows.append({"feature": c, "psi": round(v, 4), "status": rag(v, PSI_AMBER, PSI_RED)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def concentration(df, segment, exposure="amount"):
    """Exposure share, default rate and rate-vs-portfolio ratio for each segment."""
    g = df.groupby(segment, observed=True).agg(
        accounts=("default", "size"), exposure=(exposure, "sum"), bad_rate=("default", "mean")
    )
    g["exposure_share"] = g["exposure"] / g["exposure"].sum()
    g["bad_rate_ratio"] = g["bad_rate"] / df["default"].mean()
    return g.reset_index()


def exceptions(conc, segment):
    """Segments that are both material and riskier than the portfolio."""
    hit = conc[
        (conc["exposure_share"] >= EXCEPTION_MIN_SHARE)
        & (conc["bad_rate_ratio"] >= BAD_RATE_RATIO_AMBER)
    ].copy()
    hit["status"] = hit["bad_rate_ratio"].apply(
        lambda r: rag(r, BAD_RATE_RATIO_AMBER, BAD_RATE_RATIO_RED)
    )
    hit.insert(0, "dimension", segment)
    return hit.rename(columns={segment: "segment"})


def kri_summary(score_psi, dev_auc, new_auc, hhis):
    rows = [{"kri": "Score PSI (dev vs holdout)", "value": round(score_psi, 4),
             "status": rag(score_psi, PSI_AMBER, PSI_RED)}]
    drop = dev_auc - new_auc
    rows.append({"kri": "AUC drop (dev - holdout)", "value": round(drop, 4),
                 "status": rag(drop, 0.03, 0.06)})
    for name, v in hhis.items():
        rows.append({"kri": f"Exposure HHI: {name}", "value": round(v, 4),
                     "status": rag(v, HHI_AMBER, HHI_RED)})
    return pd.DataFrame(rows)
