"""End-to-end run: model, validate, monitor, report. Usage: python -m src.run_pipeline"""
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import data, model, monitoring as mon


def stressed(df):
    """SIMULATED stress scenario (not real data): larger, longer loans and weaker savings."""
    s = df.copy()
    s["amount"] = (s["amount"] * 1.3).round().astype(int)
    s["duration"] = s["duration"] + 6
    return s


def main():
    df = data.load()
    dev, hold = data.split(df)
    cols = data.model_columns(df)

    results, scores = [], {}
    for kind in ["champion", "challenger"]:
        pipe = model.build(kind, dev, cols).fit(dev[cols], dev["default"])
        for name, frame in [("development", dev), ("holdout", hold)]:
            p = pipe.predict_proba(frame[cols])[:, 1]
            scores[(kind, name)] = p
            results.append({"model": kind, "sample": name, **model.evaluate(frame["default"], p)})
    perf = pd.DataFrame(results).round(4)

    champ = lambda n: scores[("champion", n)]
    score_psi = mon.psi(champ("development"), champ("holdout"))
    dev_auc = perf.query("model=='champion' and sample=='development'")["auc"].iloc[0]
    new_auc = perf.query("model=='champion' and sample=='holdout'")["auc"].iloc[0]

    drift = mon.drift_table(dev, hold, cols)
    stress_drift = mon.drift_table(dev, stressed(hold), cols).head(5)

    full = df.copy()
    full["age_band"] = pd.cut(full["age"], [0, 25, 35, 50, 120], labels=["<=25", "26-35", "36-50", "50+"])
    segs = ["purpose", "age_band", "housing", "employment_duration"]
    concs = {s: mon.concentration(full, s) for s in segs}
    hhis = {s: mon.hhi(concs[s]["exposure"]) for s in segs}
    exc = pd.concat([mon.exceptions(concs[s], s) for s in segs]).sort_values(
        "bad_rate_ratio", ascending=False
    )
    kris = mon.kri_summary(score_psi, dev_auc, new_auc, hhis)

    out = "outputs/"
    perf.to_csv(out + "model_performance.csv", index=False)
    drift.to_csv(out + "drift_psi.csv", index=False)
    kris.to_csv(out + "kri_dashboard.csv", index=False)
    exc.round(4).to_csv(out + "exception_report.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    c = concs["purpose"].sort_values("exposure_share")
    ax[0].barh(c["purpose"], c["exposure_share"])
    ax[0].set(title="Exposure share by purpose", xlabel="share of total exposure")
    ax[1].barh(c["purpose"], c["bad_rate"], color="tab:red")
    ax[1].axvline(full["default"].mean(), color="k", ls="--", label="portfolio")
    ax[1].set(title="Default rate by purpose", xlabel="default rate")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(out + "concentration_by_purpose.png", dpi=130)

    print("MODEL PERFORMANCE\n", perf.to_string(index=False))
    print("\nKRI DASHBOARD\n", kris.to_string(index=False))
    print("\nTOP DRIFT (holdout)\n", drift.head(5).to_string(index=False))
    print("\nTOP DRIFT (SIMULATED stress)\n", stress_drift.to_string(index=False))
    print("\nEXCEPTIONS\n", exc.round(3).to_string(index=False))
    print("\nPortfolio default rate:", round(full["default"].mean(), 3))


if __name__ == "__main__":
    main()
