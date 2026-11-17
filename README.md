# Credit Risk Oversight Monitor

A small, reproducible credit-risk oversight workflow on the public German Credit dataset (1,000 loans, 30% default rate).
It builds a default model, **independently challenges it**, then monitors the portfolio the way a second-line risk team would: drift, concentration, KRIs with RAG status, and an exception report.

## What it does
| Step | What | Where |
|---|---|---|
| 1. Champion model | Logistic regression predicting default (interpretable, policy-friendly) | `src/model.py` |
| 2. Effective challenge | Gradient-boosting challenger on the same data; compare AUC, Gini and KS on a holdout | `src/model.py`, `src/run_pipeline.py` |
| 3. Fairness control | Sex-linked and nationality-linked fields and age are excluded from the model (age is used only for monitoring) | `src/data.py` |
| 4. Drift monitoring | Population Stability Index (PSI) for the score and every feature, development vs holdout | `src/monitoring.py` |
| 5. Concentration | Exposure share, default rate and normalised HHI by purpose, age band, housing, employment | `src/monitoring.py` |
| 6. KRIs and exceptions | RAG dashboard plus a report of segments that are both material (>=5% of exposure) and riskier than the portfolio | `outputs/` |

## Results (seed 42, 70/30 stratified split)
| Model | Sample | AUC | Gini | KS |
|---|---|---|---|---|
| Champion (logistic) | development | 0.824 | 0.649 | 0.516 |
| Champion (logistic) | holdout | 0.794 | 0.588 | 0.475 |
| Challenger (boosting) | development | 0.971 | 0.941 | 0.825 |
| Challenger (boosting) | holdout | 0.766 | 0.531 | 0.430 |

* The challenger scores 0.97 AUC in development but only 0.77 on the holdout, so it overfits. The simpler champion generalises better and stays the recommended model.
* Score PSI is 0.029 (GREEN). The AUC drop of 0.030 is AMBER, at the edge of its band. With a 300-loan holdout, treat this as noisy.
* Exceptions: applicants aged 25 or under (42% default rate, 1.40x the portfolio, 17% of exposure), "for free" housing, under 1 year of employment, renters and new-car loans.
* Housing exposure is concentrated (normalised HHI 0.25, RED) because 71% of the book is owner-occupiers.

## Run it
```bash
pip install -r requirements.txt
python -m src.run_pipeline      # prints the tables and writes outputs/
python -m pytest -q             # 11 tests
```

## Limitations (please read)
* **Small, old, cross-sectional data.** There is no time dimension, so "drift" compares the development sample with a random holdout and will look stable by construction. The stress scenario printed by the pipeline (larger, longer loans) is **simulated**, not real.
* **Thresholds are illustrative** (PSI 0.10/0.25 are common rules of thumb; the HHI, AUC-drop and bad-rate-ratio limits are my own). A real policy would set its own.
* Exposure is the loan `amount` column in DM, treated as outstanding balance. There is no LGD or EAD modelling.
* This is a learning project, not a production scorecard.

## Data
`data/german_credit.csv` is the Statlog (German Credit Data) dataset, via a public GitHub mirror of the UCI repository.
