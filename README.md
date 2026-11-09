# txn-intelligence-demo

Turn a raw bank transaction feed into a view of a small business's health: classify transactions, find recurring streams, estimate annual revenue, forecast with seasonality, and flag anomalous spend.

Every number comes from **synthetic data** generated in `src/txnintel/synth.py`. It contains no real company, customer or lender data.

## Why this exists
Cash-flow underwriting depends on a handful of questions a transaction feed can answer. What is this money? What repeats? What is annual revenue, and how sure are we? What happens next? What looks wrong? This repo answers each one with a small, testable function.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m txnintel      # prints the report below
pytest                  # 11 tests
```

## Sample output
```text
TRANSACTION INTELLIGENCE REPORT (synthetic data)

Classification (time-ordered holdout, 294 test rows): accuracy 97.3%, macro F1 0.972

Annual revenue estimate (trailing 12 months): $568,468  (90% range $488,581 to $649,383)

Recurring streams (monthly equivalent):
  outflow  semi-monthly  GUSTO PAYROLL                $   -39,068
  outflow  monthly       LANDLORD PROPERTIES RENT     $    -4,119
  outflow  monthly       AMAZON WEB SERVICES          $    -2,776
  outflow  monthly       NOTION LABS                  $       -94
  inflow   monthly       ACME CORP RETAINER           $     5,876

Forecast backtest, mean absolute percentage error on the last 6 months:
  seasonal model 14.6% | repeat-last-value 70.7% | same-month-last-year 9.5%

Next 6 months of revenue (point, 90% interval):
  2025-01  $   35,243   ($30,620 to $40,563)
  2025-02  $   41,543   ($36,095 to $47,814)
  2025-03  $   52,007   ($45,186 to $59,857)
  2025-04  $   64,469   ($56,014 to $74,201)
  2025-05  $   74,275   ($64,533 to $85,487)
  2025-06  $   77,150   ($67,032 to $88,796)

Anomalous spend flagged: 7
  2024-09-23  POS DEBIT 6494                     $   -571.83  z=8.4
  2023-09-29  POS DEBIT 5086                     $   -476.58  z=7.5
  2023-05-29  POS DEBIT 4303                     $   -442.02  z=7.2
  2023-08-19  POS DEBIT 9897                     $   -301.04  z=5.4
  2024-09-23  MAILCHIMP 7519                     $-13,032.73  z=5.2
  (planted 3, caught 3, total flagged 7)
```

## What each module does
| Module | Question | Method |
|---|---|---|
| `classify.py` | What is this transaction? | TF-IDF on character n-grams of the cleaned description plus the direction of the money, logistic regression. Scored on a time-ordered holdout, so later months test an earlier fit. |
| `recurring.py` | What repeats? | Group by merchant and direction. A stream recurs when gaps between payments vary little and amounts are stable. Reports cadence and a monthly-equivalent amount. |
| `revenue.py` | What is annual revenue? | Trailing twelve months of revenue-classified inflows. Transfers are excluded. A bootstrap over the months gives a 90% range. |
| `forecast.py` | What comes next? | Log-linear trend with Fourier seasonality and a 90% interval. Backtested against two baselines. |
| `anomaly.py` | What looks wrong? | Robust z-score (median and MAD on log amounts) of each outflow against its own category. Skips categories with too little history. |

## Read the results with care
- **Synthetic data is easier than real feeds.** The classifier scores high because the generator uses a limited set of merchants. About 12% of irregular outflows get vague descriptions ("POS DEBIT") to add some difficulty.
- **The forecast does not beat every baseline.** It crushes "repeat the last value", but with two years of history, "same month last year" can beat it. A model earns its place only against the baseline a person would actually use.
- **Anomaly detection catches the three planted events and also flags small purchases.** Those false flags come from vague descriptions that the classifier places in a category with tightly clustered amounts. The detector is only as good as the categories under it.
- **Recurring detection misses a stream when one payment is missing or mislabeled.** That is deliberate: a stream that is not regular is not reported as regular.
- **Not a credit model.** It stops at the inputs an underwriter would use. Policy, pricing and loss modeling are separate problems.

## Layout
```text
src/txnintel/   synth, classify, recurring, revenue, forecast, anomaly, report
tests/          11 tests, including a planted-anomaly recall check and a truth check on revenue
```

MIT licensed.
