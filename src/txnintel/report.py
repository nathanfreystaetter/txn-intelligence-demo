"""End to end: raw transactions -> a defensible view of a business's health."""
from __future__ import annotations

import pandas as pd

from . import anomaly, classify, forecast, recurring, revenue


def run(df: pd.DataFrame) -> dict:
    df = df.copy()
    holdout = classify.evaluate_holdout(df)
    df["pred_category"] = classify.predict_out_of_fold(df)
    rec = recurring.detect(df)
    est = revenue.estimate_annual(df)
    monthly = revenue.monthly_revenue(df)
    bt = forecast.backtest(monthly)
    fc = forecast.fit_forecast(monthly)
    flags = anomaly.flag_spend(df)
    result = {"classification": holdout, "recurring": rec, "revenue": est, "backtest": bt, "forecast": fc, "anomalies": flags}
    if "is_anomaly" in df:
        planted = df[df["is_anomaly"]]
        found = flags.merge(planted[["date", "description", "amount"]], on=["date", "description", "amount"])
        result["anomaly_eval"] = {"planted": len(planted), "caught": len(found), "flagged": len(flags)}
    return result


def format_report(r: dict) -> str:
    c, e, b = r["classification"], r["revenue"], r["backtest"]
    lines = [
        "TRANSACTION INTELLIGENCE REPORT (synthetic data)", "",
        f"Classification (time-ordered holdout, {c['n_test']} test rows): accuracy {c['accuracy']:.1%}, macro F1 {c['macro_f1']:.3f}", "",
        f"Annual revenue estimate (trailing {e['months_used']} months): ${e['annual_revenue']:,.0f}  (90% range ${e['p05']:,.0f} to ${e['p95']:,.0f})", "",
        "Recurring streams (monthly equivalent):",
    ]
    for _, s in r["recurring"].iterrows():
        lines.append(f"  {s['direction']:<8} {s['cadence']:<13} {s['merchant']:<28} ${s['monthly_equivalent']:>10,.0f}")
    lines += ["", "Forecast backtest, mean absolute percentage error on the last 6 months:"]
    lines.append(f"  seasonal model {b['model_mape']:.1%} | repeat-last-value {b['naive_mape']:.1%}" + (f" | same-month-last-year {b['seasonal_naive_mape']:.1%}" if "seasonal_naive_mape" in b else ""))
    lines += ["", "Next 6 months of revenue (point, 90% interval):"]
    for d, row in r["forecast"].iterrows():
        lines.append(f"  {d:%Y-%m}  ${row['forecast']:>9,.0f}   (${row['lo']:,.0f} to ${row['hi']:,.0f})")
    lines += ["", f"Anomalous spend flagged: {len(r['anomalies'])}"]
    for _, a in r["anomalies"].head(5).iterrows():
        lines.append(f"  {a['date']:%Y-%m-%d}  {a['description']:<34} ${a['amount']:>10,.2f}  z={a['z']:.1f}")
    if "anomaly_eval" in r:
        ev = r["anomaly_eval"]
        lines.append(f"  (planted {ev['planted']}, caught {ev['caught']}, total flagged {ev['flagged']})")
    return "\n".join(lines)
