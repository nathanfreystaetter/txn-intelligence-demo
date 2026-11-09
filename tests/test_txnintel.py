import numpy as np
import pandas as pd
import pytest

from txnintel import anomaly, classify, forecast, recurring, report, revenue, synth


@pytest.fixture(scope="module")
def df():
    return synth.generate()


@pytest.fixture(scope="module")
def result(df):
    return report.run(df)


def test_generator_is_deterministic():
    pd.testing.assert_frame_equal(synth.generate(seed=3), synth.generate(seed=3))


def test_normalize_drops_reference_numbers_and_keeps_direction():
    assert classify.normalize("Stripe Payout 4821", 150.0) == "STRIPE PAYOUT INFLOW"
    assert classify.normalize("Gusto Payroll 12", -900.0).endswith("OUTFLOW")


def test_classifier_beats_ninety_percent_on_later_months(df):
    assert classify.evaluate_holdout(df)["accuracy"] > 0.90


def test_recurring_finds_payroll_rent_and_retainer(result):
    rec = result["recurring"]
    found = set(rec["merchant"])
    assert {"GUSTO PAYROLL", "LANDLORD PROPERTIES RENT", "ACME CORP RETAINER"} <= found
    assert rec.set_index("merchant").loc["ACME CORP RETAINER", "direction"] == "inflow"


def test_irregular_spend_is_not_called_recurring(result):
    assert not any("STAPLES" in m or "UBER" in m for m in result["recurring"]["merchant"])


def test_revenue_estimate_is_close_to_truth_and_range_contains_it(df, result):
    truth = df[(df.category == "revenue")].set_index("date")["amount"].resample("MS").sum().iloc[-12:].sum()
    est = result["revenue"]
    assert abs(est["annual_revenue"] - truth) / truth < 0.05
    assert est["p05"] < est["annual_revenue"] < est["p95"]


def test_forecast_is_positive_and_interval_is_ordered(result):
    fc = result["forecast"]
    assert len(fc) == 6 and (fc["forecast"] > 0).all()
    assert ((fc["lo"] < fc["forecast"]) & (fc["forecast"] < fc["hi"])).all()


def test_forecast_beats_repeating_the_last_value(result):
    assert result["backtest"]["model_mape"] < result["backtest"]["naive_mape"]


def test_every_planted_anomaly_is_caught(result):
    ev = result["anomaly_eval"]
    assert ev["caught"] == ev["planted"] == 3


def test_anomaly_detector_skips_thin_categories():
    tiny = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5), "description": "X", "amount": -100.0, "pred_category": "other"})
    assert anomaly.flag_spend(tiny).empty


def test_monthly_revenue_ignores_transfers():
    d = pd.DataFrame({"date": pd.to_datetime(["2024-01-05", "2024-01-06"]), "amount": [1000.0, 5000.0], "pred_category": ["revenue", "transfer"]})
    assert revenue.monthly_revenue(d).iloc[0] == 1000.0
