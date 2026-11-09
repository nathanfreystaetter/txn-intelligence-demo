"""Recurring inflow and outflow detection from timing regularity and amount stability."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

CADENCES = [("weekly", 7, 1.5), ("semi-monthly", 15, 2.0), ("monthly", 30.4, 3.0), ("quarterly", 91, 8.0)]


def merchant_key(description: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", " ", description.upper())).strip()


def _cadence(median_gap: float) -> str | None:
    for name, days, tol in CADENCES:
        if abs(median_gap - days) <= tol:
            return name
    return None


def detect(df: pd.DataFrame, min_occurrences: int = 4, max_gap_cv: float = 0.25, max_amount_cv: float = 0.35) -> pd.DataFrame:
    """One row per recurring stream: cadence, typical amount, and a monthly-equivalent amount.

    A stream recurs when the same merchant and direction appear on a steady cadence
    (low variation in the gaps) with a stable amount. Anything looser is treated as irregular.
    """
    d = df.assign(key=df["description"].map(merchant_key), direction=np.where(df["amount"] > 0, "inflow", "outflow"))
    out = []
    for (key, direction), g in d.groupby(["key", "direction"]):
        if len(g) < min_occurrences:
            continue
        dates = g["date"].sort_values().drop_duplicates()
        gaps = dates.diff().dt.days.dropna().to_numpy(dtype=float)
        if len(gaps) < min_occurrences - 1 or gaps.mean() == 0:
            continue
        gap_cv = float(gaps.std() / gaps.mean())
        amounts = g["amount"].abs()
        amount_cv = float(amounts.std() / amounts.mean()) if amounts.mean() else np.inf
        cadence = _cadence(float(np.median(gaps)))
        if cadence is None or gap_cv > max_gap_cv or amount_cv > max_amount_cv:
            continue
        per_month = 30.4 / float(np.median(gaps))
        out.append({
            "merchant": key, "direction": direction, "cadence": cadence, "occurrences": len(g),
            "typical_amount": float(g["amount"].median()), "gap_cv": round(gap_cv, 3), "amount_cv": round(amount_cv, 3),
            "monthly_equivalent": float(g["amount"].median() * per_month),
        })
    return pd.DataFrame(out).sort_values("monthly_equivalent").reset_index(drop=True) if out else pd.DataFrame()
