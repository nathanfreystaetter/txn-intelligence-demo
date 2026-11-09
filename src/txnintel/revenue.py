"""Annual revenue estimate from transaction inflows, with an uncertainty band."""
from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_revenue(df: pd.DataFrame, category_col: str = "pred_category") -> pd.Series:
    """Revenue-classified inflows per calendar month. Transfers and refunds are not revenue."""
    rev = df[(df[category_col] == "revenue") & (df["amount"] > 0)]
    return rev.set_index("date")["amount"].resample("MS").sum()


def estimate_annual(df: pd.DataFrame, category_col: str = "pred_category", n_boot: int = 2000, seed: int = 0) -> dict:
    """Trailing twelve months of revenue, with a bootstrap range over the monthly totals.

    The range is a statement of uncertainty, not a guarantee: it answers "how much does this
    estimate move if the months had come out differently", which is what an underwriter needs.
    """
    m = monthly_revenue(df, category_col)
    window = m.iloc[-12:]
    point = float(window.sum())
    rng = np.random.default_rng(seed)
    boots = rng.choice(window.to_numpy(), size=(n_boot, len(window)), replace=True).sum(axis=1)
    lo, hi = np.percentile(boots, [5, 95])
    return {"months_used": int(len(window)), "annual_revenue": point, "p05": float(lo), "p95": float(hi)}
