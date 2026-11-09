"""Anomalous spend detection: robust z-scores of each transaction against its own category."""
from __future__ import annotations

import numpy as np
import pandas as pd


def flag_spend(df: pd.DataFrame, category_col: str = "pred_category", threshold: float = 4.0, min_n: int = 20) -> pd.DataFrame:
    """Flag outflows far outside what is normal for their category.

    Uses median and MAD on log amounts, so a few huge transactions cannot hide themselves by
    inflating the baseline. Categories with too little history are skipped rather than guessed at.
    """
    spend = df[df["amount"] < 0].copy()
    spend["log_amt"] = np.log(spend["amount"].abs())
    flagged = []
    for cat, g in spend.groupby(category_col):
        if len(g) < min_n:
            continue
        med = g["log_amt"].median()
        mad = (g["log_amt"] - med).abs().median() * 1.4826
        if mad == 0:
            continue
        z = (g["log_amt"] - med) / mad
        hit = g[z > threshold].assign(z=z[z > threshold])
        flagged.append(hit)
    cols = ["date", "description", "amount", category_col, "z"]
    return pd.concat(flagged)[cols].sort_values("z", ascending=False) if flagged else pd.DataFrame(columns=cols)
