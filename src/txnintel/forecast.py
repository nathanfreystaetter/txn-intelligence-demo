"""Revenue forecasting with trend and seasonality, backtested against honest baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _design(t: np.ndarray, month: np.ndarray, harmonics: int) -> np.ndarray:
    cols = [np.ones_like(t, dtype=float), t.astype(float)]
    for k in range(1, harmonics + 1):
        cols += [np.sin(2 * np.pi * k * month / 12), np.cos(2 * np.pi * k * month / 12)]
    return np.column_stack(cols)


def fit_forecast(monthly: pd.Series, horizon: int = 6, harmonics: int = 2, z: float = 1.645) -> pd.DataFrame:
    """Log-linear trend plus Fourier seasonality. Returns point forecast and an ~90% interval."""
    y = np.log(monthly.to_numpy(dtype=float))
    t = np.arange(len(y))
    X = _design(t, monthly.index.month.to_numpy(), harmonics)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma = float(np.sqrt((resid**2).sum() / dof))
    future_idx = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    Xf = _design(np.arange(len(y), len(y) + horizon), future_idx.month.to_numpy(), harmonics)
    mu = Xf @ beta
    return pd.DataFrame({"forecast": np.exp(mu), "lo": np.exp(mu - z * sigma), "hi": np.exp(mu + z * sigma)}, index=future_idx)


def backtest(monthly: pd.Series, holdout: int = 6, harmonics: int = 2) -> dict:
    """Hold out the last months; compare against 'repeat last value' and 'same month last year'."""
    train, test = monthly.iloc[:-holdout], monthly.iloc[-holdout:]
    model = fit_forecast(train, horizon=holdout, harmonics=harmonics)["forecast"].to_numpy()
    naive = np.repeat(train.iloc[-1], holdout)
    seasonal_naive = monthly.shift(12).iloc[-holdout:].to_numpy()
    mape = lambda p: float(np.mean(np.abs(test.to_numpy() - p) / test.to_numpy()))
    out = {"model_mape": mape(model), "naive_mape": mape(naive)}
    if not np.isnan(seasonal_naive).any():
        out["seasonal_naive_mape"] = mape(seasonal_naive)
    return out
