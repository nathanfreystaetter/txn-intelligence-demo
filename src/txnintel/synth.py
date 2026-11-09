"""Synthetic bank transactions for a small business. Fully generated, seeded, no real data.

Amounts follow the bank-feed convention: inflows positive, outflows negative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["revenue", "payroll", "rent", "software", "cloud", "marketing", "supplies", "transfer", "tax", "other"]

_MERCHANTS = {
    "revenue": ["STRIPE PAYOUT", "SQUARE DEPOSIT", "SHOPIFY PAYOUT", "ACH CREDIT CUSTOMER"],
    "payroll": ["GUSTO PAYROLL", "ADP PAYROLL"],
    "rent": ["LANDLORD PROPERTIES RENT", "WEWORK MEMBERSHIP"],
    "software": ["NOTION LABS", "SLACK TECHNOLOGIES", "ZOOM VIDEO", "QUICKBOOKS ONLINE", "ADOBE SYSTEMS"],
    "cloud": ["AMAZON WEB SERVICES", "GOOGLE CLOUD", "DIGITALOCEAN"],
    "marketing": ["GOOGLE ADS", "META ADS", "MAILCHIMP"],
    "supplies": ["STAPLES", "AMAZON MKTPLACE", "COSTCO WHOLESALE", "HOME DEPOT"],
    "transfer": ["TRANSFER TO SAVINGS", "TRANSFER FROM SAVINGS", "INTERNAL TRANSFER"],
    "tax": ["IRS USATAXPYMT", "STATE DEPT OF REVENUE"],
    "other": ["UBER TRIP", "DOORDASH", "UNITED AIRLINES", "MARRIOTT HOTELS", "SHELL OIL"],
}


def generate(months: int = 24, seed: int = 7, start: str = "2023-01-01") -> pd.DataFrame:
    """Return a labeled transaction frame: date, description, amount, category."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=int(months * 30.4), freq="D")
    rows: list[tuple] = []

    def add(d, merchant, amount, cat):
        rows.append((d, f"{merchant} {rng.integers(1000, 9999)}", round(float(amount), 2), cat))

    for d in days:
        month_pos = (d.month - 1) / 12
        season = 1 + 0.35 * np.sin(2 * np.pi * (month_pos - 0.2))  # peak mid-year
        growth = 1 + 0.015 * ((d.year - days[0].year) * 12 + d.month)
        # revenue deposits: most weekdays
        if d.weekday() < 5 and rng.random() < 0.8:
            add(d, rng.choice(_MERCHANTS["revenue"]), rng.gamma(4, 450) * season * growth, "revenue")
        # recurring: payroll on 1st and 15th, rent on the 1st, software monthly
        if d.day in (1, 15):
            add(d, "GUSTO PAYROLL", -(18000 + rng.normal(0, 350)), "payroll")
        if d.day == 1:
            add(d, "LANDLORD PROPERTIES RENT", -4200, "rent")
        if d.day == 3:
            for m, price in (("NOTION LABS", 96), ("SLACK TECHNOLOGIES", 131), ("QUICKBOOKS ONLINE", 85)):
                add(d, m, -(price + rng.normal(0, 2)), "software")
        if d.day == 10:
            add(d, "ACME CORP RETAINER", 6000 + rng.normal(0, 60), "revenue")
        if d.day == 5:
            add(d, "AMAZON WEB SERVICES", -(2400 + 40 * rng.normal()) * growth, "cloud")
        if d.weekday() == 0 and rng.random() < 0.9:
            add(d, rng.choice(_MERCHANTS["marketing"]), -rng.gamma(3, 220), "marketing")
        # irregular spend
        if rng.random() < 0.22:
            add(d, rng.choice(_MERCHANTS["supplies"]), -rng.gamma(2.5, 140), "supplies")
        if rng.random() < 0.10:
            add(d, rng.choice(_MERCHANTS["other"]), -rng.gamma(2, 70), "other")
        if rng.random() < 0.04:
            add(d, rng.choice(_MERCHANTS["transfer"]), rng.choice([-1, 1]) * rng.uniform(2000, 15000), "transfer")
        if d.month in (1, 4, 6, 9) and d.day == 15:
            add(d, rng.choice(_MERCHANTS["tax"]), -rng.uniform(6000, 11000), "tax")

    df = pd.DataFrame(rows, columns=["date", "description", "amount", "category"])

    # real feeds have vague descriptions: blur a share of the irregular outflows so the task is not trivial
    vague = df.index[(df.category.isin(["supplies", "other", "marketing", "software"])) & (rng.random(len(df)) < 0.12)]
    df.loc[vague, "description"] = [f"{rng.choice(['CARD PURCHASE', 'POS DEBIT', 'ACH DEBIT'])} {rng.integers(1000, 9999)}" for _ in vague]

    # plant anomalies: three unusually large spend events, labeled for evaluation
    idx = rng.choice(df.index[(df.category == "marketing")], 3, replace=False)
    df.loc[idx, "amount"] = -rng.uniform(8000, 15000, size=3).round(2)
    df["is_anomaly"] = False
    df.loc[idx, "is_anomaly"] = True
    return df.sort_values("date").reset_index(drop=True)
