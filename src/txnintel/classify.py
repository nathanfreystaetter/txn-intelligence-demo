"""Transaction classification: description text plus direction of money flow -> category."""
from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline


def normalize(description: str, amount: float) -> str:
    """Strip reference numbers, keep merchant words, and append the direction of the money."""
    text = re.sub(r"\d+", " ", description.upper())
    return f"{text.strip()} {'INFLOW' if amount > 0 else 'OUTFLOW'}"


def _texts(df: pd.DataFrame) -> list[str]:
    return [normalize(d, a) for d, a in zip(df["description"], df["amount"])]


def build_model() -> object:
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=10.0),
    )


def evaluate_holdout(df: pd.DataFrame, train_frac: float = 0.7) -> dict:
    """Time-ordered split: train on the first part of the feed, score on the later part."""
    df = df.sort_values("date").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    train, test = df.iloc[:cut], df.iloc[cut:]
    model = build_model().fit(_texts(train), train["category"])
    pred = model.predict(_texts(test))
    return {
        "n_train": len(train),
        "n_test": len(test),
        "accuracy": float(accuracy_score(test["category"], pred)),
        "macro_f1": float(f1_score(test["category"], pred, average="macro")),
    }


def predict_out_of_fold(df: pd.DataFrame, folds: int = 5) -> pd.Series:
    """Predicted category for every row, each from a model that never saw that row."""
    pred = cross_val_predict(build_model(), _texts(df), df["category"], cv=folds)
    return pd.Series(pred, index=df.index, name="pred_category")
