"""
Customer Lifetime Value (CLV) — Model & Utilities
Author: Ankit Kumar (github.com/hrts51)

CLV Formula used:
    CLV = (avg_purchase_value × purchase_frequency_per_year × customer_lifespan_years)

Where:
    purchase_frequency_per_year = 365 / purchase_frequency_days
    customer_lifespan_years     = 1 + (previous_purchases / purchase_frequency_per_year) / 12
    avg_purchase_value          = purchase_amount

Then a Gradient Boosting Regressor is trained to predict CLV
from behavioural features.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


def compute_clv(df: pd.DataFrame) -> pd.DataFrame:
    """Add clv_estimated and clv_tier columns to df."""
    df = df.copy()

    freq_map = {
        "Weekly": 7, "Fortnightly": 14, "Bi-Weekly": 14,
        "Monthly": 30, "Every 3 Months": 90,
        "Quarterly": 90, "Annually": 365
    }

    if "purchase_frequency_days" not in df.columns:
        df["purchase_frequency_days"] = df["frequency_of_purchases"].map(freq_map)

    # purchases per year
    df["purchase_freq_per_year"] = 365 / df["purchase_frequency_days"]

    # estimated lifespan in years based on engagement history
    df["lifespan_years"] = 1 + (df["previous_purchases"] / df["purchase_freq_per_year"]) / 12

    # CLV = spend × frequency/yr × lifespan
    df["clv_estimated"] = (
        df["purchase_amount"] *
        df["purchase_freq_per_year"] *
        df["lifespan_years"]
    ).round(2)

    # CLV tiers
    q1 = df["clv_estimated"].quantile(0.25)
    q3 = df["clv_estimated"].quantile(0.75)

    def clv_tier(v):
        if v >= q3:    return "High Value"
        elif v >= q1:  return "Mid Value"
        else:          return "Low Value"

    df["clv_tier"] = df["clv_estimated"].apply(clv_tier)
    return df


def train_clv_model(df: pd.DataFrame):
    """
    Train a Gradient Boosting Regressor to predict CLV.
    Returns: model, X_test, y_test, y_pred, feature_names, metrics
    """
    df = compute_clv(df)

    feature_cols = [
        "age", "purchase_amount", "review_rating", "previous_purchases",
        "purchase_frequency_days", "gender", "category",
        "subscription_status", "shipping_type", "discount_applied", "payment_method"
    ]

    model_df = df[feature_cols + ["clv_estimated"]].copy()

    cat_cols = ["gender", "category", "subscription_status",
                "shipping_type", "discount_applied", "payment_method"]
    le = LabelEncoder()
    for col in cat_cols:
        model_df[col] = le.fit_transform(model_df[col].astype(str))

    X = model_df.drop(columns=["clv_estimated"])
    y = model_df["clv_estimated"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "R² Score": round(r2_score(y_test, y_pred), 4),
        "MAE":      round(mean_absolute_error(y_test, y_pred), 2),
        "Mean CLV": round(y.mean(), 2),
        "Max CLV":  round(y.max(), 2),
        "Min CLV":  round(y.min(), 2),
    }

    return model, X_test, y_test, y_pred, X.columns.tolist(), metrics
