"""
A/B Test Simulation — Statistical Significance Testing
Author: Ankit Kumar (github.com/hrts51)

Tests whether group differences in purchase_amount are statistically significant.
Uses:
  - Welch's two-sample t-test (unequal variance)
  - Cohen's d effect size
  - 95% Confidence Interval on the difference
  - Bootstrap resampling for robust CI estimation
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d effect size between two groups."""
    mean_diff = np.mean(group_a) - np.mean(group_b)
    pooled_std = np.sqrt(
        ((len(group_a) - 1) * np.var(group_a, ddof=1) +
         (len(group_b) - 1) * np.var(group_b, ddof=1)) /
        (len(group_a) + len(group_b) - 2)
    )
    return mean_diff / pooled_std if pooled_std != 0 else 0.0


def effect_label(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:   return "Negligible"
    elif d < 0.5: return "Small"
    elif d < 0.8: return "Medium"
    else:         return "Large"


def bootstrap_ci(group_a: np.ndarray, group_b: np.ndarray,
                 n_boot: int = 2000, ci: float = 0.95) -> Tuple[float, float]:
    """Bootstrap confidence interval on the mean difference (A - B)."""
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_boot):
        a_sample = rng.choice(group_a, size=len(group_a), replace=True)
        b_sample = rng.choice(group_b, size=len(group_b), replace=True)
        diffs.append(a_sample.mean() - b_sample.mean())
    alpha = (1 - ci) / 2
    return float(np.percentile(diffs, alpha * 100)), float(np.percentile(diffs, (1 - alpha) * 100))


def run_ab_test(group_a: np.ndarray, group_b: np.ndarray,
                label_a: str = "Group A", label_b: str = "Group B",
                alpha: float = 0.05) -> Dict:
    """
    Run full A/B test between two groups.
    Returns a results dictionary.
    """
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    d = cohens_d(group_a, group_b)
    ci_low, ci_high = bootstrap_ci(group_a, group_b)

    return {
        "label_a":        label_a,
        "label_b":        label_b,
        "n_a":            len(group_a),
        "n_b":            len(group_b),
        "mean_a":         round(float(np.mean(group_a)), 2),
        "mean_b":         round(float(np.mean(group_b)), 2),
        "mean_diff":      round(float(np.mean(group_a) - np.mean(group_b)), 2),
        "std_a":          round(float(np.std(group_a, ddof=1)), 2),
        "std_b":          round(float(np.std(group_b, ddof=1)), 2),
        "t_statistic":    round(float(t_stat), 4),
        "p_value":        round(float(p_value), 6),
        "significant":    bool(p_value < alpha),
        "cohens_d":       round(d, 4),
        "effect_size":    effect_label(d),
        "ci_low":         round(ci_low, 2),
        "ci_high":        round(ci_high, 2),
        "alpha":          alpha,
    }


def run_all_tests(df: pd.DataFrame) -> Dict[str, Dict]:
    """Run pre-defined A/B tests on the dataset."""
    results = {}

    # Test 1 — Discount vs No Discount
    a = df[df["discount_applied"] == "Yes"]["purchase_amount"].values
    b = df[df["discount_applied"] == "No"]["purchase_amount"].values
    results["Discount Applied vs Not"] = run_ab_test(a, b, "Discount Applied", "No Discount")

    # Test 2 — Subscribed vs Not Subscribed
    a = df[df["subscription_status"] == "Yes"]["purchase_amount"].values
    b = df[df["subscription_status"] == "No"]["purchase_amount"].values
    results["Subscribed vs Not Subscribed"] = run_ab_test(a, b, "Subscribed", "Not Subscribed")

    # Test 3 — Express vs Standard Shipping
    a = df[df["shipping_type"] == "Express"]["purchase_amount"].values
    b = df[df["shipping_type"] == "Standard"]["purchase_amount"].values
    results["Express vs Standard Shipping"] = run_ab_test(a, b, "Express", "Standard")

    # Test 4 — Male vs Female spend
    a = df[df["gender"] == "Male"]["purchase_amount"].values
    b = df[df["gender"] == "Female"]["purchase_amount"].values
    results["Male vs Female Spend"] = run_ab_test(a, b, "Male", "Female")

    # Test 5 — Clothing vs Footwear
    a = df[df["category"] == "Clothing"]["purchase_amount"].values
    b = df[df["category"] == "Footwear"]["purchase_amount"].values
    results["Clothing vs Footwear Revenue"] = run_ab_test(a, b, "Clothing", "Footwear")

    # Test 6 — Loyal vs New customers
    def loyalty(p):
        if p == 1: return "New"
        elif p <= 10: return "Returning"
        else: return "Loyal"
    df = df.copy()
    df["_lt"] = df["previous_purchases"].apply(loyalty)
    a = df[df["_lt"] == "Loyal"]["purchase_amount"].values
    b = df[df["_lt"] == "New"]["purchase_amount"].values
    results["Loyal vs New Customer Spend"] = run_ab_test(a, b, "Loyal", "New")

    return results
