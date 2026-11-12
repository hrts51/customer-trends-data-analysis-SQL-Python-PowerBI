"""
Standalone script — generates A/B test result charts as PNG files in plots/
Run from: streamlit_app/ folder
Author: Ankit Kumar (github.com/hrts51)
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from ab_test import run_all_tests
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

# Load data
df = pd.read_csv("../data/raw/customer_shopping_behavior.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")
df = df.rename(columns={"purchase_amount_(usd)": "purchase_amount"})
df["review_rating"] = df.groupby("category")["review_rating"].transform(
    lambda x: x.fillna(x.median())
)
if "promo_code_used" in df.columns:
    df = df.drop(columns=["promo_code_used"])

results = run_all_tests(df)

# ── Chart 14: A/B Test Summary — Mean Comparison ──────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

colors_sig   = ["#4ade80", "#f87171"]   # significant
colors_nosig = ["#60a5fa", "#94a3b8"]   # not significant

for i, (test_name, r) in enumerate(results.items()):
    ax = axes[i]
    colors = colors_sig if r["significant"] else colors_nosig
    bars = ax.bar([r["label_a"], r["label_b"]], [r["mean_a"], r["mean_b"]],
                  color=colors, edgecolor="white", width=0.5)

    for bar, val in zip(bars, [r["mean_a"], r["mean_b"]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"${val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    significance = "✓ Significant" if r["significant"] else "✗ Not Significant"
    color_text   = "#4ade80" if r["significant"] else "#f87171"
    ax.set_title(f"{test_name}\np={r['p_value']} | d={r['cohens_d']} ({r['effect_size']})",
                 fontsize=9)
    ax.set_ylabel("Avg Purchase Amount (USD)")
    ax.text(0.5, 0.97, significance, transform=ax.transAxes,
            ha="center", va="top", fontsize=10, color=color_text, fontweight="bold")

plt.suptitle("A/B Test Results — Purchase Amount Comparisons", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("../plots/14_ab_test_summary.png", bbox_inches="tight")
plt.close()
print("Saved: 14_ab_test_summary.png")

# ── Chart 15: CI Plot (Forest Plot style) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

test_names = list(results.keys())
mean_diffs = [r["mean_diff"] for r in results.values()]
ci_lows    = [r["ci_low"]   for r in results.values()]
ci_highs   = [r["ci_high"]  for r in results.values()]
sig_flags  = [r["significant"] for r in results.values()]

y_pos = range(len(test_names))
dot_colors = ["#4ade80" if s else "#f87171" for s in sig_flags]

for i, (y, diff, lo, hi, sig) in enumerate(zip(y_pos, mean_diffs, ci_lows, ci_highs, sig_flags)):
    ax.plot([lo, hi], [y, y], color=dot_colors[i], linewidth=2.5, zorder=2)
    ax.scatter(diff, y, color=dot_colors[i], s=120, zorder=3)
    ax.text(hi + 0.5, y, f"${diff:+.2f}", va="center", fontsize=9)

ax.axvline(0, color="white", linestyle="--", linewidth=1.2, alpha=0.6)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(test_names, fontsize=9)
ax.set_xlabel("Mean Difference in Purchase Amount (USD)\n(Group A − Group B)", fontsize=10)
ax.set_title("Forest Plot — 95% Bootstrap Confidence Intervals\n(Dots left of 0 = Group B higher, right = Group A higher)",
             fontsize=11)

green_patch = mpatches.Patch(color="#4ade80", label="Significant (p < 0.05)")
red_patch   = mpatches.Patch(color="#f87171", label="Not Significant (p ≥ 0.05)")
ax.legend(handles=[green_patch, red_patch], loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig("../plots/15_ab_test_forest_plot.png", bbox_inches="tight")
plt.close()
print("Saved: 15_ab_test_forest_plot.png")

# Print results summary
print("\n── A/B Test Results ──────────────────────────────")
for name, r in results.items():
    sig = "✓ SIGNIFICANT" if r["significant"] else "✗ not significant"
    print(f"{name:40s} | p={r['p_value']:.4f} | d={r['cohens_d']:.3f} ({r['effect_size']}) | {sig}")
