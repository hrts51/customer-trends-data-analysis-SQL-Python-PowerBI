"""
Standalone script — generates CLV charts as PNG files in plots/
Run from: streamlit_app/ folder
Author: Ankit Kumar (github.com/hrts51)
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from clv_model import compute_clv, train_clv_model
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

df = compute_clv(df)

# ── Chart 11: CLV Distribution + Tier ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["clv_estimated"], bins=40, color="#a78bfa", edgecolor="white", linewidth=0.5)
axes[0].axvline(df["clv_estimated"].mean(), color="#fbbf24", linestyle="--",
                label=f"Mean: ${df['clv_estimated'].mean():,.0f}")
axes[0].set_title("CLV Distribution")
axes[0].set_xlabel("Estimated CLV (USD)")
axes[0].set_ylabel("Count")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[0].legend()

tier_counts = df["clv_tier"].value_counts()
tier_colors = {"High Value": "#4ade80", "Mid Value": "#fbbf24", "Low Value": "#f87171"}
colors = [tier_colors[t] for t in tier_counts.index]
axes[1].pie(tier_counts.values, labels=tier_counts.index, autopct="%1.1f%%",
            startangle=90, colors=colors,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("CLV Tier Breakdown")

plt.tight_layout()
plt.savefig("../plots/11_clv_distribution_tiers.png", bbox_inches="tight")
plt.close()
print("Saved: 11_clv_distribution_tiers.png")

# ── Chart 12: CLV by Category + Subscription ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

clv_cat = df.groupby("category")["clv_estimated"].mean().sort_values(ascending=False)
sns.barplot(x=clv_cat.values, y=clv_cat.index, ax=axes[0], palette="Purples_r")
axes[0].set_title("Avg CLV by Category")
axes[0].set_xlabel("Avg CLV (USD)")
axes[0].set_ylabel("")
for i, v in enumerate(clv_cat.values):
    axes[0].text(v + 10, i, f"${v:,.0f}", va="center", fontsize=9)

clv_sub = df.groupby("subscription_status")["clv_estimated"].mean()
bar_colors = ["#f87171", "#4ade80"] if "No" in clv_sub.index else ["#4ade80"]
axes[1].bar(clv_sub.index, clv_sub.values, color=bar_colors, edgecolor="white", width=0.5)
for i, (idx, v) in enumerate(clv_sub.items()):
    axes[1].text(i, v + 10, f"${v:,.0f}", ha="center", fontsize=10)
axes[1].set_title("Avg CLV: Subscribed vs Not")
axes[1].set_xlabel("Subscription Status")
axes[1].set_ylabel("Avg CLV (USD)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

plt.tight_layout()
plt.savefig("../plots/12_clv_by_category_subscription.png", bbox_inches="tight")
plt.close()
print("Saved: 12_clv_by_category_subscription.png")

# ── Chart 13: Actual vs Predicted CLV + Feature Importance ───────────────────
model, X_test, y_test, y_pred, feat_names, metrics = train_clv_model(df)
print(f"\nModel Metrics: R²={metrics['R² Score']} | MAE=${metrics['MAE']:,.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, y_pred, alpha=0.4, color="#60a5fa", s=20)
max_val = max(y_test.max(), max(y_pred))
axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
axes[0].set_title(f"Actual vs Predicted CLV  (R²={metrics['R² Score']})")
axes[0].set_xlabel("Actual CLV (USD)")
axes[0].set_ylabel("Predicted CLV (USD)")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[0].legend()

import pandas as pd
importance_df = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=True)
importance_df.plot(kind="barh", ax=axes[1], color="#a78bfa", edgecolor="white")
axes[1].set_title("Feature Importance — Gradient Boosting CLV Model")
axes[1].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("../plots/13_clv_model_results.png", bbox_inches="tight")
plt.close()
print("Saved: 13_clv_model_results.png")
