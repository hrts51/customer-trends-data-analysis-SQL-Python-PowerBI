"""
Customer Shopping Behavior Analysis — Streamlit Dashboard
Author : Ankit Kumar (github.com/hrts51)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Shopping Behavior",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: #1e2535;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .stMetric label { font-size: 0.85rem !important; color: #94a3b8 !important; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
    .section-header {
        font-size: 1.1rem; font-weight: 700;
        color: #f1f5f9; margin: 8px 0 4px 0;
        border-left: 3px solid #60a5fa;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading & preprocessing ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("../data/raw/customer_shopping_behavior.csv")

    # Clean
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df = df.rename(columns={"purchase_amount_(usd)": "purchase_amount"})
    df["review_rating"] = df.groupby("category")["review_rating"].transform(
        lambda x: x.fillna(x.median())
    )
    if "promo_code_used" in df.columns:
        df = df.drop(columns=["promo_code_used"])

    # Feature engineering
    labels = ["Young Adult (18-30)", "Adult (31-44)", "Middle-aged (45-57)", "Senior (58-70)"]
    df["age_group"] = pd.qcut(df["age"], q=4, labels=labels)

    freq_map = {"Weekly": 7, "Fortnightly": 14, "Bi-Weekly": 14,
                "Monthly": 30, "Every 3 Months": 90, "Quarterly": 90, "Annually": 365}
    df["purchase_frequency_days"] = df["frequency_of_purchases"].map(freq_map)

    df["spend_tier"] = pd.cut(df["purchase_amount"], bins=[0, 40, 70, 100],
                               labels=["Low (<$40)", "Medium ($40-$70)", "High (>$70)"])

    def loyalty(p):
        if p == 1: return "New"
        elif p <= 10: return "Returning"
        else: return "Loyal"
    df["loyalty_tier"] = df["previous_purchases"].apply(loyalty)

    # RFM
    rfm = df[["customer_id", "purchase_frequency_days", "previous_purchases", "purchase_amount"]].copy()
    rfm.columns = ["customer_id", "recency", "frequency", "monetary"]
    rfm["R"] = pd.qcut(rfm["recency"], q=5, labels=False, duplicates="drop")
    rfm["R"] = (rfm["R"].max() - rfm["R"] + 1).clip(1, 5).astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5]).astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"], q=5, labels=[1,2,3,4,5], duplicates="drop").astype(int)
    rfm["rfm_score"] = rfm["R"] + rfm["F"] + rfm["M"]

    def seg(s):
        if s >= 13: return "Champions"
        elif s >= 10: return "Loyal Customers"
        elif s >= 7: return "Potential Loyalists"
        elif s >= 5: return "At Risk"
        else: return "Lost"
    rfm["rfm_segment"] = rfm["rfm_score"].apply(seg)
    df = df.merge(rfm[["customer_id", "rfm_segment", "rfm_score"]], on="customer_id", how="left")

    # Churn label
    df["churn_risk"] = (
        (df["purchase_frequency_days"] >= 90) &
        (df["review_rating"] < 3.5) &
        (df["subscription_status"] == "No")
    ).astype(int)

    return df

df = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=60)
st.sidebar.title("🛍️ Filters")
st.sidebar.markdown("---")

gender_opts = ["All"] + sorted(df["gender"].unique().tolist())
gender = st.sidebar.selectbox("Gender", gender_opts)

category_opts = ["All"] + sorted(df["category"].unique().tolist())
category = st.sidebar.selectbox("Category", category_opts)

season_opts = ["All"] + sorted(df["season"].unique().tolist())
season = st.sidebar.selectbox("Season", season_opts)

subscription_opts = ["All", "Yes", "No"]
subscription = st.sidebar.selectbox("Subscription Status", subscription_opts)

age_range = st.sidebar.slider("Age Range", int(df["age"].min()), int(df["age"].max()), (18, 70))

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by [Ankit Kumar](https://github.com/hrts51)**")

# Apply filters
filtered = df.copy()
if gender != "All":        filtered = filtered[filtered["gender"] == gender]
if category != "All":      filtered = filtered[filtered["category"] == category]
if season != "All":        filtered = filtered[filtered["season"] == season]
if subscription != "All":  filtered = filtered[filtered["subscription_status"] == subscription]
filtered = filtered[(filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛍️ Customer Shopping Behavior Analysis")
st.caption("Interactive dashboard — use the sidebar filters to explore the data")
st.markdown("---")

# ── KPI Metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Customers",   f"{len(filtered):,}")
c2.metric("Total Revenue",     f"${filtered['purchase_amount'].sum():,.0f}")
c3.metric("Avg Spend",         f"${filtered['purchase_amount'].mean():.1f}")
c4.metric("Avg Rating",        f"{filtered['review_rating'].mean():.2f} ⭐")
c5.metric("Subscribed",        f"{(filtered['subscription_status']=='Yes').sum():,}")
c6.metric("Churn Risk",        f"{filtered['churn_risk'].sum():,}")

st.markdown("---")

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Revenue Overview",
    "👥 Customer Segments",
    "🎯 RFM Analysis",
    "⚠️ Churn Risk",
    "📋 Raw Data"
])

# ════════════════════════════════════════════════════════
# TAB 1 — Revenue Overview
# ════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Revenue by Category</div>', unsafe_allow_html=True)
        cat_rev = filtered.groupby("category")["purchase_amount"].sum().reset_index().sort_values("purchase_amount", ascending=True)
        fig = px.bar(cat_rev, x="purchase_amount", y="category", orientation="h",
                     color="purchase_amount", color_continuous_scale="Blues",
                     labels={"purchase_amount": "Revenue (USD)", "category": ""},
                     text="purchase_amount")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#cbd5e1", height=300, margin=dict(l=0,r=40,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Revenue by Season</div>', unsafe_allow_html=True)
        season_rev = filtered.groupby("season")["purchase_amount"].sum().reset_index()
        colors = {"Spring": "#4CAF50", "Summer": "#FF9800", "Fall": "#F44336", "Winter": "#2196F3"}
        fig = px.bar(season_rev, x="season", y="purchase_amount",
                     color="season", color_discrete_map=colors,
                     labels={"purchase_amount": "Revenue (USD)", "season": ""},
                     text="purchase_amount")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                          height=300, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Revenue by Gender</div>', unsafe_allow_html=True)
        gen_rev = filtered.groupby("gender")["purchase_amount"].sum().reset_index()
        fig = px.pie(gen_rev, names="gender", values="purchase_amount",
                     color_discrete_sequence=["#60a5fa", "#f472b6"],
                     hole=0.45)
        fig.update_traces(textinfo="percent+label", textfont_size=13)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#cbd5e1", height=300, margin=dict(l=0,r=0,t=10,b=0),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Top 10 Products by Revenue</div>', unsafe_allow_html=True)
        top_items = filtered.groupby("item_purchased")["purchase_amount"].sum().nlargest(10).reset_index()
        top_items = top_items.sort_values("purchase_amount", ascending=True)
        fig = px.bar(top_items, x="purchase_amount", y="item_purchased", orientation="h",
                     color="purchase_amount", color_continuous_scale="Teal",
                     labels={"purchase_amount": "Revenue (USD)", "item_purchased": ""},
                     text="purchase_amount")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#cbd5e1", height=350, margin=dict(l=0,r=60,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Purchase Amount Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(filtered, x="purchase_amount", nbins=30,
                       color_discrete_sequence=["#60a5fa"],
                       labels={"purchase_amount": "Purchase Amount (USD)", "count": "Customers"})
    fig.add_vline(x=filtered["purchase_amount"].mean(), line_dash="dash",
                  line_color="#f87171", annotation_text=f"Mean ${filtered['purchase_amount'].mean():.1f}")
    fig.add_vline(x=filtered["purchase_amount"].median(), line_dash="dash",
                  line_color="#fbbf24", annotation_text=f"Median ${filtered['purchase_amount'].median():.1f}")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#cbd5e1", height=280, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 2 — Customer Segments
# ════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Revenue by Age Group</div>', unsafe_allow_html=True)
        age_rev = filtered.groupby("age_group", observed=True)["purchase_amount"].sum().reset_index()
        fig = px.bar(age_rev, x="age_group", y="purchase_amount",
                     color="purchase_amount", color_continuous_scale="Viridis",
                     labels={"purchase_amount": "Revenue (USD)", "age_group": ""},
                     text="purchase_amount")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#cbd5e1", height=320, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Loyalty Tier Distribution</div>', unsafe_allow_html=True)
        loyalty_counts = filtered["loyalty_tier"].value_counts().reset_index()
        loyalty_counts.columns = ["tier", "count"]
        colors_loyalty = {"New": "#f87171", "Returning": "#fbbf24", "Loyal": "#4ade80"}
        fig = px.pie(loyalty_counts, names="tier", values="count",
                     color="tier", color_discrete_map=colors_loyalty, hole=0.4)
        fig.update_traces(textinfo="percent+label", textfont_size=13)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#cbd5e1", height=320, margin=dict(l=0,r=0,t=10,b=0),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Avg Spend by Shipping Type</div>', unsafe_allow_html=True)
        ship = filtered.groupby("shipping_type")["purchase_amount"].mean().reset_index().sort_values("purchase_amount", ascending=True)
        fig = px.bar(ship, x="purchase_amount", y="shipping_type", orientation="h",
                     color="purchase_amount", color_continuous_scale="Reds",
                     labels={"purchase_amount": "Avg Spend (USD)", "shipping_type": ""},
                     text="purchase_amount")
        fig.update_traces(texttemplate="$%{text:.1f}", textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#cbd5e1", height=320, margin=dict(l=0,r=50,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Payment Method Usage</div>', unsafe_allow_html=True)
        pay = filtered["payment_method"].value_counts().reset_index()
        pay.columns = ["method", "count"]
        fig = px.bar(pay, x="count", y="method", orientation="h",
                     color="count", color_continuous_scale="Purples",
                     labels={"count": "Transactions", "method": ""},
                     text="count")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#cbd5e1", height=320, margin=dict(l=0,r=40,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Subscription vs Non-Subscription — Avg Spend & Revenue</div>', unsafe_allow_html=True)
    sub_stats = filtered.groupby("subscription_status").agg(
        total_customers=("customer_id", "count"),
        avg_spend=("purchase_amount", "mean"),
        total_revenue=("purchase_amount", "sum")
    ).reset_index()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Avg Spend", "Total Revenue"],
                        horizontal_spacing=0.15)
    fig.add_trace(go.Bar(x=sub_stats["subscription_status"], y=sub_stats["avg_spend"],
                         marker_color=["#f87171","#4ade80"], text=sub_stats["avg_spend"].round(1),
                         texttemplate="$%{text}", textposition="outside", name="Avg Spend"), row=1, col=1)
    fig.add_trace(go.Bar(x=sub_stats["subscription_status"], y=sub_stats["total_revenue"],
                         marker_color=["#f87171","#4ade80"], text=sub_stats["total_revenue"],
                         texttemplate="$%{text:,.0f}", textposition="outside", name="Total Revenue"), row=1, col=2)
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                      height=320, margin=dict(l=0,r=0,t=30,b=0))
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 3 — RFM Analysis
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    **RFM (Recency · Frequency · Monetary)** scores every customer 1–15 and assigns them to a segment.
    Higher score = more valuable customer.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Customer Count per Segment</div>', unsafe_allow_html=True)
        seg_order = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Lost"]
        seg_colors = {"Champions":"#4ade80","Loyal Customers":"#86efac",
                      "Potential Loyalists":"#fbbf24","At Risk":"#fb923c","Lost":"#f87171"}
        seg_counts = filtered["rfm_segment"].value_counts().reindex(seg_order).reset_index()
        seg_counts.columns = ["segment", "count"]
        fig = px.bar(seg_counts, x="segment", y="count",
                     color="segment", color_discrete_map=seg_colors,
                     labels={"count": "Customers", "segment": ""},
                     text="count")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                          height=340, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Avg Spend per Segment</div>', unsafe_allow_html=True)
        seg_spend = filtered.groupby("rfm_segment")["purchase_amount"].mean().reindex(seg_order).reset_index()
        seg_spend.columns = ["segment", "avg_spend"]
        fig = px.bar(seg_spend, x="avg_spend", y="segment", orientation="h",
                     color="segment", color_discrete_map=seg_colors,
                     labels={"avg_spend": "Avg Purchase (USD)", "segment": ""},
                     text="avg_spend")
        fig.update_traces(texttemplate="$%{text:.1f}", textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                          height=340, margin=dict(l=0,r=60,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">RFM Score Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(filtered, x="rfm_score", nbins=13,
                       color_discrete_sequence=["#60a5fa"],
                       labels={"rfm_score": "RFM Score (3–15)", "count": "Customers"})
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#cbd5e1", height=260, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Segment × Subscription Breakdown</div>', unsafe_allow_html=True)
    pivot = filtered.groupby(["rfm_segment","subscription_status"])["customer_id"].count().reset_index()
    pivot.columns = ["Segment","Subscription","Count"]
    fig = px.bar(pivot, x="Segment", y="Count", color="Subscription",
                 color_discrete_map={"Yes":"#4ade80","No":"#f87171"},
                 barmode="group", category_orders={"Segment": seg_order},
                 labels={"Count":"Customers"})
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#cbd5e1", height=300, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 4 — Churn Risk
# ════════════════════════════════════════════════════════
with tab4:
    churn_count = filtered["churn_risk"].sum()
    total = len(filtered)
    churn_rate = churn_count / total * 100 if total > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("At-Risk Customers", f"{churn_count:,}")
    c2.metric("Safe Customers",    f"{total - churn_count:,}")
    c3.metric("Churn Rate",        f"{churn_rate:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Churn vs Safe</div>', unsafe_allow_html=True)
        churn_df = filtered["churn_risk"].map({1:"At Risk",0:"Safe"}).value_counts().reset_index()
        churn_df.columns = ["status","count"]
        fig = px.pie(churn_df, names="status", values="count",
                     color="status", color_discrete_map={"At Risk":"#f87171","Safe":"#4ade80"},
                     hole=0.5)
        fig.update_traces(textinfo="percent+label", textfont_size=13)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#cbd5e1", height=320, showlegend=False,
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Churn Risk by Category</div>', unsafe_allow_html=True)
        churn_cat = filtered.groupby("category").agg(
            churn=("churn_risk","sum"), total=("customer_id","count")
        ).reset_index()
        churn_cat["churn_rate"] = (churn_cat["churn"] / churn_cat["total"] * 100).round(1)
        fig = px.bar(churn_cat, x="category", y="churn_rate",
                     color="churn_rate", color_continuous_scale="Reds",
                     labels={"churn_rate":"Churn Rate %","category":""},
                     text="churn_rate")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#cbd5e1", height=320, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Churn Risk by Age Group</div>', unsafe_allow_html=True)
    churn_age = filtered.groupby("age_group", observed=True).agg(
        churn=("churn_risk","sum"), total=("customer_id","count")
    ).reset_index()
    churn_age["churn_rate"] = (churn_age["churn"] / churn_age["total"] * 100).round(1)
    fig = px.bar(churn_age, x="age_group", y="churn_rate",
                 color="churn_rate", color_continuous_scale="Oranges",
                 labels={"churn_rate":"Churn Rate %","age_group":""},
                 text="churn_rate")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, coloraxis_showscale=False,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#cbd5e1", height=280, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Top 20 At-Risk Customers</div>', unsafe_allow_html=True)
    at_risk = filtered[filtered["churn_risk"] == 1][
        ["customer_id","age","gender","category","purchase_amount",
         "review_rating","subscription_status","frequency_of_purchases","rfm_segment"]
    ].sort_values("review_rating").head(20)
    st.dataframe(at_risk, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════
# TAB 5 — Raw Data
# ════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} records** (after filters)")
    st.dataframe(
        filtered[[
            "customer_id","age","gender","category","item_purchased",
            "purchase_amount","season","review_rating","subscription_status",
            "shipping_type","payment_method","previous_purchases",
            "loyalty_tier","rfm_segment","churn_risk"
        ]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data as CSV", csv,
                       "filtered_customers.csv", "text/csv")
