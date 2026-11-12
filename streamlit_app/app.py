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
from clv_model import compute_clv, train_clv_model
from ab_test import run_ab_test, run_all_tests
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

    # CLV
    df = compute_clv(df)

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Revenue Overview",
    "👥 Customer Segments",
    "🎯 RFM Analysis",
    "⚠️ Churn Risk",
    "🗺️ Geographic Map",
    "💰 CLV Prediction",
    "🧪 A/B Testing",
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
# TAB 5 — Geographic Map
# ════════════════════════════════════════════════════════
with tab5:
    st.markdown("Revenue, customer count and avg spend mapped across all 50 US states.")

    # Full state name → abbreviation mapping
    state_abbrev = {
        "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
        "Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA",
        "Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA",
        "Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD",
        "Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS",
        "Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH",
        "New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC",
        "North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA",
        "Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN",
        "Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA",
        "West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY"
    }

    state_data = filtered.groupby("location").agg(
        total_revenue   = ("purchase_amount", "sum"),
        total_customers = ("customer_id",     "count"),
        avg_spend       = ("purchase_amount", "mean"),
        avg_rating      = ("review_rating",   "mean"),
        churn_count     = ("churn_risk",      "sum")
    ).reset_index()
    state_data.columns = ["state","total_revenue","total_customers","avg_spend","avg_rating","churn_count"]
    state_data["state_code"]   = state_data["state"].map(state_abbrev)
    state_data["avg_spend"]    = state_data["avg_spend"].round(1)
    state_data["avg_rating"]   = state_data["avg_rating"].round(2)
    state_data["churn_rate"]   = (state_data["churn_count"] / state_data["total_customers"] * 100).round(1)
    state_data["revenue_fmt"]  = state_data["total_revenue"].apply(lambda x: f"${x:,.0f}")

    map_metric = st.radio(
        "Color map by:",
        ["Total Revenue", "Total Customers", "Avg Spend", "Churn Rate %"],
        horizontal=True
    )

    metric_col_map = {
        "Total Revenue":    ("total_revenue",    "Blues",   "$%{customdata[0]:,.0f}"),
        "Total Customers":  ("total_customers",  "Greens",  "%{customdata[1]:,}"),
        "Avg Spend":        ("avg_spend",        "Oranges", "$%{customdata[2]:.1f}"),
        "Churn Rate %":     ("churn_rate",       "Reds",    "%{customdata[3]:.1f}%"),
    }
    col, scale, _ = metric_col_map[map_metric]

    fig_map = px.choropleth(
        state_data,
        locations="state_code",
        locationmode="USA-states",
        color=col,
        scope="usa",
        color_continuous_scale=scale,
        hover_name="state",
        hover_data={
            "state_code":     False,
            "total_revenue":  True,
            "total_customers":True,
            "avg_spend":      True,
            "churn_rate":     True,
        },
        labels={
            "total_revenue":   "Revenue (USD)",
            "total_customers": "Customers",
            "avg_spend":       "Avg Spend (USD)",
            "churn_rate":      "Churn Rate %",
        },
        title=f"US States — {map_metric}"
    )
    fig_map.update_layout(
        geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)",
                 landcolor="#1e2535", showlakes=True),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(tickfont=dict(color="#cbd5e1"), title=dict(font=dict(color="#cbd5e1")))
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Top / Bottom states table
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">🏆 Top 10 States by Revenue</div>', unsafe_allow_html=True)
        top_states = state_data.nlargest(10, "total_revenue")[
            ["state","total_customers","total_revenue","avg_spend","churn_rate"]
        ].rename(columns={
            "state":"State","total_customers":"Customers",
            "total_revenue":"Revenue (USD)","avg_spend":"Avg Spend","churn_rate":"Churn %"
        })
        top_states["Revenue (USD)"] = top_states["Revenue (USD)"].apply(lambda x: f"${x:,.0f}")
        top_states["Avg Spend"]     = top_states["Avg Spend"].apply(lambda x: f"${x:.1f}")
        top_states["Churn %"]       = top_states["Churn %"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(top_states, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-header">📉 Bottom 10 States by Revenue</div>', unsafe_allow_html=True)
        bot_states = state_data.nsmallest(10, "total_revenue")[
            ["state","total_customers","total_revenue","avg_spend","churn_rate"]
        ].rename(columns={
            "state":"State","total_customers":"Customers",
            "total_revenue":"Revenue (USD)","avg_spend":"Avg Spend","churn_rate":"Churn %"
        })
        bot_states["Revenue (USD)"] = bot_states["Revenue (USD)"].apply(lambda x: f"${x:,.0f}")
        bot_states["Avg Spend"]     = bot_states["Avg Spend"].apply(lambda x: f"${x:.1f}")
        bot_states["Churn %"]       = bot_states["Churn %"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(bot_states, use_container_width=True, hide_index=True)

    # Bar chart — top 15 states
    st.markdown('<div class="section-header">Top 15 States — Revenue Breakdown</div>', unsafe_allow_html=True)
    top15 = state_data.nlargest(15, "total_revenue").sort_values("total_revenue", ascending=True)
    fig_bar = px.bar(
        top15, x="total_revenue", y="state", orientation="h",
        color="avg_spend", color_continuous_scale="Viridis",
        labels={"total_revenue":"Revenue (USD)","state":"","avg_spend":"Avg Spend"},
        text="total_revenue",
        hover_data={"total_customers":True,"avg_spend":True,"churn_rate":True}
    )
    fig_bar.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1", height=480,
        margin=dict(l=0,r=80,t=10,b=0),
        coloraxis_colorbar=dict(title="Avg Spend",tickfont=dict(color="#cbd5e1"))
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 6 — CLV Prediction
# ════════════════════════════════════════════════════════
with tab6:
    st.markdown("""
    **Customer Lifetime Value (CLV)** estimates the total revenue a customer will generate over their relationship with the business.
    Calculated as: `CLV = Avg Purchase × Purchase Frequency/Year × Estimated Lifespan (Years)`
    Then a **Gradient Boosting Regressor** is trained to predict CLV from behavioural features.
    """)

    # ── KPIs ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean CLV",   f"${filtered['clv_estimated'].mean():,.0f}")
    c2.metric("Max CLV",    f"${filtered['clv_estimated'].max():,.0f}")
    c3.metric("Min CLV",    f"${filtered['clv_estimated'].min():,.0f}")
    c4.metric("Total Proj. Revenue", f"${filtered['clv_estimated'].sum():,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">CLV Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(filtered, x="clv_estimated", nbins=40,
                           color_discrete_sequence=["#a78bfa"],
                           labels={"clv_estimated": "Estimated CLV (USD)"})
        fig.add_vline(x=filtered["clv_estimated"].mean(), line_dash="dash",
                      line_color="#fbbf24",
                      annotation_text=f"Mean ${filtered['clv_estimated'].mean():,.0f}")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#cbd5e1", height=320, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">CLV Tier Breakdown</div>', unsafe_allow_html=True)
        tier_counts = filtered["clv_tier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        tier_colors = {"High Value":"#4ade80","Mid Value":"#fbbf24","Low Value":"#f87171"}
        fig = px.pie(tier_counts, names="tier", values="count",
                     color="tier", color_discrete_map=tier_colors, hole=0.45)
        fig.update_traces(textinfo="percent+label", textfont_size=13)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#cbd5e1", height=320, showlegend=False,
                          margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Avg CLV by Category</div>', unsafe_allow_html=True)
        clv_cat = filtered.groupby("category")["clv_estimated"].mean().reset_index().sort_values("clv_estimated", ascending=True)
        fig = px.bar(clv_cat, x="clv_estimated", y="category", orientation="h",
                     color="clv_estimated", color_continuous_scale="Purples",
                     labels={"clv_estimated":"Avg CLV (USD)","category":""},
                     text="clv_estimated")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#cbd5e1", height=300, margin=dict(l=0,r=70,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Avg CLV by Subscription Status</div>', unsafe_allow_html=True)
        clv_sub = filtered.groupby("subscription_status")["clv_estimated"].mean().reset_index()
        fig = px.bar(clv_sub, x="subscription_status", y="clv_estimated",
                     color="subscription_status",
                     color_discrete_map={"Yes":"#4ade80","No":"#f87171"},
                     labels={"clv_estimated":"Avg CLV (USD)","subscription_status":""},
                     text="clv_estimated")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                          height=300, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">CLV vs Previous Purchases (by Tier)</div>', unsafe_allow_html=True)
    sample = filtered.sample(min(600, len(filtered)), random_state=42)
    fig = px.scatter(sample, x="previous_purchases", y="clv_estimated",
                     color="clv_tier", color_discrete_map=tier_colors,
                     size="purchase_amount", hover_data=["age","gender","category"],
                     labels={"previous_purchases":"Previous Purchases",
                             "clv_estimated":"Estimated CLV (USD)", "clv_tier":"Tier"},
                     opacity=0.7)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#cbd5e1", height=380, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # ── ML Model ──
    st.markdown("---")
    st.markdown("### 🤖 Gradient Boosting CLV Prediction Model")
    st.caption("Trained on full dataset — predicts CLV from 11 customer features")

    with st.spinner("Training model..."):
        model, X_test, y_test, y_pred, feat_names, metrics = train_clv_model(df)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("R² Score", metrics["R² Score"],
               help="1.0 = perfect prediction. >0.85 is excellent.")
    mc2.metric("Mean Abs Error", f"${metrics['MAE']:,.0f}",
               help="Average prediction error in USD")
    mc3.metric("Dataset Mean CLV", f"${metrics['Mean CLV']:,.0f}")

    col5, col6 = st.columns(2)

    with col5:
        st.markdown('<div class="section-header">Actual vs Predicted CLV</div>', unsafe_allow_html=True)
        pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
        fig = px.scatter(pred_df, x="Actual", y="Predicted",
                         opacity=0.5, color_discrete_sequence=["#60a5fa"])
        # Perfect prediction line
        max_val = max(pred_df["Actual"].max(), pred_df["Predicted"].max())
        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(color="#f87171", dash="dash", width=2))
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#cbd5e1", height=360,
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        importance_df = pd.DataFrame({
            "Feature": feat_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Viridis",
                     labels={"Importance":"Importance Score","Feature":""})
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#cbd5e1", height=360,
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Top high value customers
    st.markdown('<div class="section-header">Top 20 Highest CLV Customers</div>', unsafe_allow_html=True)
    top_clv = filtered.nlargest(20, "clv_estimated")[
        ["customer_id","age","gender","category","purchase_amount",
         "previous_purchases","subscription_status","clv_estimated","clv_tier","rfm_segment"]
    ].copy()
    top_clv["clv_estimated"] = top_clv["clv_estimated"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(top_clv, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════
# TAB 7 — A/B Testing
# ════════════════════════════════════════════════════════
with tab7:
    st.markdown("""
    **A/B Testing** checks whether differences between groups are *statistically significant*
    or just random noise. Uses **Welch's t-test**, **Cohen's d effect size**, and
    **Bootstrap 95% Confidence Intervals**.
    """)

    # ── Run all pre-defined tests ──
    with st.spinner("Running statistical tests..."):
        all_results = run_all_tests(filtered)

    # ── Summary table ──
    st.markdown('<div class="section-header">📋 All Tests — Summary</div>', unsafe_allow_html=True)

    summary_rows = []
    for test_name, r in all_results.items():
        summary_rows.append({
            "Test": test_name,
            f"Mean {r['label_a']}": f"${r['mean_a']:.2f}",
            f"Mean {r['label_b']}": f"${r['mean_b']:.2f}",
            "Difference": f"${r['mean_diff']:+.2f}",
            "p-value": r["p_value"],
            "Significant?": "✅ Yes" if r["significant"] else "❌ No",
            "Effect Size": r["effect_size"],
            "Cohen's d": r["cohens_d"],
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Custom A/B test builder ──
    st.markdown('<div class="section-header">🔬 Custom A/B Test Builder</div>', unsafe_allow_html=True)
    st.caption("Pick any two groups and run your own test on the filtered data.")

    col_left, col_right = st.columns(2)
    with col_left:
        split_col = st.selectbox("Split by column:", [
            "discount_applied", "subscription_status", "gender",
            "season", "category", "shipping_type", "payment_method"
        ])
    with col_right:
        metric_col = st.selectbox("Measure:", ["purchase_amount", "review_rating", "previous_purchases"])

    unique_vals = sorted(filtered[split_col].dropna().unique().tolist())
    if len(unique_vals) >= 2:
        col_a, col_b = st.columns(2)
        with col_a:
            val_a = st.selectbox("Group A:", unique_vals, index=0)
        with col_b:
            val_b = st.selectbox("Group B:", unique_vals, index=min(1, len(unique_vals)-1))

        if val_a != val_b:
            group_a = filtered[filtered[split_col] == val_a][metric_col].values
            group_b = filtered[filtered[split_col] == val_b][metric_col].values

            if len(group_a) >= 10 and len(group_b) >= 10:
                result = run_ab_test(group_a, group_b, str(val_a), str(val_b))

                # Result cards
                r1, r2, r3, r4 = st.columns(4)
                r1.metric(f"Mean — {val_a}", f"${result['mean_a']:.2f}" if metric_col == "purchase_amount" else f"{result['mean_a']:.2f}")
                r2.metric(f"Mean — {val_b}", f"${result['mean_b']:.2f}" if metric_col == "purchase_amount" else f"{result['mean_b']:.2f}")
                r3.metric("p-value", result["p_value"],
                          delta="Significant ✅" if result["significant"] else "Not Significant ❌",
                          delta_color="normal" if result["significant"] else "off")
                r4.metric("Cohen's d", f"{result['cohens_d']:.3f}", delta=result["effect_size"])

                # Verdict banner
                if result["significant"]:
                    st.success(f"✅ **Statistically Significant** (p={result['p_value']}) — "
                               f"The difference of **${result['mean_diff']:+.2f}** between "
                               f"*{val_a}* and *{val_b}* is real, not random. "
                               f"Effect size: **{result['effect_size']}** (Cohen's d = {result['cohens_d']:.3f})")
                else:
                    st.warning(f"❌ **Not Statistically Significant** (p={result['p_value']}) — "
                               f"The difference of **${result['mean_diff']:+.2f}** could be due to chance. "
                               f"Effect size: **{result['effect_size']}** (Cohen's d = {result['cohens_d']:.3f})")

                st.info(f"**95% Bootstrap CI on difference:** "
                        f"[${result['ci_low']:+.2f}, ${result['ci_high']:+.2f}]  |  "
                        f"n({val_a}) = {result['n_a']:,}  |  n({val_b}) = {result['n_b']:,}")

                # Distribution comparison
                col_v1, col_v2 = st.columns(2)

                with col_v1:
                    st.markdown('<div class="section-header">Distribution Comparison</div>', unsafe_allow_html=True)
                    import plotly.figure_factory as ff
                    fig = ff.create_distplot(
                        [group_a.tolist(), group_b.tolist()],
                        group_labels=[str(val_a), str(val_b)],
                        colors=["#60a5fa", "#f472b6"],
                        show_hist=False, show_rug=False
                    )
                    fig.add_vline(x=result["mean_a"], line_dash="dash",
                                  line_color="#60a5fa",
                                  annotation_text=f"μ={result['mean_a']:.1f}")
                    fig.add_vline(x=result["mean_b"], line_dash="dash",
                                  line_color="#f472b6",
                                  annotation_text=f"μ={result['mean_b']:.1f}")
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#cbd5e1", height=340,
                        margin=dict(l=0,r=0,t=10,b=0),
                        legend=dict(font=dict(color="#cbd5e1"))
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_v2:
                    st.markdown('<div class="section-header">Box Plot Comparison</div>', unsafe_allow_html=True)
                    box_df = pd.DataFrame({
                        "value": np.concatenate([group_a, group_b]),
                        "group": [str(val_a)] * len(group_a) + [str(val_b)] * len(group_b)
                    })
                    fig = px.box(box_df, x="group", y="value",
                                 color="group",
                                 color_discrete_map={str(val_a): "#60a5fa", str(val_b): "#f472b6"},
                                 points="outliers",
                                 labels={"value": metric_col.replace("_", " ").title(), "group": ""})
                    fig.update_layout(
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#cbd5e1", height=340,
                        margin=dict(l=0,r=0,t=10,b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data in one or both groups (need at least 10 each). Try adjusting filters.")
        else:
            st.info("Select two different groups to compare.")

    st.markdown("---")

    # ── Effect size guide ──
    st.markdown('<div class="section-header">📖 How to Read Results</div>', unsafe_allow_html=True)
    guide_col1, guide_col2 = st.columns(2)

    with guide_col1:
        st.markdown("""
        **p-value (significance threshold = 0.05)**
        - `p < 0.05` → Statistically significant ✅
        - `p ≥ 0.05` → Not significant ❌
        - Lower p = stronger evidence against chance

        **What it means:**
        If p < 0.05, there's less than 5% probability that the
        observed difference happened by random chance.
        """)

    with guide_col2:
        st.markdown("""
        **Cohen's d (Effect Size)**
        | Range | Interpretation |
        |---|---|
        | < 0.2 | Negligible |
        | 0.2 – 0.5 | Small |
        | 0.5 – 0.8 | Medium |
        | > 0.8 | Large |

        **95% Confidence Interval:**
        Range where the true mean difference lies 95% of the time.
        If it doesn't cross 0 → significant.
        """)


# ════════════════════════════════════════════════════════
# TAB 8 — Raw Data
# ════════════════════════════════════════════════════════
with tab8:
    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} records** (after filters)")
    st.dataframe(
        filtered[[
            "customer_id","age","gender","category","item_purchased",
            "purchase_amount","season","review_rating","subscription_status",
            "shipping_type","payment_method","previous_purchases",
            "loyalty_tier","rfm_segment","churn_risk","clv_estimated","clv_tier"
        ]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data as CSV", csv,
                       "filtered_customers.csv", "text/csv")
