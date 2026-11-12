"""
Standalone script — generates US geographic revenue map as HTML + PNG
Run from: streamlit_app/ folder
Author: Ankit Kumar (github.com/hrts51)
"""

import pandas as pd
import plotly.express as px
import os

# ── Load & prep data ──────────────────────────────────────────────────────────
df = pd.read_csv("../data/raw/customer_shopping_behavior.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")
df = df.rename(columns={"purchase_amount_(usd)": "purchase_amount"})

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

state_data = df.groupby("location").agg(
    total_revenue   = ("purchase_amount", "sum"),
    total_customers = ("customer_id",     "count"),
    avg_spend       = ("purchase_amount", "mean"),
).reset_index()
state_data.columns      = ["state","total_revenue","total_customers","avg_spend"]
state_data["state_code"] = state_data["state"].map(state_abbrev)
state_data["avg_spend"]  = state_data["avg_spend"].round(1)

# ── Build map ─────────────────────────────────────────────────────────────────
fig = px.choropleth(
    state_data,
    locations="state_code",
    locationmode="USA-states",
    color="total_revenue",
    scope="usa",
    color_continuous_scale="Blues",
    hover_name="state",
    hover_data={
        "state_code":     False,
        "total_revenue":  True,
        "total_customers":True,
        "avg_spend":      True,
    },
    labels={
        "total_revenue":   "Revenue (USD)",
        "total_customers": "Customers",
        "avg_spend":       "Avg Spend (USD)",
    },
    title="Customer Revenue by US State"
)
fig.update_layout(
    geo=dict(bgcolor="rgba(14,17,23,1)", lakecolor="#1e2535",
             landcolor="#1e2535", showlakes=True,
             subunitcolor="#2d3748"),
    paper_bgcolor="rgba(14,17,23,1)",
    font=dict(color="#cbd5e1", size=13),
    title_font=dict(size=18, color="#f1f5f9"),
    height=560,
    margin=dict(l=0, r=0, t=50, b=0),
    coloraxis_colorbar=dict(
        title="Revenue (USD)",
        tickfont=dict(color="#cbd5e1"),
        title_font=dict(color="#cbd5e1")
    )
)

# Save as interactive HTML
out_html = "../plots/10_us_revenue_map.html"
fig.write_html(out_html, include_plotlyjs="cdn")
print(f"Saved interactive map → {out_html}")

# Save as static PNG
out_png = "../plots/10_us_revenue_map.png"
fig.write_image(out_png, width=1200, height=600, scale=2)
print(f"Saved static map    → {out_png}")
