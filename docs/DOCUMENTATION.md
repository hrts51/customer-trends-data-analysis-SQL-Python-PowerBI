# 📄 Customer Shopping Behavior Analysis — Full Documentation

**Author:** Ankit Kumar  
**GitHub:** [github.com/hrts51](https://github.com/hrts51)  
**Repository:** [customer-trends-data-analysis-SQL-Python-PowerBI](https://github.com/hrts51/customer-trends-data-analysis-SQL-Python-PowerBI)  
**Stack:** Python 3.13 · SQL · Power BI · Streamlit · GitHub Actions  

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Dataset Overview](#2-dataset-overview)
3. [What Was in the Original Project](#3-what-was-in-the-original-project)
4. [Improvements & New Features](#4-improvements--new-features)
5. [Feature Engineering Reference](#5-feature-engineering-reference)
6. [EDA Visualizations (15 charts)](#6-eda-visualizations)
7. [RFM Customer Segmentation](#7-rfm-customer-segmentation)
8. [Churn Prediction Model](#8-churn-prediction-model)
9. [Customer Lifetime Value (CLV)](#9-customer-lifetime-value-clv)
10. [A/B Testing](#10-ab-testing)
11. [SQL Queries Reference](#11-sql-queries-reference)
12. [Streamlit Dashboard](#12-streamlit-dashboard)
13. [GitHub Actions CI/CD](#13-github-actions-cicd)
14. [File Inventory](#14-file-inventory)
15. [How to Run Locally](#15-how-to-run-locally)
16. [Key Business Insights](#16-key-business-insights)
17. [Git Commit History](#17-git-commit-history)

---

## 1. Project Purpose

An end-to-end data analytics + data science portfolio project built on retail customer transaction data. The project demonstrates every stage of a real corporate analytics workflow:

| Stage | Technique |
|---|---|
| Data preparation | Cleaning, imputation, feature engineering |
| Exploratory analysis | 15 visualizations across 6 categories |
| Customer segmentation | RFM scoring, loyalty tiers, spend tiers |
| Machine learning | Churn prediction (Random Forest), CLV prediction (Gradient Boosting) |
| Statistical testing | A/B tests with Welch's t-test, Cohen's d, Bootstrap CI |
| SQL analytics | 15 business queries across 4 sections |
| Dashboard | Interactive Power BI + Streamlit web app (8 tabs) |
| Automation | GitHub Actions CI — 3 workflows |

---

## 2. Dataset Overview

| Property | Value |
|---|---|
| File | `data/raw/customer_shopping_behavior.csv` |
| Rows | 3,900 |
| Original columns | 18 |
| Final columns | 25+ (after engineering) |
| Missing values | 37 (Review Rating — imputed with category median) |

### Original Columns

| Column | Type | Description |
|---|---|---|
| Customer ID | int | Unique identifier (1–3900) |
| Age | int | Customer age (18–70) |
| Gender | string | Male / Female |
| Item Purchased | string | 25 unique product names |
| Category | string | Clothing · Footwear · Outerwear · Accessories |
| Purchase Amount (USD) | int | Transaction value ($20–$100) |
| Location | string | 50 US states |
| Size | string | S / M / L / XL |
| Color | string | 25 color options |
| Season | string | Spring / Summer / Fall / Winter |
| Review Rating | float | 2.5–5.0 · 37 null values |
| Subscription Status | string | Yes / No |
| Shipping Type | string | 6 types |
| Discount Applied | string | Yes / No |
| Promo Code Used | string | Identical to Discount Applied — dropped |
| Previous Purchases | int | 1–50 |
| Payment Method | string | 6 methods |
| Frequency of Purchases | string | Weekly / Monthly / Annually / etc. |

---

## 3. What Was in the Original Project

The base project (before improvements) contained:

### Python Notebook
- Load CSV, `df.info()`, `df.describe()`
- Null imputation for Review Rating (category median)
- Column rename to snake_case
- Drop redundant `promo_code_used`
- `age_group` via `pd.qcut()`
- `purchase_frequency_days` mapping
- PostgreSQL / MySQL / MSSQL load code

### SQL (10 queries)
Revenue by gender, discount buyers above average, top 5 products by rating, shipping comparison, subscription spend, discount rate by product, loyalty segments, top 3 per category, repeat buyers vs subscription, revenue by age group.

### Other files
Power BI dashboard, business problem PDF, project report PDF, presentation PPTX.

### What was missing
No requirements.txt, no .gitignore, no EDA charts, no ML models, no CLV, no A/B testing, no Streamlit app, no CI/CD, no documentation, hardcoded DB password, flat folder structure.

---

## 4. Improvements & New Features

### 4.1 Project Structure
Reorganised from flat root into professional folders:
```
data/raw/  data/processed/  notebooks/  sql/
dashboard/  plots/  reports/  docs/  streamlit_app/  .github/workflows/
```

### 4.2 Foundation Files
- `requirements.txt` — pinned dependencies
- `.gitignore` — Python, Jupyter, secrets, OS files
- `LICENSE` — updated to Ankit Kumar 2026
- `DOCUMENTATION.md` + `documentation.html` — this file

### 4.3 Python Notebook Enhancements
- Duplicate row check added
- Missing % display added
- 3 new engineered features: `spend_tier`, `loyalty_tier`, CLV columns
- Complete EDA section (9 charts)
- RFM segmentation section
- Churn prediction model section
- Cleaned CSV export
- Removed hardcoded credentials

### 4.4 Streamlit Web App (`streamlit_app/`)
Full interactive dashboard — see [Section 12](#12-streamlit-dashboard)

### 4.5 GitHub Actions CI/CD (`.github/workflows/`)
3 automated workflows — see [Section 13](#13-github-actions-cicd)

---

## 5. Feature Engineering Reference

| Feature | Type | Logic |
|---|---|---|
| `age_group` | categorical | `pd.qcut(age, q=4)` → 4 age bands |
| `purchase_frequency_days` | int | Map text → days (Weekly=7 … Annually=365) |
| `spend_tier` | categorical | `pd.cut` → Low / Medium / High |
| `loyalty_tier` | categorical | 1=New · 2–10=Returning · 11+=Loyal |
| `rfm_r` | int 1–5 | Recency score (lower days = higher score) |
| `rfm_f` | int 1–5 | Frequency score (higher purchases = higher) |
| `rfm_m` | int 1–5 | Monetary score (higher spend = higher) |
| `rfm_score` | int 3–15 | R + F + M total |
| `rfm_segment` | categorical | Mapped from rfm_score (5 segments) |
| `is_churn` | int 0/1 | Rule-based churn label |
| `churn_probability` | float 0–1 | Random Forest predicted probability |
| `purchase_freq_per_year` | float | 365 / purchase_frequency_days |
| `lifespan_years` | float | Estimated customer lifespan |
| `clv_estimated` | float | Spend × Freq/yr × Lifespan |
| `clv_tier` | categorical | High / Mid / Low Value (quartile-based) |

---

## 6. EDA Visualizations

All 15 charts saved to `plots/`.

| File | Chart | Insight |
|---|---|---|
| `01_purchase_distribution_gender.png` | Histogram + Bar | Purchase distribution + gender revenue |
| `02_revenue_category_season.png` | Bar charts | Category and seasonal revenue |
| `03_age_group_subscription.png` | Bar + Pie | Age group revenue + subscription split |
| `04_shipping_payment_discount.png` | 3-panel | Shipping spend · payment usage · discount impact |
| `05_products_ratings.png` | Bar charts | Top 10 products + category ratings |
| `06_correlation_heatmap.png` | Heatmap | Numeric feature correlations |
| `07_loyalty_subscription_heatmap.png` | Heatmap | Avg spend: loyalty × subscription |
| `08_rfm_segments.png` | Bar charts | RFM segment counts + avg spend |
| `09_churn_model.png` | Confusion + Bar | Churn model accuracy + feature importance |
| `10_us_revenue_map.png` / `.html` | Choropleth | Revenue by US state (interactive HTML) |
| `11_clv_distribution_tiers.png` | Histogram + Pie | CLV distribution + tier breakdown |
| `12_clv_by_category_subscription.png` | Bar charts | CLV by category + subscription |
| `13_clv_model_results.png` | Scatter + Bar | Actual vs predicted CLV + feature importance |
| `14_ab_test_summary.png` | 6-panel bar | All A/B test mean comparisons |
| `15_ab_test_forest_plot.png` | Forest plot | 95% CI on mean differences |

---

## 7. RFM Customer Segmentation

RFM scores every customer 1–15 and assigns a business segment.

### Proxy Measures (no dates in dataset)

| Dimension | Proxy | Direction |
|---|---|---|
| Recency (R) | `purchase_frequency_days` | Lower days → higher score |
| Frequency (F) | `previous_purchases` | Higher count → higher score |
| Monetary (M) | `purchase_amount` | Higher spend → higher score |

### Segments

| Segment | Score | Description |
|---|---|---|
| Champions | 13–15 | Best customers — retain at all costs |
| Loyal Customers | 10–12 | Regular buyers — reward them |
| Potential Loyalists | 7–9 | Promising — nurture with offers |
| At Risk | 5–6 | Declining — reactivation campaigns |
| Lost | < 5 | Inactive — win-back or abandon |

---

## 8. Churn Prediction Model

### Label Definition
Customer is churned (`is_churn=1`) if **all three** conditions hold:
- `purchase_frequency_days >= 90`
- `review_rating < 3.5`
- `subscription_status == 'No'`

### Model

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Trees | 200, Max depth 10 |
| Split | 80/20 stratified |
| Output | `churn_probability` per customer |

---

## 9. Customer Lifetime Value (CLV)

### Formula
```
CLV = purchase_amount × (365 / purchase_frequency_days) × lifespan_years

lifespan_years = 1 + (previous_purchases / purchases_per_year) / 12
```

### ML Model

| Property | Value |
|---|---|
| Algorithm | Gradient Boosting Regressor |
| Trees | 300, Max depth 5, LR=0.05 |
| R² Score | 0.9999 |
| MAE | ~$9.50 |

### CLV Tiers
- **High Value** — top 25% by CLV
- **Mid Value** — middle 50%
- **Low Value** — bottom 25%

---

## 10. A/B Testing

Statistical significance testing using:
- **Welch's two-sample t-test** (unequal variance)
- **Cohen's d** effect size
- **Bootstrap 95% Confidence Interval** (2,000 resamples)

### Pre-defined Tests

| Test | Conclusion |
|---|---|
| Discount Applied vs Not | ❌ Not significant (p=0.27, d=0.04) |
| Subscribed vs Not Subscribed | ❌ Not significant (p=0.66, d=0.02) |
| Express vs Standard Shipping | ❌ Not significant (p=0.13, d=0.08) |
| Male vs Female Spend | ❌ Not significant (p=0.38, d=0.03) |
| Clothing vs Footwear Revenue | ❌ Not significant (p=0.84, d=0.01) |
| Loyal vs New Customer Spend | ❌ Not significant (p=0.69, d=0.05) |

**Key Finding:** None of the common business assumptions produce statistically significant differences in spend. The business cannot assume discounts or subscriptions drive higher revenue — deeper segmentation is needed.

### Custom Test Builder
The Streamlit app allows running any custom A/B test on any two groups from any column.

---

## 11. SQL Queries Reference

File: `sql/customer_behavior_sql_queries.sql`

### Section 1 — Revenue Analysis (Q1–Q4)
| # | Query | Key SQL |
|---|---|---|
| Q1 | Revenue by gender + avg spend | GROUP BY, SUM, AVG |
| Q2 | Discount buyers above average + margin | Subquery, calculated column |
| Q3 | Revenue by age group + % contribution | SUM() OVER() |
| Q4 | Standard vs Express shipping | WHERE IN |

### Section 2 — Product Analysis (Q5–Q7)
| # | Query | Key SQL |
|---|---|---|
| Q5 | Top 5 products by review rating | ORDER BY DESC |
| Q6 | Top 5 products by discount rate % | CASE WHEN ratio |
| Q7 | Top 3 products per category | ROW_NUMBER() OVER(PARTITION BY) |

### Section 3 — Segmentation & Loyalty (Q8–Q10)
| # | Query | Key SQL |
|---|---|---|
| Q8 | Loyalty tier counts + avg spend | CTE, CASE WHEN, JOIN |
| Q9 | Subscribed vs not — full comparison | GROUP BY |
| Q10 | Repeat buyers vs subscription % | Window function % |

### Section 4 — Advanced Analytics (Q11–Q15)
| # | Query | Key SQL |
|---|---|---|
| Q11 | Running total + percentile rank | SUM() OVER, RANK() OVER |
| Q12 | Top 5 US states by revenue | ORDER BY, LIMIT |
| Q13 | Payment method avg spend | Aggregation |
| Q14 | Season × Category revenue pivot | CASE WHEN pivot |
| Q15 | Full RFM scoring in SQL | NTILE(), CTEs, CASE WHEN |

---

## 12. Streamlit Dashboard

File: `streamlit_app/app.py`  
Run: `streamlit run streamlit_app/app.py`

### 8 Tabs

| Tab | Content |
|---|---|
| 📊 Revenue Overview | Category/season/gender revenue · top products · distribution |
| 👥 Customer Segments | Age groups · loyalty tiers · shipping · payment · subscription |
| 🎯 RFM Analysis | Segment counts · avg spend · score distribution · segment×subscription |
| ⚠️ Churn Risk | Churn rate KPIs · by category/age · top 20 at-risk table |
| 🗺️ Geographic Map | US choropleth · 4 switchable metrics · top/bottom state tables |
| 💰 CLV Prediction | CLV KPIs · distribution · tiers · ML model results · top customers |
| 🧪 A/B Testing | 6 pre-defined tests · custom test builder · verdict banners · forest plots |
| 📋 Raw Data | Full filterable table · CSV download |

### Sidebar Filters (apply to all tabs)
Gender · Category · Season · Subscription Status · Age Range slider

### Module Files
| File | Purpose |
|---|---|
| `app.py` | Main Streamlit app |
| `clv_model.py` | CLV formula + Gradient Boosting model |
| `ab_test.py` | Welch t-test, Cohen's d, Bootstrap CI |
| `generate_geo_map.py` | Standalone US map generator |
| `generate_clv_plots.py` | Standalone CLV chart generator |
| `generate_ab_plots.py` | Standalone A/B test chart generator |

---

## 13. GitHub Actions CI/CD

Directory: `.github/workflows/`

### Workflow 1 — Notebook CI (`notebook_ci.yml`)
- **Trigger:** Push to `notebooks/` or `data/`
- **Steps:** Checkout → Python 3.13 → Install deps → Execute notebook → Upload artifact
- **Purpose:** Ensures notebook runs clean on every push

### Workflow 2 — Code Quality (`code_quality.yml`)
- **Trigger:** Push to `streamlit_app/`
- **Jobs:** `lint` (flake8), `security` (bandit)
- **Purpose:** Maintains code standards and scans for security issues

### Workflow 3 — Streamlit Test (`streamlit_test.yml`)
- **Trigger:** Push to `streamlit_app/` or `data/`
- **Steps:** Validates CLV model (R²>0.5), A/B test module (6 tests), app.py syntax
- **Purpose:** Ensures the web app always works after any change

---

## 14. File Inventory

```
customer-trends-data-analysis-SQL-Python-PowerBI/
│
├── 📁 .github/workflows/
│   ├── notebook_ci.yml              ← CI: execute notebook
│   ├── code_quality.yml             ← CI: lint + security
│   └── streamlit_test.yml           ← CI: app smoke test
│
├── 📁 data/
│   ├── raw/customer_shopping_behavior.csv
│   └── processed/customer_shopping_behavior_cleaned.csv
│
├── 📁 notebooks/
│   └── Customer_Shopping_Behavior_Analysis.ipynb
│
├── 📁 sql/
│   └── customer_behavior_sql_queries.sql
│
├── 📁 dashboard/
│   └── customer_behavior_dashboard.pbix
│
├── 📁 plots/  (15 files)
│   ├── 01–09: EDA charts (PNG)
│   ├── 10: US revenue map (PNG + HTML)
│   ├── 11–13: CLV charts (PNG)
│   └── 14–15: A/B test charts (PNG)
│
├── 📁 streamlit_app/
│   ├── app.py
│   ├── clv_model.py
│   ├── ab_test.py
│   ├── generate_geo_map.py
│   ├── generate_clv_plots.py
│   ├── generate_ab_plots.py
│   └── requirements.txt
│
├── 📁 reports/
│   ├── Business Problem Document.pdf
│   ├── Customer Shopping Behavior Analysis.pdf
│   └── Customer-Shopping-Behavior-Analysis.pptx
│
├── 📁 docs/
│   ├── DOCUMENTATION.md             ← this file
│   └── documentation.html           ← styled HTML version
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 15. How to Run Locally

### Prerequisites
- Python 3.13 (conda recommended)
- PostgreSQL / MySQL / MS SQL Server (for SQL section)
- Power BI Desktop (Windows)

### Setup
```bash
git clone https://github.com/hrts51/customer-trends-data-analysis-SQL-Python-PowerBI.git
cd customer-trends-data-analysis-SQL-Python-PowerBI

conda create -n customer-analysis python=3.13
conda activate customer-analysis
pip install -r requirements.txt
```

### Run Jupyter Notebook
```bash
jupyter notebook notebooks/Customer_Shopping_Behavior_Analysis.ipynb
```

### Run Streamlit Dashboard
```bash
cd streamlit_app
streamlit run app.py
# Opens at http://localhost:8501
```

### Generate All Plots
```bash
cd streamlit_app
python generate_geo_map.py      # US map
python generate_clv_plots.py    # CLV charts
python generate_ab_plots.py     # A/B test charts
```

### SQL Setup
```sql
CREATE DATABASE customer_behavior;
-- Then uncomment the relevant DB section in the notebook and run
-- Then open sql/customer_behavior_sql_queries.sql in your SQL client
```

---

## 16. Key Business Insights

1. **Gender Revenue Gap is Smaller Than Expected**  
   Male customers are ~68% of transactions but avg spend difference vs female is statistically insignificant (p=0.38).

2. **Clothing + Spring = Peak Revenue**  
   Clothing is the top category; Spring generates highest seasonal revenue.

3. **Subscriptions Signal Loyalty, Not Spend**  
   Subscribed customers don't spend significantly more per transaction (p=0.66) but have more previous purchases — subscription predicts retention, not ticket size.

4. **Champions Drive Disproportionate Revenue**  
   Top RFM segment (~20% of customers) generates a disproportionately large revenue share.

5. **Discounts Are Ineffective**  
   Customers using discounts don't spend more than non-discount buyers (p=0.27, d=0.04 — negligible effect). The discount program may be eroding margin without driving incremental revenue.

6. **CLV is Predictable**  
   Gradient Boosting model predicts customer lifetime value with R²=0.9999 — previous purchases and purchase frequency are the dominant drivers.

7. **Churn Profile is Actionable**  
   Churn-risk customers share 3 clear traits: infrequent (≥90 days), low rating (<3.5), not subscribed. These can be directly targeted.

8. **Geographic Revenue is Dispersed**  
   Revenue is spread fairly evenly across states — no single geographic market dominates, suggesting national rather than regional strategy is appropriate.

---

## 17. Git Commit History

| Hash | Message |
|---|---|
| `9af7e08` | Add GitHub Actions CI — notebook, code quality, streamlit test + badges |
| `f5b03d0` | Add A/B testing — Welch t-test, Cohen's d, bootstrap CI, forest plot |
| `96ad42d` | Add CLV prediction — Gradient Boosting, tier segmentation, Streamlit tab |
| `7d34a54` | Add geographic US revenue map — choropleth, interactive HTML + PNG |
| `7eccb0f` | Add Streamlit interactive dashboard (5 tabs) |
| `1f72772` | Fix LICENSE — Ankit Kumar 2026 |
| `31f7604` | Initial commit — Customer Shopping Behavior Analysis by Ankit Kumar |

---

*Documentation by Ankit Kumar — [github.com/hrts51](https://github.com/hrts51)*
