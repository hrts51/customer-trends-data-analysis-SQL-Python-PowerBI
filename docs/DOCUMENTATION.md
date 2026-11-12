# 📄 Customer Shopping Behavior Analysis — Full Documentation

**Author:** Ankit Kumar  
**GitHub:** [github.com/hrts51](https://github.com/hrts51)  
**Repository:** [customer-trends-data-analysis-SQL-Python-PowerBI](https://github.com/hrts51/customer-trends-data-analysis-SQL-Python-PowerBI)  
**Stack:** Python 3.13 · SQL (PostgreSQL / MySQL / MS SQL Server) · Power BI  

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Dataset Overview](#2-dataset-overview)
3. [What Was Already Done (Original)](#3-what-was-already-done-original)
4. [What Was Improved (This Version)](#4-what-was-improved-this-version)
5. [Feature Engineering Reference](#5-feature-engineering-reference)
6. [EDA Visualizations](#6-eda-visualizations)
7. [RFM Customer Segmentation](#7-rfm-customer-segmentation)
8. [Churn Prediction Model](#8-churn-prediction-model)
9. [SQL Queries Reference](#9-sql-queries-reference)
10. [File Inventory](#10-file-inventory)
11. [How to Run Locally](#11-how-to-run-locally)
12. [Key Business Insights](#12-key-business-insights)
13. [Git Commit History](#13-git-commit-history)

---

## 1. Project Purpose

This is an end-to-end data analytics portfolio project that simulates a real corporate analyst workflow. Starting from a raw retail dataset of 3,900 customer transactions, the project demonstrates every stage of the analytics pipeline:

- **Data preparation** — cleaning, imputation, column standardisation
- **Feature engineering** — creating new analytical dimensions
- **Exploratory analysis** — visual storytelling with charts
- **Customer segmentation** — RFM scoring and behavioural grouping
- **Predictive modelling** — churn probability with Random Forest
- **SQL analytics** — 15 business queries across 4 categories
- **Dashboard** — interactive Power BI report

---

## 2. Dataset Overview

| Property | Value |
|---|---|
| File | `customer_shopping_behavior.csv` |
| Rows | 3,900 |
| Original columns | 18 |
| Final columns (after engineering) | 24 |
| Missing values | 37 (only in `Review Rating`) |

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
| Shipping Type | string | 6 types (Free Shipping, Express, Standard, etc.) |
| Discount Applied | string | Yes / No |
| Promo Code Used | string | Yes / No — identical to Discount Applied (dropped) |
| Previous Purchases | int | Count of past purchases (1–50) |
| Payment Method | string | 6 methods (PayPal, Credit Card, Venmo, etc.) |
| Frequency of Purchases | string | Weekly / Fortnightly / Monthly / Quarterly / Annually / Every 3 Months |

---

## 3. What Was Already Done (Original)

### Python Notebook
- Loaded CSV using `pandas`
- Ran `df.info()` and `df.describe(include='all')` for initial exploration
- Checked null values — found 37 missing in `Review Rating`
- Imputed `Review Rating` nulls using **per-category median**
- Renamed all columns to `snake_case`
- Verified `promo_code_used == discount_applied` for all rows → dropped `promo_code_used`
- Created `age_group` column using `pd.qcut()` (4 equal-frequency bins)
- Created `purchase_frequency_days` by mapping text labels to numeric days
- Connected to **PostgreSQL** via `sqlalchemy + psycopg2` and loaded `df` into a `customer` table
- Included commented code for **MySQL** (`pymysql`) and **MS SQL Server** (`pyodbc`) connections

### SQL Queries (10 original)

| # | Question |
|---|---|
| Q1 | Total revenue by gender |
| Q2 | Customers who used discount but spent above average |
| Q3 | Top 5 products by average review rating |
| Q4 | Avg purchase amount: Standard vs Express shipping |
| Q5 | Subscribed vs non-subscribed — avg spend and total revenue |
| Q6 | Top 5 products by discount usage rate % |
| Q7 | Customer count per loyalty segment (New / Returning / Loyal) |
| Q8 | Top 3 most purchased products per category (ROW_NUMBER window function) |
| Q9 | Repeat buyers (>5 purchases) vs subscription status |
| Q10 | Revenue contribution by age group |

### Other Files
- `customer_behavior_dashboard.pbix` — Power BI dashboard
- `Business Problem Document.pdf` — business problem statement
- `Customer Shopping Behavior Analysis.pdf` — project report
- `Customer-Shopping-Behavior-Analysis.pptx` — presentation deck

---

## 4. What Was Improved (This Version)

### New Files
| File | Description |
|---|---|
| `requirements.txt` | Pinned dependencies for reproducible Python environment |
| `.gitignore` | Covers Python cache, venv, secrets, OS files, Jupyter checkpoints |
| `customer_shopping_behavior_cleaned.csv` | Cleaned + enriched dataset (24 columns) for Power BI or further analysis |
| `plots/` (9 PNG files) | All EDA and model visualisations saved as PNG images |
| `DOCUMENTATION.md` | This file |
| `documentation.html` | Styled HTML version of this documentation |

### Notebook Improvements
- Added duplicate row check
- Added missing value % display
- Added 3 new engineered features: `spend_tier`, `loyalty_tier`, plus full RFM + churn columns
- Added complete EDA section with 9 charts
- Added RFM segmentation section
- Added churn prediction model section
- Added cleaned CSV export
- Removed hardcoded database password — replaced with clearly labelled placeholder variables
- Updated author line to Ankit Kumar

### SQL Improvements
- Added 5 new queries (Q11–Q15)
- Organised into 4 labelled sections
- Added consistent formatting, column aliases, and comments throughout
- Added revenue % contribution using `SUM() OVER()` window function
- Added `above_avg_by` calculated column in Q2
- Added `avg_spend` join in loyalty segment query

### README Improvements
- Removed all original author references
- Added your name and GitHub link
- Added project structure tree
- Added dataset summary table
- Added 5-step Quick Start guide
- Added 6 key business insights
- Added tech stack section
- Added Python / SQL / Power BI / License badges

---

## 5. Feature Engineering Reference

| Feature | Type | Logic |
|---|---|---|
| `age_group` | categorical | `pd.qcut(age, q=4)` → Young Adult / Adult / Middle-aged / Senior |
| `purchase_frequency_days` | int | Map text → days: Weekly=7, Fortnightly=14, Monthly=30, Quarterly=90, Annually=365 |
| `spend_tier` | categorical | `pd.cut(purchase_amount, bins=[0,40,70,100])` → Low / Medium / High |
| `loyalty_tier` | categorical | `previous_purchases`: 1=New, 2–10=Returning, 11+=Loyal |
| `rfm_r` | int (1–5) | Recency score — based on `purchase_frequency_days` (lower days = higher score) |
| `rfm_f` | int (1–5) | Frequency score — based on `previous_purchases` (rank-based qcut) |
| `rfm_m` | int (1–5) | Monetary score — based on `purchase_amount` (qcut) |
| `rfm_score` | int (3–15) | Sum of rfm_r + rfm_f + rfm_m |
| `rfm_segment` | categorical | Mapped from rfm_score (see Section 7) |
| `is_churn` | int (0/1) | Rule: frequency_days≥90 AND review_rating<3.5 AND subscription=No |
| `churn_probability` | float (0–1) | Random Forest predicted probability of churn |

---

## 6. EDA Visualizations

All charts saved to the `plots/` folder.

| File | Chart Type | Key Question Answered |
|---|---|---|
| `01_purchase_distribution_gender.png` | Histogram + Bar | How is purchase amount distributed? Which gender generates more revenue? |
| `02_revenue_category_season.png` | Horizontal bar + Bar | Which product category earns most? Which season peaks? |
| `03_age_group_subscription.png` | Bar + Pie | Which age group spends most? What is the subscription split? |
| `04_shipping_payment_discount.png` | 3-panel bar | Does shipping type affect spend? Which payment method is most used? Do discounts lift spend? |
| `05_products_ratings.png` | Horizontal bar + Bar | What are the top 10 revenue products? How do categories compare on ratings? |
| `06_correlation_heatmap.png` | Heatmap | Are any numeric features correlated with purchase amount? |
| `07_loyalty_subscription_heatmap.png` | Heatmap | How does avg spend vary across loyalty tier × subscription combinations? |
| `08_rfm_segments.png` | Bar + Horizontal bar | How many customers per segment? What is the avg spend per segment? |
| `09_churn_model.png` | Confusion matrix + Bar | How accurate is the churn model? Which features drive churn? |

---

## 7. RFM Customer Segmentation

RFM (Recency, Frequency, Monetary) is a proven marketing technique to rank and segment customers by behaviour.

### Proxy Measures Used

Since the dataset has no actual transaction dates, the following proxies were applied:

| RFM Dimension | Proxy | Direction |
|---|---|---|
| Recency (R) | `purchase_frequency_days` | Lower days → Higher score (shops more recently) |
| Frequency (F) | `previous_purchases` | Higher count → Higher score |
| Monetary (M) | `purchase_amount` | Higher spend → Higher score |

### Scoring Method

Each dimension scored 1–5 using `pd.qcut()` with `duplicates='drop'` to handle discrete frequency values. Total RFM score = R + F + M (range: 3–15).

### Segments

| Segment | Score Range | Description |
|---|---|---|
| Champions | 13–15 | Best customers — frequent, high spend, recently active |
| Loyal Customers | 10–12 | Regular buyers with solid spend |
| Potential Loyalists | 7–9 | Promising customers who need engagement |
| At Risk | 5–6 | Previously good, now showing decline |
| Lost | < 5 | Inactive or low-value — difficult to recover |

---

## 8. Churn Prediction Model

### Churn Label Definition (Rule-Based)

A customer is labelled as churned (`is_churn = 1`) if **all three** conditions are true:
- `purchase_frequency_days >= 90` (infrequent shopper)
- `review_rating < 3.5` (low satisfaction)
- `subscription_status == 'No'` (not subscribed)

### Model

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Trees | 200 |
| Max depth | 10 |
| Train/test split | 80% / 20% (stratified) |
| Random state | 42 |

### Features Used

`age`, `purchase_amount`, `review_rating`, `previous_purchases`, `purchase_frequency_days`, `gender`, `category`, `shipping_type`, `discount_applied`, `payment_method`

Categorical features were label-encoded before training.

### Output

- Confusion matrix showing true/false positives and negatives
- Feature importance chart showing which variables drive churn most
- `churn_probability` column added to the final dataframe for every customer

---

## 9. SQL Queries Reference

Queries are organised into 4 sections in `customer_behavior_sql_queries.sql`.

### Section 1 — Revenue Analysis

| # | Query | Key Technique |
|---|---|---|
| Q1 | Revenue by gender + avg spend | GROUP BY, SUM, AVG |
| Q2 | Discount buyers above average + margin | Subquery, calculated column |
| Q3 | Revenue by age group + % contribution | SUM() OVER() window function |
| Q4 | Standard vs Express shipping comparison | WHERE IN, GROUP BY |

### Section 2 — Product Analysis

| # | Query | Key Technique |
|---|---|---|
| Q5 | Top 5 products by avg review rating | GROUP BY, AVG, ORDER BY |
| Q6 | Top 5 products by discount rate % | CASE WHEN, ratio calculation |
| Q7 | Top 3 products per category | ROW_NUMBER() OVER(PARTITION BY) |

### Section 3 — Customer Segmentation & Loyalty

| # | Query | Key Technique |
|---|---|---|
| Q8 | Loyalty tier counts + avg spend | CTE, CASE WHEN, JOIN |
| Q9 | Subscribed vs not — spend comparison | GROUP BY, SUM, AVG |
| Q10 | Repeat buyers vs subscription % | WHERE, window function for % |

### Section 4 — Advanced Analytics (New)

| # | Query | Key Technique |
|---|---|---|
| Q11 | Running revenue total + percentile rank | SUM() OVER, RANK() OVER |
| Q12 | Top 5 US states by revenue | GROUP BY, ORDER BY, LIMIT |
| Q13 | Payment method avg spend + revenue | Aggregation |
| Q14 | Season × Category revenue pivot | CASE WHEN conditional aggregation |
| Q15 | Full RFM scoring in SQL | NTILE(), CASE WHEN, multiple CTEs |

---

## 10. File Inventory

```
customer-trends-data-analysis-SQL-Python-PowerBI/
│
├── 📓 Customer_Shopping_Behavior_Analysis.ipynb   ← MODIFIED — full notebook
├── 🗄️  customer_behavior_sql_queries.sql           ← MODIFIED — 15 queries, 4 sections
├── 📊 customer_behavior_dashboard.pbix            ← original Power BI dashboard
│
├── 📁 plots/                                      ← NEW
│   ├── 01_purchase_distribution_gender.png
│   ├── 02_revenue_category_season.png
│   ├── 03_age_group_subscription.png
│   ├── 04_shipping_payment_discount.png
│   ├── 05_products_ratings.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_loyalty_subscription_heatmap.png
│   ├── 08_rfm_segments.png
│   └── 09_churn_model.png
│
├── 📄 customer_shopping_behavior.csv              ← original raw dataset
├── 📄 customer_shopping_behavior_cleaned.csv      ← NEW — cleaned + 24 cols
├── 📋 requirements.txt                            ← NEW
├── 🔒 .gitignore                                  ← NEW
├── 📄 DOCUMENTATION.md                            ← NEW — this file
├── 🌐 documentation.html                          ← NEW — styled HTML version
│
├── 📄 Business Problem Document.pdf               ← original
├── 📄 Customer Shopping Behavior Analysis.pdf     ← original
├── 📄 Customer-Shopping-Behavior-Analysis.pptx   ← original
└── 📜 LICENSE                                     ← original
```

---

## 11. How to Run Locally

### Prerequisites
- Python 3.13 (conda recommended)
- One of: PostgreSQL / MySQL / MS SQL Server
- Power BI Desktop (Windows)

### Setup

```bash
# 1. Clone
git clone https://github.com/hrts51/customer-trends-data-analysis-SQL-Python-PowerBI.git
cd customer-trends-data-analysis-SQL-Python-PowerBI

# 2. Create environment
conda create -n customer-analysis python=3.13
conda activate customer-analysis

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch notebook
jupyter notebook Customer_Shopping_Behavior_Analysis.ipynb
```

### Notebook Execution Order

1. Run all cells in **Section 1** (Data Import & Exploration)
2. Run all cells in **Section 2** (Data Cleaning)
3. Run all cells in **Section 3** (Feature Engineering)
4. Run all cells in **Section 4** (EDA) — charts saved to `plots/`
5. Run all cells in **Section 5** (RFM Segmentation)
6. Run all cells in **Section 6** (Churn Prediction)
7. In **Section 7** (SQL Load), uncomment the block matching your database and fill in your credentials

### SQL Setup

```sql
-- Create the database first (PostgreSQL example)
CREATE DATABASE customer_behavior;
```

Then run the notebook's SQL load cell, followed by opening `customer_behavior_sql_queries.sql` in your client.

### Power BI Setup

1. Open `customer_behavior_dashboard.pbix` in Power BI Desktop
2. Connect to your SQL database **or** import `customer_shopping_behavior_cleaned.csv` directly

---

## 12. Key Business Insights

1. **Gender Revenue Gap is Smaller Than Expected**  
   Male customers make up ~68% of transactions, but their average spend per transaction is only marginally higher than female customers — gender alone is not a strong revenue predictor.

2. **Clothing Dominates, Spring Peaks**  
   Clothing is the top revenue category across all seasons. Spring consistently drives the highest seasonal revenue.

3. **Subscriptions Drive Loyalty**  
   Subscribed customers spend more per transaction on average and have significantly more previous purchases — subscription is a strong loyalty signal.

4. **Champions Are the Revenue Core**  
   The top RFM segment (Champions) represents roughly 20% of customers but drives a disproportionately large share of total revenue. Retaining this segment should be the top priority.

5. **Churn Profile is Clear and Actionable**  
   At-risk customers share a clear profile: infrequent purchase cycle (≥90 days), low review rating (<3.5), not subscribed. These customers can be targeted with re-engagement campaigns.

6. **Discounts Don't Always Lift Spend**  
   Customers who used discounts do not consistently spend more than non-discount buyers on average. The discount strategy may be reducing revenue without driving incremental purchases.

---

## 13. Git Commit History

| Commit Hash | Message |
|---|---|
| `f1662ec` | Update author details to Ankit Kumar (hrts51) |
| `0a25dc7` | Improve project: EDA visualizations, RFM segmentation, churn prediction, enhanced SQL queries, requirements.txt, .gitignore, updated README |
| `c84947a` | Update README.md |
| `ab5bf4f` | Update README.md |
| `da66720` | Update README.md |

---

*Documentation by Ankit Kumar — [github.com/hrts51](https://github.com/hrts51)*
