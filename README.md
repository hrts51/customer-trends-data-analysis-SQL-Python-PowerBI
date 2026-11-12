# 🛍️ Customer Shopping Behavior Analysis

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![SQL](https://img.shields.io/badge/SQL-PostgreSQL%20%7C%20MySQL%20%7C%20MSSQL-blue?logo=postgresql)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow?logo=powerbi)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
[![📓 Notebook CI](https://github.com/hrts51/customer-trends-data-analysis-SQL-Python-PowerBI/actions/workflows/notebook_ci.yml/badge.svg)](https://github.com/hrts51/customer-trends-data-analysis-SQL-Python-PowerBI/actions/workflows/notebook_ci.yml)
[![🔍 Code Quality](https://github.com/hrts51/customer-trends-data-analysis-SQL-Python-PowerBI/actions/workflows/code_quality.yml/badge.svg)](https://github.com/hrts51/customer-trends-data-analysis-SQL-Python-PowerBI/actions/workflows/code_quality.yml)
[![🚀 Streamlit Test](https://github.com/hrts51/customer-trends-data-analysis-SQL-Python-PowerBI/actions/workflows/streamlit_test.yml/badge.svg)](https://github.com/hrts51/customer-trends-data-analysis-SQL-Python-PowerBI/actions/workflows/streamlit_test.yml)

An end-to-end data analytics portfolio project that mirrors real corporate analyst workflows — from raw data to actionable business insights. Covers data cleaning, exploratory data analysis, RFM customer segmentation, churn prediction, SQL analytics, and Power BI visualization.

---

## 📌 Project Overview

| Stage | Tool | Output |
|---|---|---|
| Data Cleaning & EDA | Python (pandas, seaborn, matplotlib) | Cleaned dataset + 9 visualizations |
| Feature Engineering | Python | Age group, spend tier, loyalty tier, RFM scores |
| Customer Segmentation | scikit-learn (RFM) | 5 customer segments |
| Churn Prediction | scikit-learn (Random Forest) | Churn probability per customer |
| SQL Analytics | PostgreSQL / MySQL / MS SQL Server | 15 business queries across 4 sections |
| Dashboard | Power BI | Interactive business dashboard |

---

## 🗂️ Project Structure

```
customer-trends-data-analysis-SQL-Python-PowerBI/
│
├── 📁 .github/workflows/
│   ├── notebook_ci.yml          ← Auto-runs notebook on push
│   ├── code_quality.yml         ← Linting + security scan
│   └── streamlit_test.yml       ← Verifies app loads correctly
│
├── 📁 data/
│   ├── raw/
│   │   └── customer_shopping_behavior.csv          ← Original dataset (3,900 rows)
│   └── processed/
│       └── customer_shopping_behavior_cleaned.csv  ← Cleaned + enriched (24 cols)
│
├── 📁 notebooks/
│   └── Customer_Shopping_Behavior_Analysis.ipynb   ← Main Python notebook
│
├── 📁 sql/
│   └── customer_behavior_sql_queries.sql           ← 15 business queries · 4 sections
│
├── 📁 dashboard/
│   └── customer_behavior_dashboard.pbix            ← Power BI dashboard
│
├── 📁 plots/                                       ← EDA & model charts (9 PNGs)
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
├── 📁 reports/
│   ├── Business Problem Document.pdf
│   ├── Customer Shopping Behavior Analysis.pdf
│   └── Customer-Shopping-Behavior-Analysis.pptx
│
├── 📁 docs/
│   ├── DOCUMENTATION.md                            ← Full project documentation
│   └── documentation.html                          ← Styled HTML version
│
├── 📋 requirements.txt
├── 🔒 .gitignore
├── 📜 LICENSE
└── 📖 README.md
```

---

## 📊 Dataset

- **3,900** customer transactions
- **18 original features:** Age, Gender, Item Purchased, Category, Purchase Amount, Location, Size, Color, Season, Review Rating, Subscription Status, Shipping Type, Discount Applied, Previous Purchases, Payment Method, Frequency of Purchases
- **New engineered features:** Age Group, Spend Tier, Loyalty Tier, Purchase Frequency Days, RFM Scores, RFM Segment, Churn Probability

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/hrts51/customer-trends-data-analysis-SQL-Python-PowerBI.git
cd customer-trends-data-analysis-SQL-Python-PowerBI
```

### 2. Set up Python environment

**Recommended: Conda (Python 3.13)**
```bash
conda create -n customer-analysis python=3.13
conda activate customer-analysis
pip install -r requirements.txt
```

**Or with pip (Python 3.9–3.13):**
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/Customer_Shopping_Behavior_Analysis.ipynb
```

The notebook will:
- Load and clean the raw dataset
- Engineer new features
- Generate 9 EDA charts saved to `plots/`
- Perform RFM customer segmentation
- Train a churn prediction model (Random Forest)
- Export the cleaned dataset as `customer_shopping_behavior_cleaned.csv`

### 4. Set up SQL Database

Create a database named `customer_behavior` in PostgreSQL, MySQL, or SQL Server, then uncomment and run the appropriate SQL load cell at the bottom of the notebook.

Then open `sql/customer_behavior_sql_queries.sql` in your SQL client (pgAdmin, DBeaver, MySQL Workbench) and run the 15 queries.

### 5. Open the Power BI Dashboard

Open `dashboard/customer_behavior_dashboard.pbix` in Power BI Desktop and connect it to your SQL database or `data/processed/customer_shopping_behavior_cleaned.csv`.

---

## 🔍 Key Insights

- **Male customers** account for ~68% of transactions but have only marginally higher spend per transaction
- **Clothing** is the top revenue category; **Spring** is the peak revenue season
- **Subscribed customers** spend more on average and show significantly higher loyalty
- **Champions** (top RFM segment) represent ~20% of customers but drive disproportionate revenue
- Churn risk is highest among infrequent, low-rated, non-subscribed customers

---

## 🛠️ Tech Stack

- **Python 3.13** — pandas, numpy, matplotlib, seaborn, scikit-learn
- **SQL** — PostgreSQL / MySQL / MS SQL Server
- **Power BI Desktop**
- **Jupyter Notebook**

---

## 👨‍💻 About

**Ankit Kumar**
📂 GitHub: [github.com/hrts51](https://github.com/hrts51)

---

## 📜 License

MIT — free to fork, star ⭐, and use in your own portfolio.
