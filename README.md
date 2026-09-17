# Amazon Product Rating Prediction

> **End-to-end ML project** — Linear Regression model that predicts Amazon India product star-ratings from pricing and engagement signals, served via a Flask REST API and explored through an interactive Streamlit dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Dataset](#3-dataset)
4. [Data Cleaning Steps](#4-data-cleaning-steps)
5. [Model — Linear Regression](#5-model--linear-regression)
6. [Model Metrics](#6-model-metrics)
7. [Model Equation & Coefficients](#7-model-equation--coefficients)
8. [Setup](#8-setup)
9. [Run Order](#9-run-order)
10. [Flask API Reference](#10-flask-api-reference)
11. [Streamlit UI Pages](#11-streamlit-ui-pages)
12. [Screenshots](#12-screenshots)
13. [Report](#13-report)
14. [Dependencies](#14-dependencies)
15. [Author](#15-author)

---

## 1. Project Overview

This project answers a practical business question:

> *Can the pricing strategy and engagement metrics of an Amazon India product predict its average customer rating?*

The pipeline covers every stage of a real ML workflow:

| Stage | What happens |
|---|---|
| **Raw data** | 1,465-row `amazon.csv` with messy currency strings, duplicate products, a bad rating row, and missing values |
| **Cleaning** | 7 systematic steps reduce it to 1,350 clean, unique products |
| **Modelling** | scikit-learn `Pipeline` (OHE + `LinearRegression`) trained on an 80/20 split |
| **Serving** | Flask REST API exposes `/predict` and `/model-info` endpoints |
| **Dashboard** | Streamlit single-file UI with 4 pages: Overview, EDA, Model Insights, Predict |
| **Screenshots** | Programmatically generated with matplotlib, saved to `screenshots/` |
| **Report** | Full Word document (`Amazon_Rating_Prediction_Report.docx`) — 10 sections, embedded images |

---

## 2. Project Structure

```
IBM-project/
│
├── amazon.csv                          # Raw source dataset (Amazon India, 1 465 rows × 16 cols)
├── amazon_clean.csv                    # Cleaned dataset output by train_model.py (1 350 rows)
│
├── train_model.py                      # Step 1 – Data cleaning + model training + artefact export
├── app.py                              # Step 2 – Flask REST API (port 5000)
├── ui.py                               # Step 3 – Streamlit dashboard (single file, 4 pages)
│
├── generate_screenshots.py             # Step 4 (optional) – Matplotlib-based screenshot generator
├── build_report.py                     # Step 5 (optional) – Builds the .docx project report
│
├── model.pkl                           # Trained scikit-learn Pipeline (preprocessor + LinearRegression)
├── model_info.json                     # R², MAE, RMSE, intercept, coefficients, categories, split sizes
│
├── requirements.txt                    # All Python package dependencies
│
├── screenshots/
│   ├── page1_dataset_overview.png      # Dataset Overview page screenshot
│   ├── page2_eda.png                   # EDA page screenshot
│   ├── page3_model_insights.png        # Model Insights page screenshot
│   └── page4_predict.png              # Predict page screenshot
│
└── Amazon_Rating_Prediction_Report.docx  # Full project report (cover → conclusion, ~15 pages)
```

### File-by-file description

| File | Language | Purpose |
|---|---|---|
| `amazon.csv` | — | Raw Kaggle dataset ([Kaggle Dataset Link](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)); 16 columns including prices, discounts, ratings, reviews |
| `amazon_clean.csv` | — | Cleaned output ([Google Drive Link](https://drive.google.com/file/d/1Nt2hzfH1L8QCOmCABavkbClMH5vs5f19/view?usp=sharing)); 1 350 rows; used by Streamlit for all EDA charts |
| `train_model.py` | Python | Loads raw CSV → cleans data → trains LinearRegression pipeline → saves `model.pkl`, `model_info.json`, `amazon_clean.csv` |
| `app.py` | Python / Flask | REST API; loads `model.pkl` and `model_info.json` at startup; exposes `POST /predict`, `GET /model-info`, `GET /health` |
| `ui.py` | Python / Streamlit | Single-file frontend; 4 sidebar-navigated pages; calls Flask `/predict` for live predictions |
| `generate_screenshots.py` | Python / Matplotlib | Renders all 4 UI pages as high-DPI PNGs into `screenshots/` without a browser |
| `build_report.py` | Python / python-docx | Assembles the full Word report with styled headings, tables, embedded images |
| `model.pkl` | Binary (pickle) | Serialised `sklearn.pipeline.Pipeline` — ColumnTransformer → LinearRegression |
| `model_info.json` | JSON | R², MAE, RMSE, intercept, all coefficients, category list, train/test sizes |
| `requirements.txt` | Text | Pinned package list; install with `pip install -r requirements.txt` |
| `screenshots/*.png` | PNG | One 130 DPI image per Streamlit page |
| `Amazon_Rating_Prediction_Report.docx` | Word | 10-section project report — abstract through conclusion |

---

## 3. Dataset

- **Raw Dataset Source:** [Kaggle — Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset) *(This is the link where the raw dataset was obtained)*
- **Cleaned Dataset (CSV):** [Google Drive — amazon_clean.csv](https://drive.google.com/file/d/1Nt2hzfH1L8QCOmCABavkbClMH5vs5f19/view?usp=sharing) *(Drive link to download the cleaned CSV file)*

| Property | Value |
|---|---|
| **Source** | [Kaggle — Amazon Sales Dataset (India)](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset) |
| **Raw Dataset Link** | [Download from Kaggle](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset) |
| **Cleaned Dataset Link** | [Download from Google Drive](https://drive.google.com/file/d/1Nt2hzfH1L8QCOmCABavkbClMH5vs5f19/view?usp=sharing) |
| **File** | `amazon.csv` |
| **Clean File** | `amazon_clean.csv` |
| **Raw shape** | 1,465 rows × 16 columns |
| **Clean shape** | 1,350 rows × 18 columns (after dedup + engineered `main_category`, `sub_category`) |
| **Target column** | `rating` — average star rating (float, 1–5) |

### All 16 raw columns

| Column | Raw type | Notes |
|---|---|---|
| `product_id` | string | Unique product identifier; used for deduplication |
| `product_name` | string | Full product title |
| `category` | string | Pipe-separated hierarchy, e.g. `Computers&Accessories\|Accessories&Peripherals\|...` |
| `discounted_price` | string | Sale price; format `₹399` or `₹1,099` — needs symbol + comma stripping |
| `actual_price` | string | MRP / list price; same format issues |
| `discount_percentage` | string | e.g. `64%` — needs `%` strip and `/100` normalisation |
| `rating` | string | Average star rating; one row contains `\|` (malformed) |
| `rating_count` | string | Number of ratings; format `24,269` — two values missing |
| `about_product` | string | Pipe-separated product bullet points |
| `user_id` | string | Comma-separated list of reviewer user IDs |
| `user_name` | string | Comma-separated reviewer names |
| `review_id` | string | Comma-separated review IDs |
| `review_title` | string | Comma-separated review titles |
| `review_content` | string | Comma-separated full review texts |
| `img_link` | string | Product image URL |
| `product_link` | string | Amazon product page URL |

---

## 4. Data Cleaning Steps

All cleaning is performed inside [`train_model.py`](train_model.py) before any modelling takes place. A **before/after summary** is printed to the console on every run.

### Step-by-step

#### Step 1 — Clean `discounted_price`
```python
df["discounted_price"] = (
    df["discounted_price"]
    .astype(str)
    .str.replace("₹", "", regex=False)   # remove currency symbol
    .str.replace(",", "", regex=False)   # remove thousands separator
    .str.strip()
)
df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors="coerce")
```
Converts strings like `"₹1,099"` → `1099.0` (float64).  
Any unparseable values become `NaN` and are caught by the final NaN-drop step.

---

#### Step 2 — Clean `actual_price`
Identical transformation to `discounted_price`:
```python
df["actual_price"] = (
    df["actual_price"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["actual_price"] = pd.to_numeric(df["actual_price"], errors="coerce")
```

---

#### Step 3 — Clean `discount_percentage`
```python
df["discount_percentage"] = (
    df["discount_percentage"]
    .astype(str)
    .str.replace("%", "", regex=False)   # remove percent sign
    .str.strip()
)
df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce") / 100
```
`"64%"` → `0.64` (unit fraction). Stored as float64.

---

#### Step 4 — Fix `rating` and drop the bad row
```python
df["rating"] = pd.to_numeric(df["rating"].astype(str).str.strip(), errors="coerce")
df = df.dropna(subset=["rating"])
```
One row had `rating = "|"` — a data-entry error. `pd.to_numeric(errors="coerce")` converts it to `NaN`; `dropna` removes it.  
**Row count:** 1,465 → 1,464.

---

#### Step 5 — Clean `rating_count` and impute 2 missing values
```python
df["rating_count"] = (
    df["rating_count"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")
df["rating_count"].fillna(df["rating_count"].median(), inplace=True)
```
`"24,269"` → `24269.0`. Two `NaN` values are **imputed with the column median** (no rows dropped).

---

#### Step 6 — Engineer `main_category` and `sub_category`
```python
cat_split = df["category"].astype(str).str.split("|")
df["main_category"] = cat_split.str[0].str.strip()
df["sub_category"]  = cat_split.str[1].str.strip().fillna("Other")
```
The raw `category` field can have up to 5 pipe-separated levels (e.g. `Computers&Accessories|Accessories&Peripherals|Cables&Accessories|Cables|USBCables`).  
Only the **first two levels** are kept. Missing sub-categories are filled with `"Other"`.

---

#### Step 7 — Deduplicate at product level
```python
df = df.drop_duplicates(subset=["product_id"], keep="first")
```
The raw CSV bundles multiple review rows per product under the same `product_id`.  
After dedup: **1,464 → 1,350 unique products**.

---

#### Step 8 — Final NaN drop on modelling columns
```python
feature_cols = ["discounted_price", "actual_price", "discount_percentage",
                "rating_count", "main_category", "rating"]
df = df.dropna(subset=feature_cols)
```
Removes any remaining rows where a required feature or the target is missing.

### Before / After Summary

| Metric | Value |
|---|---|
| Raw rows | 1,465 |
| After dropping bad `rating` row | 1,464 |
| After deduplication by `product_id` | 1,350 |
| Final modelling rows | **1,350** |
| Invalid rating rows dropped | 1 |
| Missing `rating_count` imputed (not dropped) | 2 |
| Duplicate product rows removed | 114 |

---

## 5. Model — Linear Regression

### Why Linear Regression?

- **Interpretable** — each coefficient is a direct, signed measure of a feature's influence on the predicted rating.
- **Appropriate target** — `rating` is a continuous variable (1.0–5.0).
- **Transparent** — the model equation can be printed, documented, and explained to non-technical stakeholders.
- **Baseline** — establishes a performance floor before trying ensemble or neural approaches.

### Pipeline Architecture

```
ColumnTransformer
├── passthrough  →  discounted_price, actual_price,
│                   discount_percentage, rating_count
└── OneHotEncoder(handle_unknown="ignore")  →  main_category
        │
        ▼
  LinearRegression()
```

Built with `sklearn.pipeline.Pipeline` → serialised as `model.pkl`.

### Features

| Feature | Type | Notes |
|---|---|---|
| `discounted_price` | Numeric (float64) | Sale price in INR |
| `actual_price` | Numeric (float64) | MRP in INR |
| `discount_percentage` | Numeric (float64, 0–1) | Fraction e.g. 0.64 = 64 % |
| `rating_count` | Numeric (float64) | Total number of ratings |
| `main_category` | Categorical → OHE | 9 categories, 9 binary columns |

**Target:** `rating` (float, 1.0–5.0)

### Train / Test Split

| Set | Rows |
|---|---|
| Training (80 %) | 1,080 |
| Test (20 %) | 270 |
| `random_state` | 42 |

---

## 6. Model Metrics

Evaluated on the **held-out 20 % test set** (270 products never seen during training).

| Metric | Value | Interpretation |
|---|---|---|
| **R² (R-squared)** | **0.1069** | ~10.7 % of rating variance explained by pricing + category features |
| **MAE (Mean Absolute Error)** | **0.2020 stars** | Average prediction is within 0.20 stars of the true rating |
| **RMSE (Root Mean Squared Error)** | **0.2683 stars** | Penalised error; no large systematic outliers |

> **Note on R²:** The modest R² is expected and explainable. Product ratings on Amazon India are tightly clustered between 3.8–4.4 (very low variance), and the available structured features (price, discount, category) capture only a fraction of what drives subjective customer satisfaction. Text-based features (sentiment, keyword analysis) and temporal signals (recency, velocity) would substantially raise R².

---

## 7. Model Equation & Coefficients

```
Rating = 4.182042
       + (-0.000012) × discounted_price
       + (+0.000011) × actual_price
       + (-0.291153) × discount_percentage
       + (+0.000001) × rating_count
       + category one-hot terms (below)
```

### Category Coefficients (relative to the OHE reference level)

| Feature | Coefficient | Direction |
|---|---|---|
| `HomeImprovement` | +0.226908 | ↑ Higher ratings |
| `OfficeProducts` | +0.152471 | ↑ Higher ratings |
| `Toys&Games` | +0.105093 | ↑ Higher ratings |
| `Computers&Accessories` | +0.101557 | ↑ Higher ratings |
| `Electronics` | −0.029565 | ↓ Lower ratings |
| `Health&PersonalCare` | −0.041342 | ↓ Lower ratings |
| `Home&Kitchen` | −0.045504 | ↓ Lower ratings |
| `MusicalInstruments` | −0.191988 | ↓ Lower ratings |
| `Car&Motorbike` | −0.277629 | ↓ Lower ratings |

### Key takeaways

- **Discount percentage has the largest numeric impact** (−0.291 per unit fraction). A product with 100 % discount would see ~0.29 lower predicted rating vs 0 % — aligning with findings that extreme discounts can attract lower-quality-expectation buyers.
- **Price features have near-zero coefficients** — confirming that raw price alone is a very weak predictor of rating.
- **HomeImprovement and OfficeProducts** products tend to rate higher; **Car&Motorbike and MusicalInstruments** lower.
- **rating_count has a positive but tiny coefficient** (+0.000001), consistent with the observation that more popular products drift toward the 4.0–4.2 mean.

---

## 8. Setup

### Prerequisites

- Python 3.9 or higher
- `pip`

### Install all dependencies

```bash
pip install -r requirements.txt
```

### What gets installed

| Package | Version | Used in |
|---|---|---|
| `flask` | ≥ 3.0.0 | `app.py` — REST API framework |
| `flask-cors` | ≥ 4.0.0 | `app.py` — Cross-origin request headers |
| `streamlit` | ≥ 1.35.0 | `ui.py` — Interactive dashboard |
| `pandas` | ≥ 2.1.0 | All files — data manipulation |
| `numpy` | ≥ 1.26.0 | All files — numerical operations |
| `scikit-learn` | ≥ 1.4.0 | `train_model.py`, `app.py` — ML pipeline |
| `matplotlib` | ≥ 3.8.0 | `ui.py`, `generate_screenshots.py` — charts |
| `seaborn` | ≥ 0.13.0 | `ui.py` — statistical visualisations |
| `scipy` | ≥ 1.12.0 | `ui.py`, `generate_screenshots.py` — OLS regression stats |
| `requests` | ≥ 2.31.0 | `ui.py` — HTTP calls to Flask |
| `python-docx` | ≥ 1.1.0 | `build_report.py` — Word document generation |
| `openpyxl` | ≥ 3.1.0 | pandas Excel support |
| `Pillow` | ≥ 10.0.0 | Image handling in python-docx |

---

## 9. Run Order

Follow this exact sequence. Each step depends on the outputs of the previous one.

### Step 1 — Train the model

```bash
python train_model.py
```

**What it does:**
- Reads `amazon.csv`
- Performs all 8 data cleaning steps (see [Section 4](#4-data-cleaning-steps))
- Trains a `LinearRegression` pipeline
- Evaluates on 20 % test split
- Prints R², MAE, RMSE and model equation to console
- Saves:
  - `model.pkl` — serialised scikit-learn Pipeline
  - `model_info.json` — metrics, coefficients, categories
  - `amazon_clean.csv` — cleaned dataset for Streamlit EDA

**Expected console output:**
```
============================================================
STEP 1 – Loading amazon.csv
============================================================
Raw shape : (1465, 16)

STEP 2 – Cleaning
  Rows with invalid rating (e.g. '|') : 1
  Missing rating_count values         : 2
  Rows before dedup                   : 1464
  Rows after  dedup                   : 1350
  Before cleaning : 1465 rows
  After  cleaning : 1350 rows

STEP 3 – Model Performance (held-out test set)
  R²   : 0.1069
  MAE  : 0.2020
  RMSE : 0.2683

STEP 4 – Model Equation
  intercept : 4.182042
  ...

[OK] model.pkl        saved
[OK] model_info.json  saved
[OK] amazon_clean.csv saved
```

---

### Step 2 — Start the Flask backend

Open a **dedicated terminal** and keep it running:

```bash
python app.py
```

**What it does:**
- Loads `model.pkl` and `model_info.json` into memory at startup
- Starts a development server on `http://127.0.0.1:5000`
- Serves three endpoints (see [Section 10](#10-flask-api-reference))

**Expected output:**
```
Starting Flask server on http://127.0.0.1:5000
 * Running on http://127.0.0.1:5000
```

> Keep this terminal open while using the Streamlit UI.

---

### Step 3 — Launch the Streamlit dashboard

Open a **second terminal**:

```bash
streamlit run ui.py
```

**What it does:**
- Opens a browser tab at `http://localhost:8501`
- Loads `amazon_clean.csv` and `model_info.json` (cached)
- Renders 4 pages navigable from the sidebar
- Page 4 (Predict) sends live `POST` requests to the Flask backend

---

### Step 4 (Optional) — Regenerate screenshots

```bash
python generate_screenshots.py
```

**What it does:**
- Reads `amazon_clean.csv` and `model_info.json`
- Renders all 4 Streamlit pages as high-resolution (130 DPI) PNGs
- Saves to `screenshots/` — no browser required (pure matplotlib)

**Output files:**
```
screenshots/page1_dataset_overview.png
screenshots/page2_eda.png
screenshots/page3_model_insights.png
screenshots/page4_predict.png
```

---

### Step 5 (Optional) — Build the Word report

```bash
python build_report.py
```

**What it does:**
- Reads `model_info.json` for all metrics and coefficients
- Embeds the 4 PNG screenshots from `screenshots/`
- Produces a fully-formatted, 10-section Word document
- Saves: `Amazon_Rating_Prediction_Report.docx`

> Requires `python-docx` (included in `requirements.txt`). Run steps 1 and 4 first so that `model_info.json` and all PNGs exist.

---

### Complete run sequence at a glance

```bash
# 1. Install dependencies (once)
pip install -r requirements.txt

# 2. Train model and produce all ML artefacts
python train_model.py

# 3a. Start Flask API (Terminal A — keep running)
python app.py

# 3b. Start Streamlit UI (Terminal B)
streamlit run ui.py

# Optional — regenerate screenshots
python generate_screenshots.py

# Optional — rebuild Word report
python build_report.py
```

---

## 10. Flask API Reference

Base URL: `http://127.0.0.1:5000`

### `GET /health`

Simple liveness check.

```bash
curl http://127.0.0.1:5000/health
```
```json
{"status": "ok"}
```

---

### `GET /model-info`

Returns all model metadata from `model_info.json`.

```bash
curl http://127.0.0.1:5000/model-info
```
```json
{
  "r2": 0.1069,
  "mae": 0.202,
  "rmse": 0.2683,
  "intercept": 4.182042,
  "coefficients": {
    "discounted_price": -1.2e-05,
    "actual_price":     1.1e-05,
    "discount_percentage": -0.291153,
    "rating_count":     1e-06,
    "Car&Motorbike":    -0.277629,
    "Computers&Accessories": 0.101557,
    ...
  },
  "categories": ["Car&Motorbike", "Computers&Accessories", ...],
  "train_size": 1080,
  "test_size":  270
}
```

---

### `POST /predict`

Predict the rating for a product.

**Request body (JSON):**

| Field | Type | Required | Notes |
|---|---|---|---|
| `discounted_price` | float | Yes | Sale price in INR |
| `actual_price` | float | Yes | MRP in INR |
| `discount_percentage` | float | Yes | As decimal — e.g. 64 % → `0.64` |
| `rating_count` | float | Yes | Number of existing ratings |
| `main_category` | string | Yes | Must be one of the 9 categories |

**Example request:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "discounted_price": 399,
    "actual_price": 1099,
    "discount_percentage": 0.64,
    "rating_count": 24269,
    "main_category": "Computers&Accessories"
  }'
```

**Example response:**
```json
{
  "predicted_rating": 4.12,
  "input": {
    "discounted_price": 399,
    "actual_price": 1099,
    "discount_percentage": 0.64,
    "rating_count": 24269,
    "main_category": "Computers&Accessories"
  }
}
```

> Predicted ratings are clamped to the valid range **[1.0, 5.0]** before being returned.

**Error response (missing field):**
```json
{"error": "Missing fields: ['main_category']"}
```

---

## 11. Streamlit UI Pages

Navigate using the **sidebar radio buttons**. Flask backend must be running for Page 4.

### Page 1 — Dataset Overview (`📊 Dataset Overview`)

| Element | Description |
|---|---|
| Metric cards | Total products, feature count, avg rating, main category count |
| Sample rows table | First 10 rows — product name, category, prices, discount, rating, rating count |
| Column dtypes table | Data type of every column in `amazon_clean.csv` |
| Missing values report | Any remaining NaN counts (should be zero after cleaning) |
| Category bar chart | Horizontal bar chart of products per `main_category` |
| Descriptive statistics | `describe()` on the 5 numeric model features |

---

### Page 2 — EDA (`🔍 EDA`)

| Chart | X-axis | Y-axis | Extra |
|---|---|---|---|
| Rating distribution histogram | Rating (1–5) | Frequency | 20 bins |
| Price vs Rating scatter | Discounted price (98th-pct capped) | Rating | OLS trendline + R value |
| Discount % vs Rating scatter | Discount percentage (%) | Rating | OLS trendline + R value |
| Rating count vs Rating scatter | log(1 + rating\_count) | Rating | OLS trendline + R value |
| Category boxplot | Main category | Rating | Sorted by median |

---

### Page 3 — Model Insights (`🧠 Model Insights`)

| Element | Description |
|---|---|
| Metric cards | R², MAE, RMSE |
| Model equation | Formatted markdown showing intercept and numeric coefficients |
| Coefficients table | All 13 features sorted by coefficient value (descending) |
| Coefficient bar chart | Horizontal bar — blue = positive, red = negative |
| OLS trendline chart | Full dataset scatter + OLS line (discounted price vs rating) |
| Train/test size cards | 1 080 training, 270 test |

---

### Page 4 — Predict (`🎯 Predict`)

| Element | Description |
|---|---|
| `discounted_price` | Number input (₹1 – ₹500 000, default ₹399) |
| `actual_price` | Number input (₹1 – ₹500 000, default ₹1 099) |
| `discount_percentage` | Slider (0 – 100 %, default 64 %) |
| `rating_count` | Number input (1 – 5 000 000, default 24 269) |
| `main_category` | Selectbox — 9 categories from `model_info.json` |
| **Predict button** | POSTs to Flask `/predict`; shows predicted rating + star display |
| Input summary table | Echoes the submitted values for reference |

> If Flask is not running, the page shows a clear error message with instructions.

---

## 12. Screenshots

Programmatically generated by `generate_screenshots.py` using matplotlib — no browser required.

| File | Page |
|---|---|
| `screenshots/page1_dataset_overview.png` | Dataset Overview — metrics, category bar chart, sample rows, statistics |
| `screenshots/page2_eda.png` | EDA — 4-panel chart grid (histogram, 2× scatter, boxplot) |
| `screenshots/page3_model_insights.png` | Model Insights — metrics, coefficient chart, OLS trendline, equation + table |
| `screenshots/page4_predict.png` | Predict — form mockup with input fields, predict button, result card |

---

## 13. Report

**File:** `Amazon_Rating_Prediction_Report.docx`  
**Generated by:** `build_report.py`  
**Approx. length:** 12–15 pages

### Report Sections

| # | Section | Contents |
|---|---|---|
| 1 | Cover page | Title, author (Himanshu), date, course name |
| 2 | Abstract | 150–180 word summary of the project, data, method, and results |
| 3 | Objective | Business question + 5 bullet-point goals |
| 4 | Dataset Description | Source, raw shape, all 16 column descriptions, 6 data quality issues |
| 5 | Data Cleaning Steps | All 7 steps with rationale and row-count before/after table |
| 6 | EDA | Key findings + embedded `page2_eda.png` |
| 7 | Model Methodology | Why Linear Regression, feature table, pipeline architecture |
| 8 | Results | Metric table, model equation (monospace), full coefficients table + `page3_model_insights.png` |
| 9 | UI Walkthrough | 4 subsections, one screenshot per page (`page1` through `page4`) |
| 10 | Conclusion & Future Scope | Summary paragraph + 7-item future work list |

---

## 14. Dependencies

Full `requirements.txt`:

```
flask>=3.0.0
flask-cors>=4.0.0
streamlit>=1.35.0
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
scipy>=1.12.0
requests>=2.31.0
python-docx>=1.1.0
openpyxl>=3.1.0
Pillow>=10.0.0
```

---

## 15. Author

**Name:** Yashwanth Raj KS 
**Course:** IBM Data Science Professional Certificate  
**Project:** Amazon Product Rating Prediction  
**Date:** 2026

---

*Built end-to-end in Python — data cleaning, machine learning, REST API, interactive dashboard, automated screenshots, and a fully formatted project report.*
