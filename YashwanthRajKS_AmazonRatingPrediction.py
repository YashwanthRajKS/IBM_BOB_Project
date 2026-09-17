"""
YashwanthRajKS_AmazonRatingPrediction.py
========================================
Submission File: Amazon Product Rating Prediction
Author: Yashwanth Raj K S

This single-file submission combines the three core components of the project:
  1. Model Training & Data Preprocessing (train_model.py)
  2. Flask REST API Backend (app.py)
  3. Streamlit Interactive Web Application (ui.py)

Note:
  The individual modular files (train_model.py, app.py, and ui.py) are retained
  in the project repository for running the pipeline in production / development.
"""

# ===== MODEL TRAINING =====

"""
train_model.py  –  Amazon Product Rating Prediction
Cleans the amazon.csv dataset, trains a Linear Regression model,
reports metrics, and saves model.pkl + model_info.json.
"""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  LOAD
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 – Loading amazon.csv")
print("=" * 60)

df = pd.read_csv("amazon.csv")
print(f"Raw shape : {df.shape}")
print(f"Columns   : {list(df.columns)}\n")

# ─────────────────────────────────────────────
# 2.  CLEAN
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 2 – Cleaning")
print("=" * 60)

before_rows = len(df)

# --- discounted_price ---
df["discounted_price"] = (
    df["discounted_price"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors="coerce")

# --- actual_price ---
df["actual_price"] = (
    df["actual_price"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["actual_price"] = pd.to_numeric(df["actual_price"], errors="coerce")

# --- discount_percentage  (strip %, convert, divide by 100) ---
df["discount_percentage"] = (
    df["discount_percentage"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
)
df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce") / 100

# --- rating  (drop the row where rating == "|") ---
df["rating"] = pd.to_numeric(df["rating"].astype(str).str.strip(), errors="coerce")
bad_rating_rows = df["rating"].isna().sum()
print(f"  Rows with invalid rating (e.g. '|') : {bad_rating_rows}")
df = df.dropna(subset=["rating"])

# --- rating_count  (strip commas, impute 2 missing values with median) ---
df["rating_count"] = (
    df["rating_count"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")
missing_rc = df["rating_count"].isna().sum()
print(f"  Missing rating_count values         : {missing_rc}")
df["rating_count"].fillna(df["rating_count"].median(), inplace=True)

# --- category  → main_category + sub_category ---
cat_split = df["category"].astype(str).str.split("|")
df["main_category"] = cat_split.str[0].str.strip()
df["sub_category"]  = cat_split.str[1].str.strip().fillna("Other")

# --- Deduplicate at product level ---
before_dedup = len(df)
df = df.drop_duplicates(subset=["product_id"], keep="first")
after_dedup  = len(df)
print(f"  Rows before dedup                   : {before_dedup}")
print(f"  Rows after  dedup                   : {after_dedup}")

# --- Drop rows with any NaN in feature / target columns ---
feature_cols = ["discounted_price", "actual_price", "discount_percentage",
                "rating_count", "main_category", "rating"]
df = df.dropna(subset=feature_cols)
after_rows = len(df)

print(f"\n  Before cleaning : {before_rows} rows")
print(f"  After  cleaning : {after_rows} rows")
print(f"\n  dtype check:")
print(df[feature_cols].dtypes.to_string())

# ─────────────────────────────────────────────
# 3.  FEATURE / TARGET SPLIT
# ─────────────────────────────────────────────
X = df[["discounted_price", "actual_price", "discount_percentage",
        "rating_count", "main_category"]]
y = df["rating"]

print(f"\nDataset used for modelling: {X.shape[0]} rows, {X.shape[1]} features")

# ─────────────────────────────────────────────
# 4.  PIPELINE  (OHE + LinearRegression)
# ─────────────────────────────────────────────
numeric_features = ["discounted_price", "actual_price",
                    "discount_percentage", "rating_count"]
categorical_features = ["main_category"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorical_features),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression()),
])

# ─────────────────────────────────────────────
# 5.  TRAIN / TEST SPLIT  &  FIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
pipeline.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 6.  EVALUATION
# ─────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "=" * 60)
print("STEP 3 – Model Performance (held-out test set)")
print("=" * 60)
print(f"  R²   : {r2:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")

# ─────────────────────────────────────────────
# 7.  MODEL EQUATION & COEFFICIENTS
# ─────────────────────────────────────────────
lr = pipeline.named_steps["regressor"]
ohe_cats = (pipeline.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .categories_[0])
feature_names = numeric_features + list(ohe_cats)
coef_dict = {name: round(float(coef), 6)
             for name, coef in zip(feature_names, lr.coef_)}

print("\n" + "=" * 60)
print("STEP 4 – Model Equation")
print("=" * 60)
print(f"  intercept : {lr.intercept_:.6f}")
for fname, coef in coef_dict.items():
    print(f"  {fname:<35} : {coef:+.6f}")

# ─────────────────────────────────────────────
# 8.  SAVE
# ─────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

categories = sorted(df["main_category"].unique().tolist())

model_info = {
    "r2": round(r2, 4),
    "mae": round(mae, 4),
    "rmse": round(rmse, 4),
    "intercept": round(float(lr.intercept_), 6),
    "coefficients": coef_dict,
    "categories": categories,
    "train_size": int(X_train.shape[0]),
    "test_size":  int(X_test.shape[0]),
}

with open("model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)

# Also save cleaned dataset for Streamlit EDA
df.to_csv("amazon_clean.csv", index=False)

print("\n[OK] model.pkl        saved")
print("[OK] model_info.json  saved")
print("[OK] amazon_clean.csv saved")
print("\nDone.")


# ===== FLASK BACKEND =====

"""
app.py  –  Flask backend for Amazon Product Rating Prediction
Endpoints:
  POST /predict       – return predicted rating for given inputs
  GET  /model-info    – return R², MAE, RMSE, coefficients
  GET  /health        – simple health-check
"""

import json
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Load model & metadata once at startup ────────────────────────
with open("model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open("model_info.json", "r") as f:
    MODEL_INFO = json.load(f)

# ── Routes ────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Simple health-check."""
    return jsonify({"status": "ok"})


@app.route("/model-info", methods=["GET"])
def model_info():
    """Return model performance metrics, coefficients, and available categories."""
    return jsonify(MODEL_INFO)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict product rating.

    Expected JSON body:
    {
        "discounted_price"   : 399,
        "actual_price"       : 1099,
        "discount_percentage": 0.64,      # as decimal, e.g. 64% → 0.64
        "rating_count"       : 24269,
        "main_category"      : "Computers&Accessories"
    }
    Returns:
    {
        "predicted_rating": 4.12,
        "input": { ... }
    }
    """
    data = request.get_json(force=True)

    required = ["discounted_price", "actual_price",
                "discount_percentage", "rating_count", "main_category"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        row = pd.DataFrame([{
            "discounted_price"   : float(data["discounted_price"]),
            "actual_price"       : float(data["actual_price"]),
            "discount_percentage": float(data["discount_percentage"]),
            "rating_count"       : float(data["rating_count"]),
            "main_category"      : str(data["main_category"]),
        }])

        pred = float(MODEL.predict(row)[0])
        # Clamp to valid rating range [1, 5]
        pred = max(1.0, min(5.0, round(pred, 2)))

        return jsonify({"predicted_rating": pred, "input": data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)


# ===== STREAMLIT UI =====

"""
ui.py  –  Streamlit frontend for Amazon Product Rating Prediction
Single-file, four pages:
  1. Dataset Overview
  2. EDA
  3. Model Insights
  4. Predict
"""

import json

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Rating Predictor",
    page_icon="⭐",
    layout="wide",
)

FLASK_URL = "http://127.0.0.1:5000"

# ── Load data & model info ────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("amazon_clean.csv")
    except FileNotFoundError:
        st.error("amazon_clean.csv not found. Please run train_model.py first.")
        st.stop()
    return df


@st.cache_data
def load_model_info():
    try:
        with open("model_info.json") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("model_info.json not found. Please run train_model.py first.")
        st.stop()


df = load_data()
info = load_model_info()

# ── Sidebar navigation ────────────────────────────────────────────
st.sidebar.title("🛒 Amazon Rating Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dataset Overview", "🔍 EDA", "🧠 Model Insights", "🎯 Predict"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Model: Linear Regression | Data: Amazon India")

# ═══════════════════════════════════════════════════════════════════
# PAGE 1 – Dataset Overview
# ═══════════════════════════════════════════════════════════════════
if page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown(
        "The dataset contains Amazon India product listings with prices, "
        "discounts, ratings, and review counts."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Products", f"{df.shape[0]:,}")
    c2.metric("Features", df.shape[1])
    c3.metric("Avg Rating", f"{df['rating'].mean():.2f}")
    c4.metric("Main Categories", df["main_category"].nunique())

    st.markdown("### Sample Rows")
    st.dataframe(
        df[["product_id", "product_name", "main_category", "sub_category",
            "discounted_price", "actual_price", "discount_percentage",
            "rating", "rating_count"]].head(10),
        use_container_width=True,
    )

    st.markdown("### Column Data Types")
    dtype_df = pd.DataFrame(
        {"Column": df.dtypes.index, "Dtype": df.dtypes.values.astype(str)}
    )
    st.dataframe(dtype_df, use_container_width=True)

    st.markdown("### Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values in the cleaned dataset.")
    else:
        st.dataframe(missing.rename("Missing Count").reset_index(), use_container_width=True)

    st.markdown("### Main Category Distribution")
    cat_counts = df["main_category"].value_counts().reset_index()
    cat_counts.columns = ["main_category", "count"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(cat_counts["main_category"], cat_counts["count"],
            color="#3b82d4", edgecolor="white")
    ax.set_xlabel("Number of Products")
    ax.set_title("Products per Main Category")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Descriptive Statistics")
    st.dataframe(
        df[["discounted_price", "actual_price", "discount_percentage",
            "rating", "rating_count"]].describe().round(2),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 – EDA
# ═══════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")

    # -- Rating distribution --
    st.markdown("### Rating Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["rating"], bins=20, color="#3b82d4", edgecolor="white", rwidth=0.85)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Product Ratings")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # -- Price vs Rating --
    st.markdown("### Discounted Price vs Rating")
    fig, ax = plt.subplots(figsize=(8, 4))
    price_cap = df["discounted_price"].clip(upper=df["discounted_price"].quantile(0.98))
    ax.scatter(price_cap, df["rating"], alpha=0.35, s=20,
               color="#3b82d4", edgecolors="none")
    m, b, r, p, se = stats.linregress(price_cap, df["rating"])
    xline = np.linspace(price_cap.min(), price_cap.max(), 200)
    ax.plot(xline, m * xline + b, color="#e05d44", linewidth=2, label=f"OLS (R={r:.2f})")
    ax.set_xlabel("Discounted Price (₹)")
    ax.set_ylabel("Rating")
    ax.set_title("Discounted Price vs Rating (98th-pct price cap)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # -- Discount % vs Rating --
    st.markdown("### Discount Percentage vs Rating")
    fig, ax = plt.subplots(figsize=(8, 4))
    disc_pct = df["discount_percentage"] * 100
    ax.scatter(disc_pct, df["rating"], alpha=0.35, s=20,
               color="#7c5cd8", edgecolors="none")
    m2, b2, r2, _, _ = stats.linregress(disc_pct, df["rating"])
    xline2 = np.linspace(disc_pct.min(), disc_pct.max(), 200)
    ax.plot(xline2, m2 * xline2 + b2, color="#e05d44", linewidth=2,
            label=f"OLS (R={r2:.2f})")
    ax.set_xlabel("Discount Percentage (%)")
    ax.set_ylabel("Rating")
    ax.set_title("Discount Percentage vs Rating")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # -- Rating count vs Rating --
    st.markdown("### Rating Count vs Rating")
    fig, ax = plt.subplots(figsize=(8, 4))
    rc_log = np.log1p(df["rating_count"])
    ax.scatter(rc_log, df["rating"], alpha=0.35, s=20,
               color="#f59e0b", edgecolors="none")
    m3, b3, r3, _, _ = stats.linregress(rc_log, df["rating"])
    xline3 = np.linspace(rc_log.min(), rc_log.max(), 200)
    ax.plot(xline3, m3 * xline3 + b3, color="#e05d44", linewidth=2,
            label=f"OLS (R={r3:.2f})")
    ax.set_xlabel("log(1 + rating_count)")
    ax.set_ylabel("Rating")
    ax.set_title("Rating Count (log-scale) vs Rating")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # -- Category box plot --
    st.markdown("### Rating Distribution by Main Category")
    fig, ax = plt.subplots(figsize=(12, 5))
    cat_order = df.groupby("main_category")["rating"].median().sort_values().index.tolist()
    data_list = [df[df["main_category"] == c]["rating"].dropna().values for c in cat_order]
    bp = ax.boxplot(data_list, patch_artist=True, vert=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("#3b82d4")
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(cat_order) + 1))
    ax.set_xticklabels(cat_order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Rating")
    ax.set_title("Rating Distribution by Main Category")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 – Model Insights
# ═══════════════════════════════════════════════════════════════════
elif page == "🧠 Model Insights":
    st.title("🧠 Model Insights")

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{info['r2']:.4f}")
    c2.metric("MAE",       f"{info['mae']:.4f}")
    c3.metric("RMSE",      f"{info['rmse']:.4f}")

    st.markdown("### Model Equation")
    intercept = info["intercept"]
    eq_parts = [f"**{intercept:.4f}**"]
    numeric_features = ["discounted_price", "actual_price",
                        "discount_percentage", "rating_count"]
    for feat in numeric_features:
        coef = info["coefficients"].get(feat, 0)
        sign = "+" if coef >= 0 else "–"
        eq_parts.append(f"{sign} {abs(coef):.6f} × {feat}")
    eq_str = "  \n".join(eq_parts)
    st.markdown(
        f"**Rating** = {intercept:.4f}  \n"
        + "  \n".join([
            f"{'**+**' if info['coefficients'].get(f, 0) >= 0 else '**–**'} "
            f"`{abs(info['coefficients'].get(f, 0)):.6f}` × *{f}*"
            for f in numeric_features
        ])
        + "  \n+ *(one-hot encoded main_category coefficients below)*"
    )

    # Coefficients table
    st.markdown("### Feature Coefficients")
    coef_df = (
        pd.DataFrame(list(info["coefficients"].items()),
                     columns=["Feature", "Coefficient"])
        .sort_values("Coefficient", ascending=False)
    )
    st.dataframe(coef_df, use_container_width=True)

    # Coefficient bar chart
    fig, ax = plt.subplots(figsize=(10, max(4, len(coef_df) * 0.35)))
    colors = ["#3b82d4" if v >= 0 else "#e05d44" for v in coef_df["Coefficient"]]
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Linear Regression Coefficients")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # OLS trendline over full dataset (actual_price vs rating)
    st.markdown("### OLS Regression Trendline – Discounted Price vs Rating")
    fig, ax = plt.subplots(figsize=(9, 5))
    price_cap = df["discounted_price"].clip(upper=df["discounted_price"].quantile(0.98))
    ax.scatter(price_cap, df["rating"], alpha=0.25, s=18,
               color="#3b82d4", edgecolors="none", label="Data points")
    m, b, r, p, se = stats.linregress(price_cap, df["rating"])
    xline = np.linspace(price_cap.min(), price_cap.max(), 300)
    ax.plot(xline, m * xline + b, color="#e05d44", linewidth=2.5,
            label=f"OLS trendline  R={r:.3f}")
    ax.set_xlabel("Discounted Price (₹)")
    ax.set_ylabel("Rating")
    ax.set_title(f"Full Dataset – OLS Trendline  (R² = {info['r2']:.4f})")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Training & Test Set Sizes")
    c1, c2 = st.columns(2)
    c1.metric("Training samples", f"{info['train_size']:,}")
    c2.metric("Test samples",     f"{info['test_size']:,}")


# ═══════════════════════════════════════════════════════════════════
# PAGE 4 – Predict
# ═══════════════════════════════════════════════════════════════════
elif page == "🎯 Predict":
    st.title("🎯 Predict Product Rating")
    st.markdown(
        "Fill in the product details below and click **Predict** "
        "to get the estimated Amazon rating."
    )

    categories = info.get("categories", [])

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            discounted_price = st.number_input(
                "Discounted Price (₹)", min_value=1.0, max_value=500000.0,
                value=399.0, step=1.0
            )
            actual_price = st.number_input(
                "Actual / MRP (₹)", min_value=1.0, max_value=500000.0,
                value=1099.0, step=1.0
            )
        with c2:
            discount_pct_input = st.slider(
                "Discount Percentage (%)", min_value=0, max_value=100, value=64
            )
            rating_count = st.number_input(
                "Number of Ratings", min_value=1, max_value=5000000,
                value=24269, step=1
            )
        main_category = st.selectbox("Main Category", options=categories)
        submitted = st.form_submit_button("⭐ Predict Rating", use_container_width=True)

    if submitted:
        payload = {
            "discounted_price":    discounted_price,
            "actual_price":        actual_price,
            "discount_percentage": discount_pct_input / 100,
            "rating_count":        rating_count,
            "main_category":       main_category,
        }
        try:
            resp = requests.post(f"{FLASK_URL}/predict", json=payload, timeout=5)
            if resp.status_code == 200:
                pred = resp.json()["predicted_rating"]
                st.success(f"### Predicted Rating: ⭐ {pred}")
                stars = int(round(pred))
                st.markdown("**Visual indicator:** " + "⭐" * stars + "☆" * (5 - stars))

                st.markdown("---")
                st.markdown("**Input summary**")
                summary = {
                    "Discounted Price (₹)": f"₹{discounted_price:,.0f}",
                    "Actual Price (₹)":     f"₹{actual_price:,.0f}",
                    "Discount":             f"{discount_pct_input}%",
                    "Rating Count":         f"{rating_count:,}",
                    "Category":             main_category,
                }
                st.table(pd.DataFrame(summary.items(), columns=["Field", "Value"]))
            else:
                st.error(f"Flask error {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot reach Flask backend at http://127.0.0.1:5000. "
                "Please run `python app.py` in a separate terminal."
            )
