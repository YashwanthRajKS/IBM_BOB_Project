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
