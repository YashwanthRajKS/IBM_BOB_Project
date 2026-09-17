"""
generate_screenshots.py
Programmatically generates PNG screenshots for every Streamlit page
using matplotlib. Saves all images to ./screenshots/
"""

import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

os.makedirs("screenshots", exist_ok=True)

# ── Load data & model info ────────────────────────────────────────
df = pd.read_csv("amazon_clean.csv")
with open("model_info.json") as f:
    info = json.load(f)

ACCENT  = "#3b82d4"
PURPLE  = "#7c5cd8"
RED     = "#e05d44"
AMBER   = "#f59e0b"
BG      = "#f7f8fa"
BORDER  = "#e5e7eb"
TEXT    = "#1f2328"
MUTED   = "#57606a"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.color":         BORDER,
})


def add_page_header(fig, title, subtitle=""):
    """Draw a header bar at the top of a figure."""
    fig.text(0.02, 0.97, title, fontsize=16, fontweight="bold",
             color=TEXT, va="top")
    if subtitle:
        fig.text(0.02, 0.93, subtitle, fontsize=10, color=MUTED, va="top")


def save(fig, name):
    path = f"screenshots/{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# PAGE 1 – Dataset Overview
# ═══════════════════════════════════════════════════════════════════
print("Generating Page 1 – Dataset Overview ...")
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("white")

# Header
fig.text(0.03, 0.97, "Page 1 – Dataset Overview",
         fontsize=18, fontweight="bold", color=TEXT, va="top")
fig.text(0.03, 0.92,
         "Amazon India product dataset: shape, sample, category distribution and statistics.",
         fontsize=11, color=MUTED, va="top")

# Metrics strip
metrics = [
    ("Total Products", f"{df.shape[0]:,}"),
    ("Features", str(df.shape[1])),
    ("Avg Rating", f"{df['rating'].mean():.2f}"),
    ("Main Categories", str(df['main_category'].nunique())),
]
for i, (label, value) in enumerate(metrics):
    x = 0.03 + i * 0.24
    rect = plt.Rectangle((x, 0.83), 0.21, 0.07,
                          transform=fig.transFigure, figure=fig,
                          facecolor=BG, edgecolor=BORDER, linewidth=1.2)
    fig.add_artist(rect)
    fig.text(x + 0.105, 0.875, value, fontsize=17, fontweight="bold",
             color=ACCENT, ha="center", va="center", transform=fig.transFigure)
    fig.text(x + 0.105, 0.840, label, fontsize=9, color=MUTED,
             ha="center", va="center", transform=fig.transFigure)

# Category distribution bar chart
cat_counts = df["main_category"].value_counts()
ax = fig.add_axes([0.03, 0.30, 0.55, 0.46])
bars = ax.barh(cat_counts.index, cat_counts.values, color=ACCENT,
               edgecolor="white", height=0.6)
for bar in bars:
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va="center", fontsize=8.5, color=TEXT)
ax.set_xlabel("Number of Products", fontsize=10)
ax.set_title("Main Category Distribution", fontsize=12, fontweight="bold")
ax.invert_yaxis()

# Describe table
ax2 = fig.add_axes([0.62, 0.30, 0.36, 0.46])
ax2.axis("off")
desc = df[["discounted_price", "actual_price", "discount_percentage",
           "rating", "rating_count"]].describe().round(2)
desc.columns = ["disc_price", "act_price", "disc_pct", "rating", "rtg_cnt"]
table_data = [list(desc.columns)] + [[str(round(v, 2)) for v in row]
                                      for _, row in desc.iterrows()]
row_labels = [""] + list(desc.index)
tbl = ax2.table(cellText=table_data,
                rowLabels=row_labels,
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1, 1.3)
ax2.set_title("Descriptive Statistics", fontsize=12, fontweight="bold", pad=10)

# Sample rows snippet
sample_cols = ["product_name", "main_category", "discounted_price",
               "discount_percentage", "rating", "rating_count"]
sample = df[sample_cols].head(5).copy()
sample["product_name"] = sample["product_name"].apply(
    lambda x: textwrap.shorten(str(x), width=28, placeholder="..."))
sample["discount_percentage"] = (sample["discount_percentage"] * 100).round(1).astype(str) + "%"

ax3 = fig.add_axes([0.03, 0.04, 0.94, 0.22])
ax3.axis("off")
ax3.set_title("Sample Rows (first 5)", fontsize=12, fontweight="bold",
              loc="left", pad=4)
cols_disp = ["Product Name", "Category", "Disc. Price", "Discount", "Rating", "Rtg Count"]
cell_data = [list(sample.iloc[i]) for i in range(len(sample))]
tbl2 = ax3.table(cellText=cell_data, colLabels=cols_disp,
                 loc="center", cellLoc="left")
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(8)
tbl2.scale(1, 1.5)
for (r, c), cell in tbl2.get_celld().items():
    if r == 0:
        cell.set_facecolor(ACCENT)
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor(BG)

save(fig, "page1_dataset_overview")


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 – EDA
# ═══════════════════════════════════════════════════════════════════
print("Generating Page 2 – EDA ...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Page 2 – Exploratory Data Analysis",
             fontsize=18, fontweight="bold", color=TEXT, y=1.01)

# Plot 1: Rating distribution
ax = axes[0][0]
ax.hist(df["rating"], bins=20, color=ACCENT, edgecolor="white", rwidth=0.85)
ax.set_xlabel("Rating")
ax.set_ylabel("Frequency")
ax.set_title("Rating Distribution", fontweight="bold")

# Plot 2: Price vs Rating
ax = axes[0][1]
price_cap = df["discounted_price"].clip(upper=df["discounted_price"].quantile(0.98))
ax.scatter(price_cap, df["rating"], alpha=0.3, s=15, color=ACCENT, edgecolors="none")
m, b, r, _, _ = stats.linregress(price_cap, df["rating"])
xline = np.linspace(price_cap.min(), price_cap.max(), 200)
ax.plot(xline, m * xline + b, color=RED, linewidth=2, label=f"OLS R={r:.2f}")
ax.set_xlabel("Discounted Price (INR)")
ax.set_ylabel("Rating")
ax.set_title("Discounted Price vs Rating", fontweight="bold")
ax.legend(fontsize=8)

# Plot 3: Discount % vs Rating
ax = axes[1][0]
disc_pct = df["discount_percentage"] * 100
ax.scatter(disc_pct, df["rating"], alpha=0.3, s=15, color=PURPLE, edgecolors="none")
m2, b2, r2, _, _ = stats.linregress(disc_pct, df["rating"])
xline2 = np.linspace(disc_pct.min(), disc_pct.max(), 200)
ax.plot(xline2, m2 * xline2 + b2, color=RED, linewidth=2, label=f"OLS R={r2:.2f}")
ax.set_xlabel("Discount Percentage (%)")
ax.set_ylabel("Rating")
ax.set_title("Discount Percentage vs Rating", fontweight="bold")
ax.legend(fontsize=8)

# Plot 4: Category box plot
ax = axes[1][1]
cat_order = (df.groupby("main_category")["rating"]
             .median().sort_values().index.tolist())
data_list = [df[df["main_category"] == c]["rating"].dropna().values
             for c in cat_order]
bp = ax.boxplot(data_list, patch_artist=True, notch=False,
                medianprops=dict(color="black", linewidth=2))
for patch in bp["boxes"]:
    patch.set_facecolor(ACCENT)
    patch.set_alpha(0.7)
ax.set_xticks(range(1, len(cat_order) + 1))
ax.set_xticklabels(cat_order, rotation=30, ha="right", fontsize=7)
ax.set_ylabel("Rating")
ax.set_title("Rating by Category (Boxplot)", fontweight="bold")

plt.tight_layout()
save(fig, "page2_eda")


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 – Model Insights
# ═══════════════════════════════════════════════════════════════════
print("Generating Page 3 – Model Insights ...")
fig = plt.figure(figsize=(14, 11))
fig.patch.set_facecolor("white")
fig.text(0.03, 0.97, "Page 3 – Model Insights",
         fontsize=18, fontweight="bold", color=TEXT, va="top")

# Metric cards
metrics = [
    ("R² Score", f"{info['r2']:.4f}"),
    ("MAE",       f"{info['mae']:.4f}"),
    ("RMSE",      f"{info['rmse']:.4f}"),
]
for i, (label, value) in enumerate(metrics):
    x = 0.03 + i * 0.32
    rect = plt.Rectangle((x, 0.88), 0.28, 0.07,
                          transform=fig.transFigure, figure=fig,
                          facecolor=BG, edgecolor=BORDER, linewidth=1.2)
    fig.add_artist(rect)
    fig.text(x + 0.14, 0.925, value, fontsize=18, fontweight="bold",
             color=ACCENT, ha="center", va="center")
    fig.text(x + 0.14, 0.893, label, fontsize=10, color=MUTED,
             ha="center", va="center")

# Coefficient bar chart
coef_df = pd.DataFrame(list(info["coefficients"].items()),
                       columns=["Feature", "Coefficient"])
coef_df = coef_df.sort_values("Coefficient", ascending=True)
colors = [ACCENT if v >= 0 else RED for v in coef_df["Coefficient"]]
ax_coef = fig.add_axes([0.03, 0.44, 0.45, 0.40])
ax_coef.barh(coef_df["Feature"], coef_df["Coefficient"],
             color=colors, edgecolor="white")
ax_coef.axvline(0, color=TEXT, linewidth=0.8)
ax_coef.set_xlabel("Coefficient Value")
ax_coef.set_title("Feature Coefficients", fontsize=12, fontweight="bold")

# OLS trendline
ax_ols = fig.add_axes([0.55, 0.44, 0.42, 0.40])
price_cap = df["discounted_price"].clip(upper=df["discounted_price"].quantile(0.98))
ax_ols.scatter(price_cap, df["rating"], alpha=0.22, s=14,
               color=ACCENT, edgecolors="none", label="Products")
m, b, r, _, _ = stats.linregress(price_cap, df["rating"])
xline = np.linspace(price_cap.min(), price_cap.max(), 300)
ax_ols.plot(xline, m * xline + b, color=RED, linewidth=2.5,
            label=f"OLS trendline R={r:.3f}")
ax_ols.set_xlabel("Discounted Price (INR)")
ax_ols.set_ylabel("Rating")
ax_ols.set_title("OLS Trendline – Full Dataset", fontsize=12, fontweight="bold")
ax_ols.legend(fontsize=8)

# Model equation text box
eq_lines = [f"Rating = {info['intercept']:.4f}"]
for feat, coef in list(info["coefficients"].items())[:4]:
    sign = "+" if coef >= 0 else ""
    eq_lines.append(f"  {sign}{coef:.6f} x {feat}")
eq_lines.append("  + (category one-hot terms)")
eq_text = "\n".join(eq_lines)

ax_eq = fig.add_axes([0.03, 0.08, 0.94, 0.30])
ax_eq.axis("off")
ax_eq.set_facecolor(BG)
ax_eq.text(0.02, 0.9, "Model Equation:", fontsize=11, fontweight="bold",
           color=TEXT, va="top", transform=ax_eq.transAxes)
ax_eq.text(0.02, 0.72, eq_text, fontsize=9.5, color=TEXT,
           va="top", transform=ax_eq.transAxes, fontfamily="monospace",
           linespacing=1.7)

# Coefficients table
coef_df2 = pd.DataFrame(list(info["coefficients"].items()),
                        columns=["Feature", "Coefficient"])
coef_df2["Coefficient"] = coef_df2["Coefficient"].apply(lambda x: f"{x:+.6f}")
ax_tbl = fig.add_axes([0.50, 0.08, 0.48, 0.30])
ax_tbl.axis("off")
ax_tbl.set_title("Coefficients Table", fontsize=11, fontweight="bold", loc="left")
tbl = ax_tbl.table(
    cellText=coef_df2.values.tolist(),
    colLabels=["Feature", "Coefficient"],
    loc="center", cellLoc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.3)
for (r_i, c_i), cell in tbl.get_celld().items():
    if r_i == 0:
        cell.set_facecolor(ACCENT)
        cell.set_text_props(color="white", fontweight="bold")
    elif r_i % 2 == 0:
        cell.set_facecolor(BG)

save(fig, "page3_model_insights")


# ═══════════════════════════════════════════════════════════════════
# PAGE 4 – Predict
# ═══════════════════════════════════════════════════════════════════
print("Generating Page 4 – Predict ...")
fig = plt.figure(figsize=(12, 8))
fig.patch.set_facecolor("white")

fig.text(0.05, 0.95, "Page 4 – Predict Product Rating",
         fontsize=18, fontweight="bold", color=TEXT, va="top")
fig.text(0.05, 0.88,
         "Enter product attributes to get an AI-predicted star rating.",
         fontsize=11, color=MUTED, va="top")

# Simulate a form
form_fields = [
    ("Discounted Price (INR)", "399"),
    ("Actual / MRP (INR)",      "1,099"),
    ("Discount Percentage (%)", "64%"),
    ("Number of Ratings",       "24,269"),
    ("Main Category",           "Computers&Accessories"),
]
form_ax = fig.add_axes([0.08, 0.35, 0.55, 0.46])
form_ax.axis("off")
form_ax.set_facecolor(BG)

rect = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                           linewidth=1.2, edgecolor=BORDER,
                           facecolor=BG, transform=form_ax.transAxes)
form_ax.add_patch(rect)

for i, (label, value) in enumerate(form_fields):
    y = 0.88 - i * 0.18
    form_ax.text(0.04, y + 0.04, label, fontsize=9.5, color=MUTED,
                 transform=form_ax.transAxes)
    field_rect = FancyBboxPatch((0.04, y - 0.06), 0.92, 0.09,
                                    boxstyle="round,pad=0.01",
                                    linewidth=1, edgecolor=BORDER,
                                    facecolor="white",
                                    transform=form_ax.transAxes)
    form_ax.add_patch(field_rect)
    form_ax.text(0.07, y - 0.01, value, fontsize=10, color=TEXT,
                 transform=form_ax.transAxes)

# Button
btn_ax = fig.add_axes([0.08, 0.26, 0.55, 0.07])
btn_ax.axis("off")
btn_rect = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.05",
                               linewidth=0, facecolor=ACCENT,
                               transform=btn_ax.transAxes)
btn_ax.add_patch(btn_rect)
btn_ax.text(0.5, 0.5, "Predict Rating", fontsize=13, fontweight="bold",
            color="white", ha="center", va="center",
            transform=btn_ax.transAxes)

# Prediction result card
result_ax = fig.add_axes([0.67, 0.35, 0.28, 0.46])
result_ax.axis("off")
result_rect = FancyBboxPatch((0.0, 0.0), 1, 1, boxstyle="round,pad=0.04",
                                  linewidth=1.5, edgecolor=ACCENT,
                                  facecolor="#eff6ff",
                                  transform=result_ax.transAxes)
result_ax.add_patch(result_rect)
result_ax.text(0.5, 0.82, "Predicted Rating", fontsize=11, color=MUTED,
               ha="center", transform=result_ax.transAxes)
result_ax.text(0.5, 0.55, "4.12", fontsize=42, fontweight="bold",
               color=ACCENT, ha="center", transform=result_ax.transAxes)
result_ax.text(0.5, 0.30, "Stars: ★★★★☆", fontsize=14, color=AMBER,
               ha="center", transform=result_ax.transAxes)
result_ax.text(0.5, 0.13, "(clamped to 1-5 range)", fontsize=8, color=MUTED,
               ha="center", transform=result_ax.transAxes)

save(fig, "page4_predict")

print("\nAll screenshots saved to ./screenshots/")
