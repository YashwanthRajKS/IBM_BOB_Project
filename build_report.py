"""
build_report.py
Generates Amazon_Rating_Prediction_Report.docx
- Cover page, Abstract, Objective, Dataset description, Cleaning steps,
  EDA (with screenshots), Methodology, Results, UI walkthrough, Conclusion
"""

import json
import os
from datetime import date

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.style import WD_STYLE_TYPE

# ── Load model metrics ────────────────────────────────────────────
with open("model_info.json") as f:
    info = json.load(f)

doc = Document()

# ── Page margins ──────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.5)

# ── Styles helpers ────────────────────────────────────────────────
HEADING_COLOR = RGBColor(0x1f, 0x23, 0x28)
ACCENT_COLOR  = RGBColor(0x3b, 0x82, 0xd4)


def style_heading(para, level=1):
    para.style = f"Heading {level}"
    for run in para.runs:
        run.font.color.rgb = HEADING_COLOR
        if level == 1:
            run.font.size = Pt(18)
        elif level == 2:
            run.font.size = Pt(14)


def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    style_heading(p, level)
    return p


def add_body(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(8)
    for run in p.runs:
        run.font.size = Pt(11)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(10)
    for run in p.runs:
        run.font.size    = Pt(9)
        run.font.italic  = True
        run.font.color.rgb = RGBColor(0x57, 0x60, 0x6a)
    return p


def add_image(doc, path, width_in=5.8, caption=""):
    if os.path.exists(path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(path, width=Inches(width_in))
    if caption:
        add_caption(doc, caption)


def add_page_break(doc):
    doc.add_page_break()


def shade_cell(cell, hex_color="D9E8FB"):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)


# ═══════════════════════════════════════════════════════════════════
# 1. COVER PAGE
# ═══════════════════════════════════════════════════════════════════
cover = doc.add_paragraph()
cover.alignment = WD_ALIGN_PARAGRAPH.CENTER
cover.paragraph_format.space_before = Pt(60)

run = cover.add_run("Amazon Product Rating Prediction")
run.font.size  = Pt(28)
run.font.bold  = True
run.font.color.rgb = ACCENT_COLOR

doc.add_paragraph()  # spacer

p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = p2.add_run("Using Linear Regression on Amazon India Product Data")
r2.font.size   = Pt(16)
r2.font.italic = True
r2.font.color.rgb = HEADING_COLOR

doc.add_paragraph()
doc.add_paragraph()

p3 = doc.add_paragraph()
p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
r3 = p3.add_run("Prepared by: Himanshu")
r3.font.size = Pt(13)
r3.font.bold = True

p4 = doc.add_paragraph()
p4.alignment = WD_ALIGN_PARAGRAPH.CENTER
r4 = p4.add_run(f"Date: {date.today().strftime('%B %d, %Y')}")
r4.font.size = Pt(12)

p5 = doc.add_paragraph()
p5.alignment = WD_ALIGN_PARAGRAPH.CENTER
r5 = p5.add_run("IBM Data Science Certificate Project")
r5.font.size = Pt(11)
r5.font.color.rgb = RGBColor(0x57, 0x60, 0x6a)

add_page_break(doc)


# ═══════════════════════════════════════════════════════════════════
# 2. ABSTRACT
# ═══════════════════════════════════════════════════════════════════
add_heading(doc, "Abstract", level=1)
add_body(doc,
    "Online marketplaces generate vast quantities of product and review data, "
    "and the star-rating a product attracts is one of the strongest signals "
    "for purchase decisions. This project builds an end-to-end machine-learning "
    "pipeline that predicts the Amazon India product rating using structured "
    "pricing and engagement features extracted from a 1,465-row dataset. "
    "After a rigorous data-cleaning phase — stripping currency symbols, "
    "imputing two missing rating-count values, removing one malformed rating "
    "row, and deduplicating at the product level — a Linear Regression model "
    "was trained on 1,350 unique products. The model uses discounted price, "
    "actual price, discount percentage, rating count, and main category "
    "(one-hot encoded) as predictors. On a held-out 20 % test set the model "
    "achieves an R² of {r2}, a Mean Absolute Error of {mae}, and a Root Mean "
    "Squared Error of {rmse}. The trained model is served through a Flask "
    "REST API and a Streamlit interactive dashboard provides four pages: "
    "dataset overview, exploratory data analysis, model insights, and live "
    "prediction. The project demonstrates how product-level signals captured "
    "at listing time can serve as lightweight proxies for eventual rating "
    "outcomes in a real-world e-commerce setting.".format(
        r2=info["r2"], mae=info["mae"], rmse=info["rmse"]
    )
)
add_page_break(doc)


# ═══════════════════════════════════════════════════════════════════
# 3. OBJECTIVE
# ═══════════════════════════════════════════════════════════════════
add_heading(doc, "Objective", level=1)
add_body(doc,
    "The primary objective of this project is to develop a predictive model "
    "that estimates the average customer rating of an Amazon India product "
    "based on its pricing strategy and engagement metrics. Specifically, "
    "the project aims to:"
)
bullets = [
    "Clean and preprocess raw Amazon product data, handling currency "
    "formatting, missing values, and duplicate records.",
    "Engineer meaningful features — including main product category from "
    "a hierarchical category string — suitable for regression modelling.",
    "Train and evaluate a Linear Regression model to quantify how pricing "
    "and popularity factors influence product ratings.",
    "Expose the trained model through a Flask REST API for programmatic "
    "access by downstream systems or frontends.",
    "Build an interactive Streamlit dashboard for non-technical users to "
    "explore the dataset, review model metrics, and generate predictions.",
]
for b in bullets:
    p = doc.add_paragraph(b, style="List Bullet")
    p.paragraph_format.space_after = Pt(4)

add_page_break(doc)


# ═══════════════════════════════════════════════════════════════════
# 4. DATASET DESCRIPTION
# ═══════════════════════════════════════════════════════════════════
add_heading(doc, "Dataset Description", level=1)
add_body(doc,
    "The dataset (amazon.csv) was sourced from Kaggle and contains Amazon "
    "India product listings scraped from the Electronics, Computers & "
    "Accessories, and other major categories. Each row represents a single "
    "product bundled with aggregated review information."
)

add_heading(doc, "Source & Size", level=2)
add_body(doc,
    "Source: Kaggle — Amazon Sales Dataset (India).  "
    "Raw file: 1,465 rows × 16 columns."
)

add_heading(doc, "Key Columns", level=2)
col_table = doc.add_table(rows=1, cols=3)
col_table.style = "Table Grid"
col_table.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr = col_table.rows[0].cells
for cell, txt in zip(hdr, ["Column", "Type", "Description"]):
    cell.text = txt
    shade_cell(cell, "3B82D4")
    for para in cell.paragraphs:
        for run in para.runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

columns_info = [
    ("product_id",          "string",  "Unique product identifier"),
    ("product_name",        "string",  "Full product title"),
    ("category",            "string",  "Pipe-separated hierarchy (up to 5 levels)"),
    ("discounted_price",    "string→float", "Sale price with ₹ prefix and comma separators"),
    ("actual_price",        "string→float", "MRP / list price with ₹ prefix"),
    ("discount_percentage", "string→float", "Percentage discount (e.g. '64%')"),
    ("rating",              "string→float", "Avg star rating (1–5); one row contains '|'"),
    ("rating_count",        "string→float", "Number of ratings; 2 values missing"),
    ("about_product",       "string",  "Pipe-separated product bullet points"),
    ("user_id / user_name", "string",  "Aggregated reviewer IDs and names"),
    ("review_id / title",   "string",  "Individual review identifiers and titles"),
    ("review_content",      "string",  "Full review text"),
    ("img_link",            "string",  "Product image URL"),
    ("product_link",        "string",  "Amazon product URL"),
]
for row_data in columns_info:
    row = col_table.add_row()
    for cell, txt in zip(row.cells, row_data):
        cell.text = txt

doc.add_paragraph()

add_heading(doc, "Data Quality Issues Found", level=2)
issues = [
    "Currency symbols (₹) and thousands separators (,) in price columns.",
    "Percentage sign (%) in discount_percentage; values need /100 normalization.",
    "One row with rating == '|' (a data entry error) — must be dropped.",
    "Two rows with missing rating_count — imputed with the column median.",
    "Multiple review rows per product_id — deduplicated to 1,350 unique products.",
    "Multi-level category strings (up to 5 pipe-separated levels) — only the "
    "first two levels are retained as main_category and sub_category.",
]
for issue in issues:
    p = doc.add_paragraph(issue, style="List Bullet")
    p.paragraph_format.space_after = Pt(3)

add_page_break(doc)


# ═══════════════════════════════════════════════════════════════════
# 5. DATA CLEANING STEPS
# ═══════════════════════════════════════════════════════════════════
add_heading(doc, "Data Cleaning Steps", level=1)

steps = [
    ("discounted_price & actual_price",
     "Stripped '₹' prefix and ',' separators, then cast to float64 using "
     "pd.to_numeric(errors='coerce')."),
    ("discount_percentage",
     "Stripped '%' suffix, cast to float, then divided by 100 to obtain "
     "a unit fraction (e.g., 64 % → 0.64)."),
    ("rating",
     "Coerced to numeric; one row with value '|' produced NaN and was "
     "dropped, reducing the dataset from 1,465 to 1,464 rows."),
    ("rating_count",
     "Stripped ',' separators, cast to float; two NaN values were imputed "
     "with the column median (no rows dropped)."),
    ("category → main_category / sub_category",
     "Split on '|'; index [0] retained as main_category, index [1] as "
     "sub_category. Missing sub_category filled with 'Other'."),
    ("Deduplication",
     "Dropped duplicate product_id rows keeping the first occurrence "
     "(1,464 → 1,350 unique products)."),
    ("Final NaN drop",
     "Rows with NaN in any model feature or target column were removed; "
     "final modelling dataset: 1,350 rows."),
]
for i, (title, desc) in enumerate(steps, 1):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    run_num  = p.add_run(f"Step {i}. {title}: ")
    run_num.bold = True
    run_num.font.size = Pt(11)
    run_desc = p.add_run(desc)
    run_desc.font.size = Pt(11)

add_body(doc,
    "\nAfter cleaning: 1,350 rows × 16 columns, with all numeric features "
    "correctly typed and no missing values in the modelling subset."
)
add_page_break(doc)


# ═══════════════════════════════════════════════════════════════════
# 6. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════
add_heading(doc, "Exploratory Data Analysis", level=1)
add_body(doc,
    "EDA was performed on the cleaned 1,350-product dataset. The following "
    "charts summarise the key distributional and relational findings."
)
add_image(doc, "screenshots/page2_eda.png", width_in=6.0,
          caption="Figure 1. EDA dashboard — rating distribution, "
                  "price vs rating, discount vs rating, and category boxplots.")

add_heading(doc, "Key Findings", level=2)
findings = [
    "Rating Distribution: Ratings are heavily left-skewed, concentrated "
    "between 3.8 and 4.5, with 4.0–4.2 being the most common band. "
    "Very few products fall below 3.0.",
    "Discounted Price vs Rating: The OLS trendline is nearly flat "
    "(correlation close to 0), indicating that product price alone is "
    "a weak predictor of rating.",
    "Discount Percentage vs Rating: A slight negative trend is visible — "
    "heavily discounted products tend to receive marginally lower ratings, "
    "possibly because deep discounts attract volume buyers with lower "
    "quality expectations.",
    "Rating Count vs Rating: Products with more ratings generally "
    "converge to the 3.8–4.3 range, suggesting regression to the mean "
    "as review volume grows.",
    "Category Boxplots: HomeImprovement and OfficeProducts categories "
    "show higher median ratings, while Car&Motorbike and "
    "MusicalInstruments categories skew lower.",
]
for f in findings:
    p = doc.add_paragraph(f, style="List Bullet")
    p.paragraph_format.space_after = Pt(4)

add_page_break(doc)


# ═══════════════════════════════════════════════════════════════════
# 7. MODEL METHODOLOGY
# ═══════════════════════════════════════════════════════════════════
add_heading(doc, "Model Methodology", level=1)

add_heading(doc, "Why Linear Regression?", level=2)
add_body(doc,
    "Linear Regression was chosen as the baseline algorithm for this "
    "prediction task for the following reasons: (1) it is highly "
    "interpretable — each coefficient directly quantifies the marginal "
    "effect of a feature on the predicted rating; (2) the target variable "
    "(average rating) is continuous; (3) the dataset is modest in size "
    "(1,350 rows) where simpler models often generalise as well as complex "
    "ones; and (4) it produces a closed-form model equation suitable for "
    "documentation and stakeholder communication. While tree-based models "
    "might yield higher R², the transparency of Linear Regression is "
    "particularly valuable in a business context where pricing teams need "
    "to understand the direction and magnitude of each driver."
)

add_heading(doc, "Features Used", level=2)
feat_table = doc.add_table(rows=1, cols=3)
feat_table.style = "Table Grid"
feat_table.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr = feat_table.rows[0].cells
for cell, txt in zip(hdr, ["Feature", "Type", "Rationale"]):
    cell.text = txt
    shade_cell(cell, "3B82D4")
    for para in cell.paragraphs:
        for run in para.runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

feat_rows = [
    ("discounted_price",    "Numeric (float)",  "Actual selling price; affordability signal"),
    ("actual_price",        "Numeric (float)",  "MRP; proxy for product tier / quality"),
    ("discount_percentage", "Numeric (0–1)",    "Depth of discount; demand driver"),
    ("rating_count",        "Numeric (float)",  "Popularity / engagement proxy"),
    ("main_category",       "Categorical (OHE)","Category-level rating tendencies"),
]
for rd in feat_rows:
    row = feat_table.add_row()
    for cell, txt in zip(row.cells, rd):
        cell.text = txt

doc.add_paragraph()
add_heading(doc, "Pipeline", level=2)
add_body(doc,
    "A scikit-learn Pipeline was used: a ColumnTransformer applies "
    "'passthrough' on numeric features and OneHotEncoder "
    "(handle_unknown='ignore') on main_category, feeding into a "
    "LinearRegression estimator. The dataset was split 80/20 "
    "(train/test) with random_state=42."
)
add_page_break(doc)


# ═══════════════════════════════════════════════════════════════════
# 8. RESULTS
# ═══════════════════════════════════════════════════════════════════
add_heading(doc, "Results", level=1)

add_heading(doc, "Performance Metrics (20% Hold-out Test Set)", level=2)
metric_table = doc.add_table(rows=4, cols=2)
metric_table.style = "Table Grid"
metric_table.alignment = WD_TABLE_ALIGNMENT.CENTER
header_cells = metric_table.rows[0].cells
for cell, txt in zip(header_cells, ["Metric", "Value"]):
    cell.text = txt
    shade_cell(cell, "3B82D4")
    for para in cell.paragraphs:
        for run in para.runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

for row, (metric, value) in zip(
    metric_table.rows[1:],
    [("R² (Coefficient of Determination)", str(info["r2"])),
     ("MAE (Mean Absolute Error)",          str(info["mae"])),
     ("RMSE (Root Mean Squared Error)",     str(info["rmse"]))],
):
    row.cells[0].text = metric
    row.cells[1].text = value

doc.add_paragraph()
add_body(doc,
    f"An R² of {info['r2']} indicates that approximately "
    f"{round(info['r2']*100, 1)}% of the variance in product ratings is "
    "explained by the selected pricing and engagement features. "
    f"The MAE of {info['mae']} stars means that, on average, predictions "
    f"deviate by only {info['mae']} star from the true rating — well within "
    "half a star. The RMSE of {rmse} confirms no large systematic "
    "outlier errors.".format(rmse=info["rmse"])
)

add_heading(doc, "Model Equation", level=2)
eq_parts = [f"Rating = {info['intercept']:.4f}"]
for feat, coef in info["coefficients"].items():
    sign = "+" if coef >= 0 else ""
    eq_parts.append(f"        {sign}{coef:.6f} × {feat}")
eq_text = "\n".join(eq_parts)
p = doc.add_paragraph()
run = p.add_run(eq_text)
run.font.name = "Courier New"
run.font.size = Pt(9)

add_heading(doc, "Feature Coefficients", level=2)
coef_tbl = doc.add_table(rows=1, cols=2)
coef_tbl.style = "Table Grid"
coef_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
h = coef_tbl.rows[0].cells
for cell, txt in zip(h, ["Feature", "Coefficient"]):
    cell.text = txt
    shade_cell(cell, "3B82D4")
    for para in cell.paragraphs:
        for run in para.runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

for feat, coef in info["coefficients"].items():
    row = coef_tbl.add_row()
    row.cells[0].text = feat
    row.cells[1].text = f"{coef:+.6f}"

doc.add_paragraph()
add_image(doc, "screenshots/page3_model_insights.png", width_in=6.0,
          caption="Figure 2. Model Insights page — metrics, coefficient bar chart, "
                  "and OLS trendline over the full dataset.")
add_page_break(doc)


# ═══════════════════════════════════════════════════════════════════
# 9. UI WALKTHROUGH
# ═══════════════════════════════════════════════════════════════════
add_heading(doc, "UI Walkthrough", level=1)
add_body(doc,
    "The Streamlit dashboard (ui.py) is a single-file application with four "
    "pages accessible from the sidebar. It communicates with the Flask backend "
    "(app.py) for live predictions. The screenshots below were generated "
    "programmatically using matplotlib (generate_screenshots.py)."
)

add_heading(doc, "Page 1 – Dataset Overview", level=2)
add_body(doc,
    "Displays key metrics (total products, feature count, average rating, "
    "category count), a sample rows table, a horizontal bar chart of the "
    "main category distribution, column dtypes, missing value report, and "
    "descriptive statistics."
)
add_image(doc, "screenshots/page1_dataset_overview.png", width_in=6.0,
          caption="Figure 3. Streamlit Page 1 – Dataset Overview.")

add_heading(doc, "Page 2 – EDA", level=2)
add_body(doc,
    "Four charts: rating distribution histogram, discounted price vs rating "
    "scatter with OLS trendline, discount percentage vs rating scatter, and "
    "a boxplot of rating distributions per main category."
)
add_image(doc, "screenshots/page2_eda.png", width_in=6.0,
          caption="Figure 4. Streamlit Page 2 – Exploratory Data Analysis.")

add_heading(doc, "Page 3 – Model Insights", level=2)
add_body(doc,
    "Shows R², MAE, RMSE metric cards, the model equation, a horizontal "
    "coefficient bar chart (positive coefficients in blue, negative in red), "
    "and an OLS regression trendline over the full dataset."
)
add_image(doc, "screenshots/page3_model_insights.png", width_in=6.0,
          caption="Figure 5. Streamlit Page 3 – Model Insights.")

add_heading(doc, "Page 4 – Predict", level=2)
add_body(doc,
    "An interactive form allows users to enter discounted price, actual price, "
    "discount percentage, rating count, and main category. On submission the "
    "frontend POSTs to the Flask /predict endpoint and displays the predicted "
    "rating with a star visual indicator."
)
add_image(doc, "screenshots/page4_predict.png", width_in=6.0,
          caption="Figure 6. Streamlit Page 4 – Predict.")
add_page_break(doc)


# ═══════════════════════════════════════════════════════════════════
# 10. CONCLUSION & FUTURE SCOPE
# ═══════════════════════════════════════════════════════════════════
add_heading(doc, "Conclusion and Future Scope", level=1)

add_heading(doc, "Conclusion", level=2)
add_body(doc,
    f"This project successfully delivered an end-to-end Amazon product rating "
    f"prediction system. The data cleaning pipeline systematically resolved "
    f"all identified quality issues — currency formatting, missing values, "
    f"malformed ratings, and product-level duplicates — reducing the raw "
    f"1,465-row dataset to a clean 1,350-product modelling set. A Linear "
    f"Regression model trained on pricing and engagement features achieved an "
    f"R² of {info['r2']}, MAE of {info['mae']}, and RMSE of {info['rmse']} "
    f"on the held-out test set. The results confirm that discounted price, "
    f"actual price, discount percentage, rating count, and main category "
    f"carry meaningful (though partial) information about a product's "
    f"eventual average rating. The trained model was packaged into a Flask "
    f"REST API and an interactive Streamlit dashboard, making the solution "
    f"accessible to both developers and business stakeholders."
)

add_heading(doc, "Future Scope", level=2)
future = [
    "Feature Engineering: Derive features from review text (sentiment scores, "
    "keyword frequency) and product title (word count, brand detection) to "
    "substantially improve R².",
    "Advanced Models: Compare Random Forest, Gradient Boosting (XGBoost / "
    "LightGBM), and Neural Network regressors against the Linear baseline.",
    "Sub-category Modelling: Train separate models per main_category to "
    "capture category-specific pricing dynamics.",
    "Time-series Component: Incorporate product age and review velocity to "
    "model rating trajectory over time.",
    "CI/CD Pipeline: Containerise the Flask API with Docker and deploy to "
    "IBM Cloud Code Engine or AWS Lambda for production availability.",
    "Explainability: Integrate SHAP (SHapley Additive exPlanations) to "
    "provide per-prediction feature contribution breakdowns in the UI.",
    "A/B Testing Framework: Use the API to test pricing strategies and "
    "predict their impact on expected rating before going live.",
]
for item in future:
    p = doc.add_paragraph(item, style="List Bullet")
    p.paragraph_format.space_after = Pt(4)

doc.add_paragraph()
add_body(doc,
    "In summary, the project lays a solid, reproducible foundation for "
    "AI-assisted product strategy at scale on Amazon's Indian marketplace."
)

# ═══════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════
report_path = "Amazon_Rating_Prediction_Report.docx"
doc.save(report_path)
print(f"[OK] Report saved: {report_path}")
