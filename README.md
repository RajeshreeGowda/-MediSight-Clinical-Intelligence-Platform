# 🏥 MediSight — Clinical Intelligence Platform

> **A full-stack Data & AI portfolio project** built on the MIMIC-III Clinical Database Demo, spanning SQL analytics, Excel modelling, Machine Learning, Power BI dashboarding, and a GenAI-powered Streamlit application.

**Author:** Rajeshwari G D | **Dataset:** MIMIC-III Clinical Database Demo | **Goal:** Predict 30-day hospital readmission risk and surface clinical insights through AI

---

## 📌 Project Overview

MediSight is a **5-layer end-to-end clinical analytics platform** that takes raw hospital data through every stage of a modern data pipeline — from raw SQL to a live AI chatbot:

| Layer | Tool | What it does |
|-------|------|--------------|
| 1 · SQL | MySQL | DDL, 5 analytics queries (CTEs, window functions, CASE) |
| 2 · Excel | Excel | Master dataset, clinical risk score computation, EDA |
| 3 · ML | Python / XGBoost | Readmission prediction model, SMOTE, SHAP explainability |
| 4 · Power BI | Power BI | Interactive dashboard — KPIs, slicers, risk monitor |
| 5 · GenAI | LangChain + Groq + Streamlit | RAG chatbot, risk predictor, discharge report generator |

---

## 🗂️ Repository Structure

```
MediSight/
│
├── 01_SQL/
│   └── MediSight_SQL.sql           # DDL + 5 analytics queries
│
├── 02_Excel/
│   ├── medisight_master.xlsx       # Source dataset with clinical risk scoring
│   ├── MediSight_Master_Dataset.xlsx  # Full master dataset for ML
│   └── risk_scores.xlsx            # ML model output scores
│
├── 03_ML/
│   └── readmission_model_FINAL.ipynb  # 14-cell Jupyter notebook
│
├── 04_PowerBI/
│   └── MediSight_BIDashboard.pbix  # Power BI dashboard file
│
├── 05_GenAI/
│   ├── app.py                      # Streamlit app (4 tabs)
│   ├── rag_pipeline.py             # ChromaDB vector indexing pipeline
│   ├── create_notes.py             # Generates 10 synthetic discharge notes
│   ├── discharge_notes/            # Auto-created by create_notes.py
│   ├── chroma_db/                  # Auto-created by rag_pipeline.py
│   └── .env                        # GROQ_API_KEY (not committed)
│
├── .gitignore
└── README.md
```

---

## 🔬 Layer 1 — SQL Analytics

**File:** `01_SQL/MediSight_SQL.sql`

Five production-grade queries on 4 tables (`patients`, `admissions`, `diagnose1`, `prescriptions`):

| Query | Technique | Business Question |
|-------|-----------|-------------------|
| Q1 | GROUP BY + AVG | Readmission rate by insurance type |
| Q2 | CTE + CASE | Length-of-stay risk tier segmentation |
| Q3 | JOIN + HAVING | Top 10 ICD-9 diagnoses by readmission count |
| Q4 | Window Functions (LAG, SUM OVER) | Monthly admissions trend + MoM growth |
| Q5 | Multi-CTE + LEFT JOIN | Chronic care cohort with drug burden analysis |

```sql
-- Example: Window function for monthly trend
SELECT month, admissions,
  SUM(admissions) OVER (ORDER BY month) AS running_total,
  LAG(admissions, 1) OVER (ORDER BY month) AS prev_month
FROM monthly ORDER BY month;
```

**To run:** Import using MySQL Workbench against the MIMIC-III Demo database.

---

## 📊 Layer 2 — Excel Modelling

**Files:** `02_Excel/medisight_master.xlsx`, `MediSight_Master_Dataset.xlsx`, `risk_scores.xlsx`

- Master dataset exported from SQL (patients, admissions, diagnoses, drugs)
- Clinical risk score engineered from 6 features: age, LOS, insurance, discharge location, diagnosis count, drug burden
- Target variable thresholded at 70th percentile → ~30% readmission rate (matches clinical benchmarks)
- Risk output from ML model stored in `risk_scores.xlsx`

---

## 🤖 Layer 3 — Machine Learning

**File:** `03_ML/readmission_model_FINAL.ipynb`

**14-cell Jupyter notebook** covering the full ML pipeline:

1. Library installation & imports
2. Data loading from Excel (MIMIC-III master dataset)
3. Target column fix — clinical risk score if raw target is all zeros
4. EDA — class balance, feature distributions, correlation heatmap
5. Preprocessing — label encoding, feature engineering (`high_complexity`, `elderly`, `long_stay`)
6. SMOTE oversampling (adaptive `k_neighbors` for small datasets)
7. Model benchmarking — Logistic Regression, Decision Tree, Random Forest, XGBoost
8. XGBoost final model with hyperparameter tuning
9. SHAP explainability plots
10. Artifact export — `readmission_model.pkl`, `risk_scores.csv`

**Key results:**
- Model: XGBoost | Target metric: AUC ≥ 0.85
- Features: age, gender, LOS, insurance, discharge location, num_diagnoses, num_drugs + 3 engineered features
- SHAP top drivers: LOS days, age, discharge destination, diagnosis complexity

**To run:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap imbalanced-learn openpyxl joblib
jupyter notebook 03_ML/readmission_model_FINAL.ipynb
```

> Update the file path in Cell 3 to point to your local `MediSight_Master_Dataset.xlsx`.

---

## 📈 Layer 4 — Power BI Dashboard

**File:** `04_PowerBI/MediSight_BIDashboard.pbix`

Interactive dashboard with 3 pages:

- **Overview** — Total patients, readmission rate, avg LOS KPIs with insurance and age group slicers
- **Risk Monitor** — `% High Risk`, `Avg Risk Score` DAX measures, patient risk distribution
- **Trends** — Monthly admissions trend, diagnosis breakdown

**DAX Measures used:**
```dax
Avg LOS Days = AVERAGE(medisight_master[los_days])
% High Risk = DIVIDE(COUNTROWS(FILTER(..., [risk_tier]="High Risk")), COUNTROWS(...))
Avg Risk Score = AVERAGE(risk_scores[risk_score])
```

> Requires Power BI Desktop. Connect data source to the Excel files in `02_Excel/`.

---

## 🧠 Layer 5 — GenAI Streamlit App

**Files:** `05_GenAI/app.py`, `rag_pipeline.py`, `create_notes.py`

A 4-tab Streamlit application powered by **LangChain + Groq LLM + ChromaDB RAG**:

| Tab | Feature | Tech |
|-----|---------|------|
| 📋 Clinical Q&A | Ask questions about discharge summaries | RAG · ChromaDB · Groq LLaMA-3.3-70b |
| 🎯 Risk Predictor | Real-time 30-day readmission risk | XGBoost model · joblib |
| 📄 Discharge Report | AI-generated clinical discharge summary + PDF download | Groq LLM · fpdf2 |
| 📊 Analytics Dashboard | Live KPIs + Plotly charts from ML output | Plotly Express |

### Setup & Run

```bash
# 1. Install dependencies
pip install streamlit langchain langchain-groq langchain-community \
            langchain-text-splitters chromadb sentence-transformers \
            joblib plotly fpdf2 python-dotenv

# 2. Set your Groq API key
echo "GROQ_API_KEY=gsk_your_key_here" > 05_GenAI/.env

# 3. Generate discharge notes + build vector DB
cd 05_GenAI
python create_notes.py
python rag_pipeline.py

# 4. Launch the app
streamlit run app.py
```

App runs at `http://localhost:8501`

### Architecture

```
Discharge Notes (.txt)
        ↓
create_notes.py → discharge_notes/
        ↓
rag_pipeline.py → ChromaDB (all-MiniLM-L6-v2 embeddings)
        ↓
app.py (Streamlit)
  ├── ChromaDB vector search → Groq LLaMA → Clinical Q&A
  ├── XGBoost .pkl → Risk score + tier
  ├── Groq LLaMA → Discharge report → PDF
  └── risk_scores.csv → Plotly dashboard
```

---

## ⚙️ Tech Stack

| Category | Tools |
|----------|-------|
| Database | MySQL, MySQL Workbench |
| Data | Python (pandas, numpy, openpyxl) |
| ML | scikit-learn, XGBoost, SHAP, imbalanced-learn |
| GenAI | LangChain, Groq (LLaMA-3.3-70b-versatile), ChromaDB, HuggingFace Embeddings |
| App | Streamlit, Plotly Express, fpdf2 |
| BI | Microsoft Power BI Desktop |
| Dataset | MIMIC-III Clinical Database Demo (PhysioNet) |

---

## 📋 Prerequisites

- Python 3.9+
- MySQL 8.0+ (for SQL layer)
- Power BI Desktop (for `.pbix` file)
- [Groq API key](https://console.groq.com) (free tier available)
- MIMIC-III Demo access from [PhysioNet](https://physionet.org/content/mimiciii-demo/1.4/)

---

## 👩‍💻 About

**Rajeshwari G D**  
Data & AI Portfolio Project | Built with MIMIC-III Clinical Database Demo  
Skills demonstrated: SQL · Excel · Machine Learning · Power BI · LangChain · RAG · Streamlit

---

## ⚠️ Disclaimer

This project uses the **MIMIC-III Demo dataset** for educational and portfolio purposes only. It is not intended for clinical use. All patient data is de-identified per PhysioNet's data use agreement.
