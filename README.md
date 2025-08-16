## 30-day hospital readmission prediction — Streamlit

### Installation

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run streamlit_app.py
```

### Data
- Upload `patients.csv` and `admissions.csv` via the UI, or generate a synthetic dataset from the app.
- Default expected columns:
  - `patients.csv`: `patient_id`, `age`, `sex`, `comorbidities`, `treatments`, `length_of_stay`, `diagnosis`
  - `admissions.csv`: `patient_id`, `past_admissions`, `days_since_last_discharge`, `readmitted_30d`

### Features
- Preprocessing: imputation, scaling, One-Hot encoding
- Models: Logistic Regression (baseline) and XGBoost (optional if available)
- Evaluation: accuracy, precision, recall, F1, ROC-AUC; ROC curves
- Explainability: SHAP plots for XGBoost (optional, requires SHAP install)
- Per-patient prediction (select a test patient)

### Notes
- SHAP is optional and not installed by default to avoid Windows/Build Tools issues. If you want SHAP:
  1) Install Microsoft C++ Build Tools (C++ build tools workload).
  2) `pip install shap`
