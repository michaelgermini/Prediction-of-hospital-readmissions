## 30-day hospital readmission prediction — Streamlit

Live demo: https://prediction-of-hospital-readmissions.streamlit.app/

### What this app does
An end-to-end, lightweight ML app to predict 30-day hospital readmissions. It includes:
- Data ingestion from two CSVs (`patients.csv`, `admissions.csv`) or a synthetic generator
- Preprocessing (imputation, scaling, one-hot encoding)
- Models: Logistic Regression (baseline) and XGBoost (optional if available)
- Evaluation on a holdout set: accuracy, precision, recall, F1, ROC-AUC; ROC curves
- Explainability: SHAP plots for XGBoost (optional, if SHAP is installed)
- Per-patient predictions in the UI

This project is designed for clarity and portability (single-file app), making it easy to deploy on Streamlit Community Cloud.

### Data schema
- `patients.csv` (examples):
  - `patient_id`, `age`, `sex`, `comorbidities`, `treatments`, `length_of_stay`, `diagnosis`
- `admissions.csv` (examples):
  - `patient_id`, `past_admissions`, `days_since_last_discharge`, `readmitted_30d`

If no files are provided, the app can generate a synthetic dataset with similar structure for demo purposes.

### Installation (local)
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

### Run locally
```bash
streamlit run streamlit_app.py
```
Main file path: `streamlit_app.py`

### Usage
- In the sidebar, upload `patients.csv` and `admissions.csv` or leave the "Use a synthetic dataset" checkbox enabled
- Set the patient ID and target column names if they differ from defaults
- Click "Train models" to run preprocessing, training and evaluation
- Inspect metrics, ROC curves, and (optionally) SHAP plots for XGBoost
- Try per-patient prediction on the test split

### Deployment (Streamlit Community Cloud)
1) Push this repo to GitHub (already set up in this project)
2) On Streamlit Cloud, choose:
   - Repository: `michaelgermini/Prediction-of-hospital-readmissions`
   - Branch: `master`
   - Main file path: `streamlit_app.py`
3) Deploy

### Notes on dependencies
- SHAP is optional and not installed by default to avoid Windows build-tool requirements. If you want SHAP locally:
  - Install Microsoft C++ Build Tools (C++ workload)
  - Then run: `pip install shap`
- XGBoost is optional in the app; if unavailable, the app will still work with Logistic Regression

### Repository structure
- `streamlit_app.py` — main Streamlit app
- `requirements.txt` — Python dependencies (without SHAP by default)
- `README.md` — this documentation
- `src/` — reserved for future extensions
- `.gitignore` — excludes virtualenv, artifacts, data, etc.
- `artifacts/` — created at runtime for models and metrics (ignored by Git)

### License
This project is provided as-is for educational and demo purposes.
