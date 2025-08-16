import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	precision_recall_fscore_support,
	roc_auc_score,
	roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Optional deps
try:
	import shap  # type: ignore
except Exception:
	shap = None  # type: ignore

try:
	from xgboost import XGBClassifier  # type: ignore
except Exception:
	XGBClassifier = None  # type: ignore

RANDOM_SEED = 42


def ensure_directory(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def generate_synthetic_data(n: int = 1500) -> Tuple[pd.DataFrame, pd.DataFrame]:
	rng = np.random.default_rng(RANDOM_SEED)

	patient_ids = np.arange(1, n + 1)
	ages = rng.integers(18, 95, size=n)
	sexes = rng.choice(["M", "F"], size=n, p=[0.48, 0.52])
	comorbidities = rng.integers(0, 6, size=n)
	treatments = rng.choice(["none", "standard", "intensive"], size=n, p=[0.2, 0.6, 0.2])
	length_of_stay = np.clip(rng.normal(5, 2, size=n).round().astype(int), 1, None)
	diagnosis = rng.choice(["cardio", "resp", "neuro", "metabolic", "other"], size=n)

	patients_df = pd.DataFrame(
		{
			"patient_id": patient_ids,
			"age": ages,
			"sex": sexes,
			"comorbidities": comorbidities,
			"treatments": treatments,
			"length_of_stay": length_of_stay,
			"diagnosis": diagnosis,
		}
	)

	past_admissions = rng.integers(0, 8, size=n)
	days_since_last_discharge = np.clip((rng.exponential(30, size=n)).astype(int), 0, 365)

	risk = (
		0.02 * (ages - 50)
		+ 0.5 * (comorbidities >= 3).astype(float)
		+ 0.15 * (length_of_stay >= 7).astype(float)
		+ 0.1 * (past_admissions >= 2).astype(float)
		- 0.005 * days_since_last_discharge
		+ rng.normal(0, 0.5, size=n)
	)
	prob = 1 / (1 + np.exp(-risk))
	readmitted_30d = (rng.random(size=n) < prob).astype(int)

	admissions_df = pd.DataFrame(
		{
			"patient_id": patient_ids,
			"past_admissions": past_admissions,
			"days_since_last_discharge": days_since_last_discharge,
			"readmitted_30d": readmitted_30d,
		}
	)

	return patients_df, admissions_df


@st.cache_data(show_spinner=False)
def read_csv_uploaded(file) -> pd.DataFrame:
	return pd.read_csv(file)


def load_and_merge(
	patients_df: pd.DataFrame,
	admissions_df: pd.DataFrame,
	id_col: str,
	label_col: str,
) -> Tuple[pd.DataFrame, List[str]]:
	if id_col not in patients_df.columns or id_col not in admissions_df.columns:
		raise ValueError(f"ID column '{id_col}' is missing from one of the files.")
	if label_col not in admissions_df.columns:
		raise ValueError(f"Target column '{label_col}' is missing from admissions.")

	merged = pd.merge(admissions_df, patients_df, on=id_col, how="inner")

	non_feature_cols = [id_col, label_col]
	feature_cols = [c for c in merged.columns if c not in non_feature_cols]
	return merged, feature_cols


def build_preprocess_pipeline(df: pd.DataFrame, feature_cols: List[str]):
	candidate_df = df[feature_cols].copy()

	numeric_cols = candidate_df.select_dtypes(include=[np.number]).columns.tolist()
	categorical_cols = [c for c in feature_cols if c not in numeric_cols]

	numeric_pipeline = Pipeline(
		steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
	)

	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_pipeline, numeric_cols),
			("cat", categorical_pipeline, categorical_cols),
		],
		remainder="drop",
		sparse_threshold=0.0,
		verbose_feature_names_out=False,
	)

	return preprocessor


def fit_models(
	preprocessor: ColumnTransformer,
	X_train: pd.DataFrame,
	y_train: np.ndarray,
) -> Tuple[Pipeline, Optional[object], List[str]]:
	# Baseline: Logistic Regression as a full pipeline
	logreg_pipeline = Pipeline(
		steps=[
			("preprocess", preprocessor),
			(
				"clf",
				LogisticRegression(
					max_iter=200,
					class_weight="balanced",
					solver="lbfgs",
					random_state=RANDOM_SEED,
				),
			),
		]
	)

	logreg_pipeline.fit(X_train, y_train)

	# XGBoost fitted on transformed features (if available)
	xgb_model = None
	feature_names: List[str] = []
	X_train_transformed = preprocessor.fit_transform(X_train)
	try:
		feature_names = preprocessor.get_feature_names_out().tolist()
	except Exception:
		feature_names = [f"f{i}" for i in range(X_train_transformed.shape[1])]

	if XGBClassifier is not None:
		xgb_model = XGBClassifier(
			n_estimators=300,
			max_depth=4,
			learning_rate=0.05,
			subsample=0.9,
			colsample_bytree=0.9,
			eval_metric="logloss",
			random_state=RANDOM_SEED,
			n_jobs=0,
			reg_lambda=1.0,
			tree_method="hist",
		)
		xgb_model.fit(X_train_transformed, y_train)

	return logreg_pipeline, xgb_model, feature_names


def evaluate_models(
	logreg_pipeline: Pipeline,
	xgb_model: Optional[object],
	preprocessor: ColumnTransformer,
	X_test: pd.DataFrame,
	y_test: np.ndarray,
) -> Tuple[Dict, Optional[Dict], np.ndarray, Optional[np.ndarray]]:
	# LogReg
	logreg_proba = logreg_pipeline.predict_proba(X_test)[:, 1]
	logreg_pred = (logreg_proba >= 0.5).astype(int)

	acc = accuracy_score(y_test, logreg_pred)
	precision, recall, f1, _ = precision_recall_fscore_support(
		y_test, logreg_pred, average="binary", zero_division=0
	)
	roc_auc = roc_auc_score(y_test, logreg_proba)
	logreg_metrics = {
		"accuracy": float(acc),
		"precision": float(precision),
		"recall": float(recall),
		"f1": float(f1),
		"roc_auc": float(roc_auc),
	}

	# XGB (optional)
	xgb_metrics: Optional[Dict] = None
	xgb_proba: Optional[np.ndarray] = None
	if xgb_model is not None:
		X_test_transformed = preprocessor.transform(X_test)
		xgb_proba = xgb_model.predict_proba(X_test_transformed)[:, 1]
		xgb_pred = (xgb_proba >= 0.5).astype(int)

		acc = accuracy_score(y_test, xgb_pred)
		precision, recall, f1, _ = precision_recall_fscore_support(
			y_test, xgb_pred, average="binary", zero_division=0
		)
		roc_auc = roc_auc_score(y_test, xgb_proba)
		xgb_metrics = {
			"accuracy": float(acc),
			"precision": float(precision),
			"recall": float(recall),
			"f1": float(f1),
			"roc_auc": float(roc_auc),
		}

	return logreg_metrics, xgb_metrics, logreg_proba, xgb_proba


def plot_roc(y_true: np.ndarray, proba_dict: Dict[str, np.ndarray]):
	fig, ax = plt.subplots(figsize=(6, 5))
	for name, proba in proba_dict.items():
		fpr, tpr, _ = roc_curve(y_true, proba)
		auc_val = roc_auc_score(y_true, proba)
		ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
	ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
	ax.set_xlabel("False Positive Rate")
	ax.set_ylabel("True Positive Rate")
	ax.set_title("ROC curves")
	ax.legend()
	fig.tight_layout()
	return fig


def shap_summary_figure(model: object, X_transformed: np.ndarray, feature_names: List[str]):
	if shap is None:
		return None
	try:
		explainer = shap.TreeExplainer(model)
		shap_values = explainer.shap_values(X_transformed)
		fig = plt.figure(figsize=(7, 5))
		shap.summary_plot(
			shap_values,
			X_transformed,
			feature_names=feature_names,
			show=False,
		)
		plt.tight_layout()
		return fig
	except Exception as e:
		st.warning(f"SHAP unavailable: {e}")
		return None


def shap_bar_figure(model: object, X_transformed: np.ndarray, feature_names: List[str]):
	if shap is None:
		return None
	try:
		explainer = shap.TreeExplainer(model)
		shap_values = explainer.shap_values(X_transformed)
		fig = plt.figure(figsize=(7, 5))
		shap.summary_plot(
			shap_values,
			X_transformed,
			feature_names=feature_names,
			plot_type="bar",
			show=False,
		)
		plt.tight_layout()
		return fig
	except Exception as e:
		st.warning(f"SHAP unavailable: {e}")
		return None


def main():
	st.set_page_config(page_title="30-day Readmissions", layout="wide")
	st.title("🏥 30-day hospital readmission prediction")

	with st.sidebar:
		st.header("Data")
		patients_file = st.file_uploader("patients.csv", type=["csv"], accept_multiple_files=False)
		admissions_file = st.file_uploader("admissions.csv", type=["csv"], accept_multiple_files=False)
		st.write("Or")
		synth = st.checkbox("Use a synthetic dataset", value=True)

		st.header("Columns")
		id_col = st.text_input("Patient ID column", value="patient_id")
		label_col = st.text_input("Target column (0/1)", value="readmitted_30d")

		st.header("Actions")
		run_btn = st.button("Train models")

	# Load data
	df_patients, df_adm = None, None
	if not synth and patients_file is not None and admissions_file is not None:
		df_patients = read_csv_uploaded(patients_file)
		df_adm = read_csv_uploaded(admissions_file)
	else:
		df_patients, df_adm = generate_synthetic_data(n=1800)

	if run_btn:
		try:
			merged, feature_cols = load_and_merge(df_patients, df_adm, id_col=id_col, label_col=label_col)
		except Exception as e:
			st.error(str(e))
			st.stop()

		st.subheader("Merged data preview")
		st.dataframe(merged.head(20))

		X = merged[feature_cols]
		y = merged[label_col].astype(int)

		preprocessor = build_preprocess_pipeline(merged, feature_cols)

		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
		)

		with st.spinner("Training…"):
			logreg_pipeline, xgb_model, feature_names = fit_models(preprocessor, X_train, y_train)
			logreg_metrics, xgb_metrics, logreg_proba, xgb_proba = evaluate_models(
				logreg_pipeline, xgb_model, preprocessor, X_test, y_test
			)

		st.subheader("Performance (test)")
		rows = [{"model": "LogisticRegression", **logreg_metrics}]
		if xgb_metrics is not None:
			rows.append({"model": "XGBoost", **xgb_metrics})
		else:
			rows.append({"model": "XGBoost", "note": "XGBoost not available"})
		metrics_df = pd.DataFrame(rows)
		st.dataframe(metrics_df.set_index("model", drop=False))

		st.subheader("ROC curves")
		roc_inputs: Dict[str, np.ndarray] = {"LogReg": logreg_proba}
		if xgb_proba is not None:
			roc_inputs["XGBoost"] = xgb_proba
		roc_fig = plot_roc(y_test.values, roc_inputs)
		st.pyplot(roc_fig, clear_figure=True)

		# SHAP for XGB
		st.subheader("Explainability (SHAP, XGBoost)")
		if (shap is None) or (xgb_model is None):
			st.info("SHAP and/or XGBoost are not available in this environment.")
		else:
			X_train_transformed = preprocessor.transform(X_train)
			fig1 = shap_summary_figure(xgb_model, X_train_transformed, feature_names)
			if fig1 is not None:
				st.pyplot(fig1, clear_figure=True)
			fig2 = shap_bar_figure(xgb_model, X_train_transformed, feature_names)
			if fig2 is not None:
				st.pyplot(fig2, clear_figure=True)

		# Single patient prediction
		st.subheader("Per-patient prediction")
		st.caption("Select a test patient to view predictions from the models.")
		with st.form("predict_form"):
			idx_options = list(X_test.index.astype(int)[:100])
			selected_idx = st.selectbox("Patient (index)", idx_options)
			submit_pred = st.form_submit_button("Predict")
		if submit_pred:
			row = X_test.loc[selected_idx:selected_idx]
			logreg_p = float(logreg_pipeline.predict_proba(row)[:, 1][0])
			result = {"LogReg_proba": round(logreg_p, 4)}
			if xgb_model is not None:
				xgb_p = float(xgb_model.predict_proba(preprocessor.transform(row))[:, 1][0])
				result["XGB_proba"] = round(xgb_p, 4)
			st.write(result)

		# Save artifacts (optional)
		artifacts_dir = Path("artifacts")
		ensure_directory(artifacts_dir)
		joblib.dump(logreg_pipeline, artifacts_dir / "logreg_model.joblib")
		if xgb_model is not None:
			xgb_model.save_model(str(artifacts_dir / "xgb_model.json"))
		with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
			json.dump({"logistic_regression": logreg_metrics, "xgboost": xgb_metrics}, f, indent=2, ensure_ascii=False)

		st.success("Models and metrics have been saved to the 'artifacts/' folder.")


if __name__ == "__main__":
	main()
