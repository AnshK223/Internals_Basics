"""
Task 1 — Experiment Tracking & Model Comparison
Run: python src/train.py
Outputs: results/step1_s1.json, models/best_model.pkl, models/run_ids.json
"""
import os, json, warnings
warnings.filterwarnings("ignore")

import joblib, shutil
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE, "data", "training_data.csv")
MODELS_DIR  = os.path.join(BASE, "models")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

EXPERIMENT_NAME = "cropsense-irrigation-hours"
FEATURES = ["soil_moisture_pct", "crop_type_index", "field_size_hectares", "temperature_c"]
TARGET   = "irrigation_hours"

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def metrics(y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mp   = mape(y_true, y_pred)
    return mae, rmse, r2, mp

df = pd.read_csv(DATA_PATH)
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_tracking_uri("file:///" + os.path.join(BASE, "mlruns").replace("\\", "/"))
mlflow.set_experiment(EXPERIMENT_NAME)

models_cfg = [
    ("LinearRegression", LinearRegression(),
     {"fit_intercept": True}),
    ("GradientBoosting", GradientBoostingRegressor(random_state=42),
     {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42}),
]

results = []
for name, model, params in models_cfg:
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("experiment_type", "baseline_comparison")
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae, rmse, r2, mp = metrics(y_test, preds)
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2, "mape": mp})
        mlflow.sklearn.log_model(model, artifact_path="model")
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))
        results.append({"name": name, "mae": round(mae,4), "rmse": round(rmse,4),
                        "r2": round(r2,4), "mape": round(mp,4), "run_id": run.info.run_id})
        print(f"{name}: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  MAPE={mp:.2f}%")

best = min(results, key=lambda x: x["mae"])
print(f"\nBest: {best['name']}  MAE={best['mae']}")

shutil.copy(os.path.join(MODELS_DIR, f"{best['name']}.pkl"),
            os.path.join(MODELS_DIR, "best_model.pkl"))

json.dump({
    "experiment_name": EXPERIMENT_NAME,
    "models": [{"name": r["name"], "mae": r["mae"], "rmse": r["rmse"],
                "r2": r["r2"], "mape": r["mape"]} for r in results],
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": best["mae"],
}, open(os.path.join(RESULTS_DIR, "step1_s1.json"), "w"), indent=2)

json.dump({"run_ids": {r["name"]: r["run_id"] for r in results},
           "best_model": best["name"], "best_mae": best["mae"],
           "best_run_id": best["run_id"]},
          open(os.path.join(MODELS_DIR, "run_ids.json"), "w"), indent=2)

print("Saved results/step1_s1.json")
