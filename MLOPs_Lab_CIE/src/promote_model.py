"""
Task 4 — Model Promotion
Run AFTER register_model.py.
- Assigns alias "production" to version 1
- Trains a challenger with random_state=99
- Registers as version 2
- Compares MAE; moves alias if challenger wins
Outputs: results/step4_s7.json
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")

import joblib, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE, "models")
RESULTS_DIR = os.path.join(BASE, "results")
DATA_PATH   = os.path.join(BASE, "data", "training_data.csv")
REGISTERED_NAME = "cropsense-irrigation-hours-predictor"
FEATURES = ["soil_moisture_pct", "crop_type_index", "field_size_hectares", "temperature_c"]
TARGET   = "irrigation_hours"

mlflow.set_tracking_uri("file:///" + os.path.join(BASE, "mlruns").replace("\\", "/"))
mlflow.set_experiment("cropsense-irrigation-hours")
client = MlflowClient()

meta = json.load(open(os.path.join(MODELS_DIR, "run_ids.json")))
champion_mae = meta["best_mae"]

# ── Assign "production" alias to version 1 ───────────────────────────────────
client.set_registered_model_alias(REGISTERED_NAME, "production", "1")
print("Alias 'production' set to version 1")

# ── Train challenger with random_state=99 ────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

challenger_model = GradientBoostingRegressor(random_state=99, n_estimators=100,
                                             learning_rate=0.1, max_depth=3)
with mlflow.start_run(run_name="GradientBoosting_challenger") as run:
    mlflow.set_tag("experiment_type", "baseline_comparison")
    mlflow.log_params({"n_estimators": 100, "learning_rate": 0.1,
                       "max_depth": 3, "random_state": 99})
    challenger_model.fit(X_train, y_train)
    preds = challenger_model.predict(X_test)
    challenger_mae = float(mean_absolute_error(y_test, preds))
    mlflow.log_metric("mae", challenger_mae)
    mlflow.sklearn.log_model(challenger_model, artifact_path="model")
    challenger_run_id = run.info.run_id
    artifact_uri = run.info.artifact_uri + "/model"

print(f"Challenger MAE={challenger_mae:.4f}  Champion MAE={champion_mae:.4f}")

# Register challenger as version 2
mv2 = client.create_model_version(
    name=REGISTERED_NAME, source=artifact_uri, run_id=challenger_run_id)
for _ in range(20):
    mv2 = client.get_model_version(REGISTERED_NAME, mv2.version)
    if mv2.status == "READY":
        break
    time.sleep(1)

challenger_version = int(mv2.version)

# ── Promote if challenger is better ──────────────────────────────────────────
if challenger_mae < champion_mae:
    client.set_registered_model_alias(REGISTERED_NAME, "production", str(challenger_version))
    action          = "promoted"
    champion_version = challenger_version
    print(f"Promoted: alias 'production' → version {challenger_version}")
else:
    action          = "kept"
    champion_version = 1
    print(f"Kept: alias 'production' stays on version 1")

result = {
    "registered_model_name": REGISTERED_NAME,
    "alias_name":            "production",
    "champion_version":      champion_version,
    "challenger_version":    challenger_version,
    "action":                action,
}
json.dump(result, open(os.path.join(RESULTS_DIR, "step4_s7.json"), "w"), indent=2)
print("Saved results/step4_s7.json")
