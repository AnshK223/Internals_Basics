"""
Task 3 — Model Versioning
Run AFTER train.py.
Registers the best model in MLflow Model Registry.
Outputs: results/step3_s6.json
"""
import os, json, warnings
warnings.filterwarnings("ignore")

import mlflow
from mlflow import MlflowClient

BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE, "models")
RESULTS_DIR = os.path.join(BASE, "results")
REGISTERED_NAME = "cropsense-irrigation-hours-predictor"

mlflow.set_tracking_uri("file:///" + os.path.join(BASE, "mlruns").replace("\\", "/"))
client = MlflowClient()

meta = json.load(open(os.path.join(MODELS_DIR, "run_ids.json")))
best_run_id = meta["best_run_id"]
best_mae    = meta["best_mae"]

# Build artifact URI for that run's model
run = client.get_run(best_run_id)
artifact_uri = run.info.artifact_uri + "/model"

# Register
try:
    client.create_registered_model(REGISTERED_NAME)
except Exception:
    pass  # already exists

mv = client.create_model_version(
    name=REGISTERED_NAME,
    source=artifact_uri,
    run_id=best_run_id,
)

# Wait until READY
import time
for _ in range(20):
    mv = client.get_model_version(REGISTERED_NAME, mv.version)
    if mv.status == "READY":
        break
    time.sleep(1)

version = int(mv.version)
print(f"Registered version {version}  run_id={best_run_id}")

result = {
    "registered_model_name": REGISTERED_NAME,
    "version":               version,
    "run_id":                best_run_id,
    "source_metric":         "mae",
    "source_metric_value":   best_mae,
}
json.dump(result, open(os.path.join(RESULTS_DIR, "step3_s6.json"), "w"), indent=2)
print("Saved results/step3_s6.json")
