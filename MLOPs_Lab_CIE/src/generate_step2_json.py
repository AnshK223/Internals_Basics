"""
Task 2 helper — generates results/step2_s3.json after docker build+run.
Run this AFTER you have built and tested the Docker image.
Edit 'prediction' below with the actual output from docker run.
Or just run: python src/generate_step2_json.py --prediction <value>
"""
import argparse, json, os

BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

p = argparse.ArgumentParser()
p.add_argument("--prediction", type=float, required=True,
               help="Value printed by: docker run cropsense-predictor:v1 ...")
args = p.parse_args()

result = {
    "image_name": "cropsense-predictor",
    "image_tag":  "v1",
    "base_image": "python:3.11-slim",
    "test_input": {
        "soil_moisture_pct":   25.1,
        "crop_type_index":     3,
        "field_size_hectares": 16.5,
        "temperature_c":       28.7,
    },
    "prediction": args.prediction,
}
json.dump(result, open(os.path.join(RESULTS_DIR, "step2_s3.json"), "w"), indent=2)
print("Saved results/step2_s3.json")
