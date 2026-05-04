"""
Task 2 — CLI Predictor (runs inside Docker)
Usage:
  python src/predict_cli.py --soil_moisture_pct 25.1 --crop_type_index 3 \
                            --field_size_hectares 16.5 --temperature_c 28.7
"""
import argparse, json, os
import numpy as np
import joblib

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "best_model.pkl")

def main():
    p = argparse.ArgumentParser(description="CropSense Irrigation Predictor")
    p.add_argument("--soil_moisture_pct",   type=float, required=True)
    p.add_argument("--crop_type_index",     type=float, required=True)
    p.add_argument("--field_size_hectares", type=float, required=True)
    p.add_argument("--temperature_c",       type=float, required=True)
    args = p.parse_args()

    model = joblib.load(MODEL_PATH)
    X = np.array([[args.soil_moisture_pct, args.crop_type_index,
                   args.field_size_hectares, args.temperature_c]])
    pred = round(float(model.predict(X)[0]), 4)

    print(json.dumps({"predicted_irrigation_hours": pred}, indent=2))
    return pred

if __name__ == "__main__":
    main()
