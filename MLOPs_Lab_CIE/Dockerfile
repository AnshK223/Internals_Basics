FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/predict_cli.py src/predict_cli.py
COPY models/best_model.pkl models/best_model.pkl

ENTRYPOINT ["python", "src/predict_cli.py"]
