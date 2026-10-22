"""Unit and API tests for the student placement project."""
from pathlib import Path
import joblib
import pandas as pd
from fastapi.testclient import TestClient
from app import app
from train import DATA_PATH, MODEL_PATH, FEATURES, load_and_validate_data

ROOT = Path(__file__).resolve().parent
client = TestClient(app)

def test_dataset_exists_and_validates():
    assert DATA_PATH.exists()
    df = load_and_validate_data()
    assert not df.empty
    assert set(FEATURES + ["Placement"]).issubset(df.columns)

def test_model_file_exists_and_loads():
    assert MODEL_PATH.exists()
    assert joblib.load(MODEL_PATH) is not None

def test_model_prediction_is_binary():
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([{"CGPA": 8.5, "Attendance": 90, "CodingScore": 85, "Projects": 3, "Internship": 1}], columns=FEATURES)
    assert int(model.predict(sample)[0]) in [0, 1]

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_prediction_api():
    response = client.post("/predict", json={"cgpa": 8.5, "attendance": 90, "coding_score": 85, "projects": 3, "internship": 1})
    assert response.status_code == 200
    assert response.json()["placement_prediction"] in [0, 1]
    assert response.json()["label"] in ["PLACED", "NOT PLACED"]

def test_prediction_rejects_invalid_input():
    response = client.post("/predict", json={"cgpa": 15, "attendance": 90, "coding_score": 85, "projects": 3, "internship": 1})
    assert response.status_code == 422
