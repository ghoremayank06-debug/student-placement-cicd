"""FastAPI service for student placement predictions."""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
FEATURES = ["CGPA", "Attendance", "CodingScore", "Projects", "Internship"]
app = FastAPI(title="Student Placement Prediction API", version="1.0.0")

class StudentInput(BaseModel):
    cgpa: float = Field(ge=0, le=10)
    attendance: float = Field(ge=0, le=100)
    coding_score: float = Field(ge=0, le=100)
    projects: int = Field(ge=0)
    internship: int = Field(ge=0, le=1)

def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained. Run python train.py first.")
    try:
        return joblib.load(MODEL_PATH)
    except (OSError, ValueError, EOFError) as exc:
        raise HTTPException(status_code=500, detail="Unable to load model") from exc

@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}

@app.get("/model-info")
def model_info():
    return {"model": "DecisionTreeClassifier", "features": FEATURES, "model_file_exists": MODEL_PATH.exists()}

@app.post("/predict")
def predict(student: StudentInput):
    model = get_model()
    values = {
        "CGPA": student.cgpa,
        "Attendance": student.attendance,
        "CodingScore": student.coding_score,
        "Projects": student.projects,
        "Internship": student.internship,
    }
    prediction = int(model.predict(pd.DataFrame([values], columns=FEATURES))[0])
    return {"placement_prediction": prediction, "label": "PLACED" if prediction == 1 else "NOT PLACED"}
