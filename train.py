"""Train and validate the student placement prediction model."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_placement.csv"
MODEL_PATH = ROOT / "model.pkl"
FEATURES = ["CGPA", "Attendance", "CodingScore", "Projects", "Internship"]
TARGET = "Placement"

def load_and_validate_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    df = df[required].copy()
    if df.empty:
        raise ValueError("Dataset is empty")
    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not df["CGPA"].between(0, 10).all():
        raise ValueError("CGPA must be between 0 and 10")
    if not df["Attendance"].between(0, 100).all():
        raise ValueError("Attendance must be between 0 and 100")
    if not df["CodingScore"].between(0, 100).all():
        raise ValueError("CodingScore must be between 0 and 100")
    if not df["Projects"].ge(0).all():
        raise ValueError("Projects cannot be negative")
    if not df["Internship"].isin([0, 1]).all():
        raise ValueError("Internship must be 0 or 1")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Placement must be 0 or 1")
    if df[TARGET].nunique() < 2:
        raise ValueError("Placement target must contain both classes 0 and 1")
    return df

def train_model() -> float:
    df = load_and_validate_data()
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y
    )
    model = DecisionTreeClassifier(random_state=1)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    if accuracy < 0.80:
        raise ValueError(f"Accuracy below 80%: {accuracy:.3f}")
    joblib.dump(model, MODEL_PATH)
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        raise RuntimeError("Model file was not generated correctly")
    print(f"Model saved to {MODEL_PATH}")
    return accuracy

if __name__ == "__main__":
    train_model()
