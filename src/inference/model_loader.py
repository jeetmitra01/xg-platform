from pathlib import Path
import joblib

DEFAULT_MODEL_PATH = Path("artifacts/models/xg_logreg.joblib")

def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first or place an artifact there."
        )
    return joblib.load(model_path)
