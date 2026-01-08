from pathlib import Path
import joblib
import pandas as pd

from src.features.geometry import shot_distance, shot_angle
from src.modeling.pipeline import build_pipeline

ARTIFACT_DIR = Path("artifacts/models")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance"] = df.apply(lambda r: shot_distance(r["x"], r["y"]), axis=1)
    df["angle"] = df.apply(lambda r: shot_angle(r["x"], r["y"]), axis=1)
    return df

def main(train_path: str, model_name: str = "xg_logreg.joblib"):
    df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    df = add_features(df).dropna()

    X = df[["distance", "angle", "shot_body_part", "shot_type", "play_pattern"]]
    y = df["is_goal"].astype(int)

    pipe = build_pipeline()
    pipe.fit(X, y)

    out = ARTIFACT_DIR / model_name
    joblib.dump(pipe, out)
    print(f"Saved model to: {out}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--model_name", default="xg_logreg.joblib")
    args = ap.parse_args()
    main(args.train_path, args.model_name)
