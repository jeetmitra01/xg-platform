from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

NUM_COLS = ["distance", "angle"]
CAT_COLS = ["shot_body_part", "shot_type", "play_pattern"]

def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )

    model = LogisticRegression(max_iter=2000)

    return Pipeline(steps=[("preprocess", pre), ("model", model)])
