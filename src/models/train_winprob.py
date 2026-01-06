import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import mlflow
import mlflow.sklearn

FEATURES = ["minute", "score_diff", "xg_diff", "home_red", "away_red"]
LABEL = "outcome"  # H / D / A

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/match_states.parquet")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.data).dropna(subset=FEATURES + [LABEL, "match_id"]).copy()

    X = df[FEATURES]
    y = df[LABEL]
    groups = df["match_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    tr, te = next(gss.split(X, y, groups=groups))

    X_tr, X_te = X.iloc[tr], X.iloc[te]
    y_tr, y_te = y.iloc[tr], y.iloc[te]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=4000,
            solver="lbfgs",
            multi_class="multinomial"
        ))
    ])

    mlflow.set_experiment("winprob_wc2018")

    with mlflow.start_run():
        mlflow.log_param("features", ",".join(FEATURES))
        mlflow.log_param("test_size", args.test_size)

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)

        classes = list(model.named_steps["clf"].classes_)
        mlflow.log_param("classes", ",".join(classes))

        ll = log_loss(y_te, proba, labels=classes)
        mlflow.log_metric("log_loss", ll)

        Path("models").mkdir(exist_ok=True)
        joblib.dump({"model": model, "classes": classes}, "models/winprob.pkl")
        mlflow.log_artifact("models/winprob.pkl")

        print("Saved models/winprob.pkl")
        print("classes:", classes)
        print("log_loss:", ll)

if __name__ == "__main__":
    main()
