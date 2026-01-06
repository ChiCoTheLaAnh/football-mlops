import argparse
import pandas as pd
import numpy as np

import joblib
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/shots.parquet")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.data).copy()

    # Basic cleaning
    df = df.dropna(subset=["match_id", "x", "y", "body_part", "shot_type", "distance_to_goal", "shot_angle", "is_goal"])
    # Remove inf/-inf in numeric features (can happen if dx==0 or bad coords)
    num_cols = ["distance_to_goal", "shot_angle"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=num_cols)
    
    print("distance_to_goal min/max:", df["distance_to_goal"].min(), df["distance_to_goal"].max())
    print("shot_angle min/max:", df["shot_angle"].min(), df["shot_angle"].max())

    df["is_goal"] = df["is_goal"].astype(int)

    # Features/labels
    feature_cols_num = ["distance_to_goal", "shot_angle"]
    feature_cols_cat = ["body_part", "shot_type"]

    X = df[feature_cols_num + feature_cols_cat]
    y = df["is_goal"]
    groups = df["match_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
            ("num", StandardScaler(), feature_cols_num),
        ]
    )

    model = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=4000, solver="lbfgs", C=1.0)),
        ]
    )

    mlflow.set_experiment("xg_baseline_wc2018")

    with mlflow.start_run():
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("model", "logreg")
        mlflow.log_param("features_num", ",".join(feature_cols_num))
        mlflow.log_param("features_cat", ",".join(feature_cols_cat))
        mlflow.log_param("n_rows", len(df))

        model.fit(X_train, y_train)
        
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/xg_baseline.pkl")
        print("Saved local model: models/xg_baseline.pkl")

        p = model.predict_proba(X_test)[:, 1]

        ll = log_loss(y_test, p)
        auc = roc_auc_score(y_test, p)
        brier = brier_score_loss(y_test, p)

        mlflow.log_metric("log_loss", ll)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("brier", brier)

        # Calibration plot artifact
        prob_true, prob_pred = calibration_curve(y_test, p, n_bins=10, strategy="uniform")
        plt.figure()
        plt.plot(prob_pred, prob_true, marker="o")
        plt.xlabel("Predicted probability")
        plt.ylabel("Empirical probability")
        plt.title("Calibration curve (xG baseline)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("calibration.png")
        mlflow.log_artifact("calibration.png")

        # Save model to MLflow
        input_example = X_train.head(5)
        mlflow.sklearn.log_model(model, name="model", input_example=input_example)
        print("Metrics:")
        print(" log_loss:", ll)
        print(" roc_auc:", auc)
        print(" brier:", brier)
        print("Saved MLflow model + calibration plot.")


if __name__ == "__main__":
    main()
