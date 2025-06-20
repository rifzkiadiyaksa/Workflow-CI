import json
import time
import os
import joblib

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

X_train = train_df.drop(columns="annual_income_category")
y_train = train_df["annual_income_category"]
X_test = test_df.drop(columns="annual_income_category")
y_test = test_df["annual_income_category"]

input_example = X_train.head(1)


def objective(trial, run_id):
    params = {
        "boosting_type": "gbdt",
        "device": "cpu",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.15),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name=f"LightGBM_Trial_{run_id}"):
        model = LGBMClassifier(**params)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        start_predict = time.time()
        preds = model.predict(X_test)
        predict_time = time.time() - start_predict

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        mlflow.log_params(params)
        mlflow.log_param("trial_id", run_id)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("train_time_seconds", train_time)
        mlflow.log_metric("predict_time_seconds", predict_time)

        metrics_dict = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "train_time_seconds": train_time,
            "predict_time_seconds": predict_time,
        }
        mlflow.log_text(json.dumps(metrics_dict, indent=2), "metric_info.json")

        mlflow.sklearn.log_model(
            model, f"model_trial_{run_id}", input_example=input_example
        )

        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax_cm, cmap="Blues")
        plt.tight_layout()
        mlflow.log_figure(fig_cm, "confusion_matrix.png")

        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
        importances = model.feature_importances_
        feature_names = X_train.columns
        indices = np.argsort(importances)[::-1]
        ax_fi.bar(range(len(importances)), importances[indices], align="center")
        ax_fi.set_xticks(range(len(importances)))
        ax_fi.set_xticklabels(feature_names[indices], rotation=90)
        ax_fi.set_title(f"Feature Importance - Trial {run_id}")
        plt.tight_layout()
        mlflow.log_figure(fig_fi, "feature_importance.png")

        github_artifacts_dir = f"github_artifacts/run_{run_id}"
        os.makedirs(github_artifacts_dir, exist_ok=True)
        joblib.dump(
            model, os.path.join(github_artifacts_dir, f"model_trial_{run_id}.pkl")
        )
        fig_cm.savefig(os.path.join(github_artifacts_dir, "confusion_matrix.png"))
        plt.close(fig_cm)
        fig_fi.savefig(os.path.join(github_artifacts_dir, "feature_importance.png"))
        plt.close(fig_fi)

        return acc


study = optuna.create_study(direction="maximize")
for i in range(15):
    study.optimize(lambda trial: objective(trial, i + 1), n_trials=1)


best_model_params = study.best_trial.params
final_model_for_serving = LGBMClassifier(**best_model_params)
final_model_for_serving.fit(X_train, y_train)

with mlflow.start_run(run_name="Final_Serving_Model_Logging") as run:
    mlflow.sklearn.log_model(
        final_model_for_serving, "serving_model", input_example=input_example
    )
    final_run_id = run.info.run_id

    run_id_file_path = os.path.join(os.path.dirname(__file__), "serving_run_id.txt")
    with open(run_id_file_path, "w") as f:
        f.write(final_run_id)