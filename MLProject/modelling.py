# modelling.py - Versi Final dengan Path ke Sub-folder untuk CI/CD

import os
import json
import joblib
import pandas as pd
import mlflow
import optuna
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)

# --- PENGATURAN MLFLOW UNTUK LOKAL (CI) ---
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Automated-Hyperparameter-Tuning-CI")


# --- MEMUAT DATA DARI SUB-FOLDER ---
# PERBAIKAN: Path disesuaikan untuk membaca dari folder 'lung-cancer-model'
data_folder = 'lung-cancer-model'
train_file_path = os.path.join(data_folder, 'lung_cancer_train_preprocessed.csv')
test_file_path = os.path.join(data_folder, 'lung_cancer_test_preprocessed.csv')

try:
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    print(f"File CSV berhasil dimuat dari folder '{data_folder}'.")
except FileNotFoundError:
    print(f"ERROR: File CSV tidak ditemukan! Pastikan folder '{data_folder}' ada dan berisi file 'lung_cancer_train_preprocessed.csv' dan 'lung_cancer_test_preprocessed.csv'.")
    exit(1)


X_train = train_df.drop('LUNG_CANCER', axis=1)
y_train = train_df['LUNG_CANCER']
X_test = test_df.drop('LUNG_CANCER', axis=1)
y_test = test_df['LUNG_CANCER']


# --- FASE 1: FUNGSI UNTUK TUNING (MENGGUNAKAN OPTUNA) ---
def objective(trial):
    """Fungsi ini akan dijalankan berulang kali oleh Optuna untuk setiap trial."""
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 21, step=2),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "algorithm": trial.suggest_categorical(
            "algorithm", ["ball_tree", "kd_tree", "brute"]
        ),
        "p": trial.suggest_int("p", 1, 2),
    }

    # Menambahkan nested=True agar bisa berjalan di dalam 'mlflow run'
    with mlflow.start_run(run_name=f"Tuning_Trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        
        model = KNeighborsClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        
    return accuracy


# --- MENJALANKAN FASE 1: PROSES TUNING ---
print("\n--- Memulai Fase 1: Tuning Hyperparameter ---")
study = optuna.create_study(direction="maximize")
N_TRIALS = 20
study.optimize(objective, n_trials=N_TRIALS)
print("--- Fase 1 Selesai ---")


# --- FASE 2: FINALISASI OTOMATIS (MELATIH & MENYIMPAN MODEL TERBAIK) ---
print("\n--- Memulai Fase 2: Finalisasi Model Terbaik ---")

best_params = study.best_params
best_accuracy_from_tuning = study.best_value

print(f"Parameter terbaik ditemukan: {best_params}")
print(f"Akurasi terbaik dari tuning: {best_accuracy_from_tuning:.4f}")

# Membuat satu run baru yang terpisah KHUSUS untuk menyimpan model terbaik
with mlflow.start_run(run_name="Best-Model-Final", nested=True) as final_run:
    
    final_run_id = final_run.info.run_id
    print(f"Final Run ID untuk model terbaik: {final_run_id}")
    
    # Membuat file serving_run_id.txt agar bisa dibaca oleh main.yml
    with open("serving_run_id.txt", "w") as f:
        f.write(final_run_id)

    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy_from_tuning", best_accuracy_from_tuning)

    print("Melatih ulang model final dengan parameter terbaik...")
    final_model = KNeighborsClassifier(**best_params)
    final_model.fit(X_train, y_train)

    # Evaluasi komprehensif pada model final
    y_pred_final = final_model.predict(X_test)
    
    final_accuracy = accuracy_score(y_test, y_pred_final)
    final_precision = precision_score(y_test, y_pred_final, average="weighted", zero_division=0)
    final_recall = recall_score(y_test, y_pred_final, average="weighted", zero_division=0)
    final_f1 = f1_score(y_test, y_pred_final, average="weighted", zero_division=0)

    print("Logging metrik dari model final...")
    mlflow.log_metric("final_accuracy", final_accuracy)
    mlflow.log_metric("final_precision", final_precision)
    mlflow.log_metric("final_recall", final_recall)
    mlflow.log_metric("final_f1_score", final_f1)

    print("Menyimpan artefak model final...")
    joblib.dump(final_model, "best_model.pkl")
    # Menggunakan nama artefak 'serving_model' agar konsisten dengan main.yml
    mlflow.log_artifact("best_model.pkl", artifact_path="serving_model")

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(final_model, X_test, y_test, ax=ax, cmap=plt.cm.Blues)
    ax.set_title("Final Model Confusion Matrix")
    plt.tight_layout()
    mlflow.log_figure(fig, "final_confusion_matrix.png")
    plt.close(fig)

    mlflow.set_tag("model_status", "best_after_tuning")

    print("\n--- Fase 2 Selesai. Model terbaik telah disimpan di run 'Best-Model-Final'. ---")