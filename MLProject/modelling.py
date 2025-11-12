import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import json

def main():
    # Pastikan MLflow menyimpan log di folder lokal (bukan drive Windows)
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")  

    # 1. Load dataset
    data = pd.read_csv("heart_disease_uci_preprocessing.csv")

    # 2. Pastikan kolom target 'num' ada
    if "num" not in data.columns:
        raise ValueError("Kolom target 'num' tidak ditemukan di dataset!")

    # 3. Konversi target ke binary (1 = ada penyakit, 0 = tidak)
    data["num"] = data["num"].apply(lambda x: 1 if x > 0 else 0)

    # 4. Ubah semua kolom bertipe object menjadi angka
    for col in data.columns:
        if data[col].dtype == "object":
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    # 5. Pisahkan fitur dan target
    X = data.drop(columns=["num"])
    y = data["num"]

    # 6. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 7. Mulai MLflow run
    mlflow.set_experiment("CI-Auto-Retrain-Model")

    with mlflow.start_run() as run:
        mlflow.sklearn.autolog()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Simpan metrik dan model
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Simpan run_id agar bisa dipakai workflow CI
        run_info = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id
        }
        with open("latest_run.json", "w") as f:
            json.dump(run_info, f)

        print(f"✅ Model retrained successfully. Accuracy: {acc:.4f}")
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()
