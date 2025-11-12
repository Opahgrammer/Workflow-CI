import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def main():
    # ======================================================
    # 🔧 1. Setup MLflow lokal (agar tidak error di Ubuntu runner)
    # ======================================================
    tracking_dir = os.path.join(os.getcwd(), "mlruns")
    os.makedirs(tracking_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{tracking_dir}")

    # ======================================================
    # 📂 2. Load dataset
    # ======================================================
    data_path = "heart_disease_uci_preprocessing.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")
    
    data = pd.read_csv(data_path)

    # ======================================================
    # 🧩 3. Validasi dan preprocessing
    # ======================================================
    if "num" not in data.columns:
        raise ValueError("Kolom target 'num' tidak ditemukan di dataset!")

    # Ubah kolom target jadi binary (1 = sakit, 0 = sehat)
    data["num"] = data["num"].apply(lambda x: 1 if x > 0 else 0)

    # Konversi kolom bertipe object menjadi angka
    for col in data.columns:
        if data[col].dtype == "object":
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    # ======================================================
    # ✂️ 4. Split data
    # ======================================================
    X = data.drop(columns=["num"])
    y = data["num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ======================================================
    # 🚀 5. Jalankan MLflow experiment
    # ======================================================
    mlflow.set_experiment("CI-Auto-Retrain-Model")

    with mlflow.start_run() as run:
        mlflow.sklearn.autolog()

        # Buat dan latih model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluasi
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log metrik dan model
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Simpan info run untuk pipeline CI/CD
        run_info = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
        }

        with open("latest_run.json", "w") as f:
            json.dump(run_info, f, indent=4)

        print("\n==============================")
        print(f"✅ Model retrained successfully!")
        print(f"📊 Accuracy: {acc:.4f}")
        print(f"🧠 Run ID: {run.info.run_id}")
        print(f"🧪 Experiment ID: {run.info.experiment_id}")
        print("==============================\n")


if __name__ == "__main__":
    main()