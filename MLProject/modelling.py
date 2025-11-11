# MLProject/modelling.py
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # 1. Load dataset
    data = pd.read_csv("heart_disease_uci_preprocessing.csv")

    X = data.drop(columns=["num"])
    y = data["num"]

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Start MLflow run
    mlflow.set_experiment("CI-Auto-Retrain-Model")

    with mlflow.start_run():
        mlflow.sklearn.autolog()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"✅ Model retrained successfully. Accuracy: {acc:.4f}")

        mlflow.log_metric("accuracy", acc)

if __name__ == "__main__":
    main()