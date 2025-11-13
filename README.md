# 🚀 Workflow-CI: MLflow Auto Retrain & Docker Build

Project ini dibuat untuk mengimplementasikan **Continuous Integration (CI)** menggunakan **GitHub Actions** dan **MLflow**, agar proses training model Machine Learning, penyimpanan artefak, serta pembuatan image Docker dapat dilakukan secara otomatis setiap kali ada perubahan pada repository.

## 📁 Struktur Project

```text
Workflow-CI/
├── .github/
│   └── workflows/
│       └── mlflow-ci.yml            # File workflow GitHub Actions
├── MLProject/
│   ├── modelling.py                 # Script training model
│   ├── conda.yaml                   # (Opsional) File environment MLflow
│   ├── heart_disease_uci_preprocessing.csv  # Dataset
│   └── latest_run.json              # File otomatis berisi run_id & experiment_id
└── README.md                        # Dokumentasi project
