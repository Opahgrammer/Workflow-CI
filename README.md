🚀 Workflow-CI: MLflow Auto Retrain & Docker Build

Project ini dibuat untuk mengimplementasikan Continuous Integration (CI) menggunakan GitHub Actions dan MLflow, agar proses training model Machine Learning, penyimpanan artefak, serta pembuatan image Docker dapat dilakukan secara otomatis setiap kali ada perubahan pada repository.

📁 Struktur Project
Workflow-CI/
├── .github/
│   └── workflows/
│       └── mlflow-ci.yml        # File workflow GitHub Actions
├── MLProject/
│   ├── modelling.py             # Script training model
│   ├── conda.yaml               # (Opsional) File environment MLflow
│   ├── heart_disease_uci_preprocessing.csv  # Dataset
│   └── latest_run.json          # File otomatis berisi run_id & experiment_id
└── README.md                    # Dokumentasi project

⚙️ 1. Persiapan Awal

Clone repository ini:

git clone https://github.com/USERNAME/Workflow-CI.git
cd Workflow-CI


Pastikan sudah memiliki akun Docker Hub

Buat repository baru (contoh: fajar17/heart-disease-mlflow)

Buat Personal Access Token di:
👉 Docker Hub → Account Settings → Security → New Access Token

Simpan token tersebut — akan digunakan di langkah selanjutnya.

🔑 2. Menambahkan Secrets di GitHub

Agar workflow bisa login ke Docker Hub dan melakukan push image otomatis:

Buka repo kamu di GitHub → tab ⚙️ Settings

Pilih Secrets and variables → Actions

Klik New repository secret

Tambahkan dua secret berikut:

Name	Value
DOCKERHUB_USERNAME	Username akun Docker Hub kamu
DOCKERHUB_TOKEN	Personal Access Token dari Docker Hub

⚠️ Jangan gunakan password akun Docker Hub asli! Gunakan token akses pribadi (PAT).

🧠 3. Menjalankan Training Model (Opsional)

Sebelum menjalankan workflow otomatis, kamu bisa tes lokal terlebih dahulu:

cd MLProject
python modelling.py


Jika berhasil, akan muncul:

Folder baru bernama mlruns/

File latest_run.json berisi run_id dan experiment_id

🤖 4. Menjalankan Workflow CI

Terdapat dua cara untuk menjalankan workflow otomatis:

🟢 Opsi 1 — Otomatis (Saat Push ke Main Branch)

Setiap kali kamu melakukan commit dan push perubahan ke folder MLProject/, workflow akan berjalan otomatis:

git add .
git commit -m "update model training"
git push origin main

🔵 Opsi 2 — Manual (Lewat GitHub Actions)

Masuk ke tab Actions di repository kamu.

Pilih workflow bernama MLflow CI Retrain and Build Docker.

Klik tombol Run workflow → jalankan secara manual.

🧱 5. Alur Workflow CI
Tahap	Penjelasan
1️⃣	GitHub Actions mendeteksi perubahan pada folder MLProject/.
2️⃣	Setup Python environment dan dependensi (mlflow, pandas, scikit-learn).
3️⃣	Jalankan modelling.py untuk retrain model.
4️⃣	MLflow menyimpan model ke folder mlruns/ dan file latest_run.json.
5️⃣	Workflow membangun Docker image dari model MLflow (mlflow models build-docker).
6️⃣	Docker image dikirim (push) ke akun Docker Hub kamu.
7️⃣	Artefak (mlruns) di-upload ke GitHub Actions sebagai bukti hasil workflow.
📦 6. Cek Hasil di GitHub Actions

Buka tab Actions di repository kamu.

Pilih run terbaru → pastikan semua langkah berstatus ✅ sukses.

Scroll ke bagian paling bawah — akan muncul kotak Artifacts.

Klik mlflow-model-artifacts → download untuk melihat isi folder mlruns hasil training.

🐳 7. Verifikasi di Docker Hub

Masuk ke Docker Hub

Buka repository kamu (contoh: fajar17/heart-disease-mlflow)

Pastikan image baru muncul:

heart-disease-mlflow:latest


✅ Berarti workflow berhasil membangun dan mengunggah image Docker hasil training terbaru.

🧩 8. Hasil Akhir (Checklist Penilaian)
Kriteria	Status
Membuat folder MLProject	✅
Membuat workflow CI (GitHub Actions)	✅
Menyimpan artefak hasil model ke GitHub	✅
Membuat dan push Docker Image ke Docker Hub	✅
📜 9. Rangkuman Akhir

Dengan workflow ini, setiap perubahan kode pada folder MLProject akan secara otomatis:

Menjalankan ulang proses training model ML.

Menyimpan hasilnya ke MLflow (mlruns).

Membangun image Docker dari model yang baru.

Mengunggah image ke Docker Hub.

Menyimpan artefak hasil training di GitHub.

💡 Tips Tambahan

Jika workflow gagal, buka tab Actions → Run logs untuk melihat pesan error.

Pastikan struktur path di modelling.py menggunakan path relatif (file:mlruns) — bukan drive lokal seperti D:\.

Jalankan modelling.py lokal untuk memastikan tidak ada error sebelum push ke GitHub.

✨ Kesimpulan

Project ini berhasil mengimplementasikan Workflow CI otomatis menggunakan MLflow dan Docker Hub.
Setiap update pada script training akan memicu retraining model, pembuatan artefak, serta build Docker image secara otomatis — menjadikan pipeline ini efisien, terukur, dan siap untuk deployment selanjutnya 🚀
