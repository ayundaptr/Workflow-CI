import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Memulai proses training BASIC...")

# --- Penanganan Path Baru (Data sudah di dalam MLProject) ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# Karena folder data sekarang "tetangga" dengan script ini, kita langsung panggil
csv_path = os.path.join(current_dir, "waterquality_preprocessing", "waterquality_preprocessing.csv")

print(f"Membaca data dari: {csv_path}")

try:
    df = pd.read_csv(csv_path)
    print("Data berhasil dimuat!")
except FileNotFoundError:
    # Backup: Coba cari tanpa path absolut (untuk fleksibilitas terminal)
    csv_path_alt = os.path.join("waterquality_preprocessing", "waterquality_preprocessing.csv")
    if os.path.exists(csv_path_alt):
        df = pd.read_csv(csv_path_alt)
        print("Data berhasil dimuat (via alt path)!")
    else:
        print(f"Error: File tetap tidak ditemukan. Cek folder MLProject kamu.")
        exit(1)

# --- 2. Persiapan Data ---
X = df.drop(columns=["is_safe"]) 
y = df["is_safe"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. MLflow Autolog (Standar Level Basic) ---
mlflow.sklearn.autolog()

# --- 4. Training & Logging ---
# nested=True sangat penting agar sukses di GitHub Actions (mencegah konflik Active Run ID)
with mlflow.start_run(run_name="Basic_Run", nested=True):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log model ke folder "model"
    mlflow.sklearn.log_model(model, "model")
    print("--------------------------------------------------")
    print("Training Basic Selesai!")
    print("Model dan metrik otomatis telah dicatat oleh Autolog.")
    print("--------------------------------------------------")