import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load data (Path sudah benar menggunakan ../)
df = pd.read_csv("../waterquality_preprocessing/waterquality_preprocessing.csv")
X = df.drop(columns=["is_safe"]) 
y = df["is_safe"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Setup MLflow
# Catatan: mlflow.set_experiment boleh tetap ada atau dihapus jika sudah diatur di file 'MLProject'
mlflow.set_experiment("Water_Quality_Basic")
mlflow.sklearn.autolog()

# 3. Training Model (TANPA mlflow.start_run agar tidak bentrok ID)
# Autolog akan otomatis merekam ke run yang sedang aktif di GitHub Actions
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Re-triggering Workflow for Skilled Level