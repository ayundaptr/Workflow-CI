import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Tambahkan print agar terminal tidak kosong
print("Memulai proses training...")

# Pastikan path ini benar (gunakan ../ jika csv ada di luar folder ini)
df = pd.read_csv("../waterquality_preprocessing/waterquality_preprocessing.csv")
X = df.drop(columns=["is_safe"]) 
y = df["is_safe"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Water_Quality_Basic")
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Basic_Run"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # simpan model ke folder khusus agar mudah dicopy
    mlflow.sklearn.log_model(model, "model")
    print("Training Selesai! Model telah disimpan di folder mlruns.")