# =====================================================
# modelling.py
# =====================================================

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.tree import export_graphviz

# =====================================================
# Model Evaluation
# =====================================================
def evaluate_model(model, X_test, y_test, output_dir):
    """
    Evaluasi model dan menyimpan artefak evaluasi:
    - metric_info.json
    - confusion_matrix.png
    - classification_report.txt
    - feature_importance.png
    """

    os.makedirs(output_dir, exist_ok=True)

    # Prediction
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    # ===============================
    # Save metrics (JSON)
    # ===============================
    with open(os.path.join(output_dir, "metric_info.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # ===============================
    # Confusion Matrix (PNG)
    # ===============================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # ===============================
    # Classification Report (TXT)
    # ===============================
    report = classification_report(y_test, y_pred)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # ===============================
    # Feature Importance (PNG)
    # ===============================
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(8, 5))
        feat_importance = model.feature_importances_
        feat_names = X_test.columns
        sns.barplot(x=feat_importance, y=feat_names, palette="viridis")
        plt.xlabel("Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance.png"))
        plt.close()

    return metrics


# =====================================================
# RandomForest Visual Artifact
# =====================================================
def save_estimator_html(model, feature_names, output_dir, max_depth=3):
    """
    Simpan satu estimator RandomForest sebagai artefak visual:
    - estimator.dot: Data mentah grafik pohon.
    - estimator.html: Panduan instruksi visualisasi.
    Batasi kedalaman pohon untuk mempermudah visualisasi di level Advanced.
    """

    # Memastikan direktori output tersedia
    os.makedirs(output_dir, exist_ok=True)

    # Mengambil satu pohon (estimator pertama) untuk divisualisasikan
    estimator = model.estimators_[0]

    # 1. Simpan file .dot (format standar Graphviz)
    dot_filename = "estimator.dot"
    dot_path = os.path.join(output_dir, dot_filename)
    export_graphviz(
        estimator,
        out_file=dot_path,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        max_depth=max_depth # Membatasi kedalaman agar tidak terlalu kompleks
    )

    # 2. Simpan file .html (sebagai panduan pembaca di MLflow/DagsHub)
    html_path = os.path.join(output_dir, "estimator.html")
    with open(html_path, "w") as f:
        f.write(
            f"""
            <!DOCTYPE html>
            <html>
              <head>
                <title>RandomForest Estimator Info</title>
                <style>
                  body {{ font-family: sans-serif; line-height: 1.6; padding: 20px; }}
                  code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
                </style>
              </head>
              <body>
                <h1>RandomForest Tree Visualization</h1>
                <p>Estimator disimpan sebagai file DOT di: <code>artifacts/{dot_filename}</code></p>
                
                <h3>Cara Melihat Pohon:</h3>
                <ul>
                  <li>Gunakan <b>Graphviz</b> secara lokal.</li>
                  <li>Gunakan VSCode extension <b>'Graphviz Preview'</b> untuk melihat langsung di editor.</li>
                  <li>Atau salin isi file <code>.dot</code> ke <a href="http://webgraphviz.com/" target="_blank">WebGraphviz.com</a>.</li>
                </ul>

                <p><i>Catatan: Kedalaman pohon dibatasi hingga <b>max_depth={max_depth}</b> agar struktur logika model mudah dipahami dan divisualisasi.</i></p>
              </body>
            </html>
            """
        )

        # =====================================================
# modelling.py
# =====================================================

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.tree import export_graphviz

# =====================================================
# Model Evaluation
# =====================================================
def evaluate_model(model, X_test, y_test, output_dir):
    """
    Evaluasi model dan menyimpan artefak evaluasi:
    - metric_info.json
    - confusion_matrix.png
    - classification_report.txt
    - feature_importance.png
    """

    os.makedirs(output_dir, exist_ok=True)

    # Prediction
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    # ===============================
    # Save metrics (JSON)
    # ===============================
    with open(os.path.join(output_dir, "metric_info.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # ===============================
    # Confusion Matrix (PNG)
    # ===============================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # ===============================
    # Classification Report (TXT)
    # ===============================
    report = classification_report(y_test, y_pred)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # ===============================
    # Feature Importance (PNG)
    # ===============================
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(8, 5))
        feat_importance = model.feature_importances_
        feat_names = X_test.columns
        sns.barplot(x=feat_importance, y=feat_names, palette="viridis")
        plt.xlabel("Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance.png"))
        plt.close()

    return metrics


# =====================================================
# RandomForest Visual Artifact
# =====================================================
def save_estimator_html(model, feature_names, output_dir, max_depth=3):
    """
    Simpan satu estimator RandomForest sebagai artefak visual:
    - estimator.dot: Data mentah grafik pohon.
    - estimator.html: Panduan instruksi visualisasi.
    Batasi kedalaman pohon untuk mempermudah visualisasi di level Advanced.
    """

    # Memastikan direktori output tersedia
    os.makedirs(output_dir, exist_ok=True)

    # Mengambil satu pohon (estimator pertama) untuk divisualisasikan
    estimator = model.estimators_[0]

    # 1. Simpan file .dot (format standar Graphviz)
    dot_filename = "estimator.dot"
    dot_path = os.path.join(output_dir, dot_filename)
    export_graphviz(
        estimator,
        out_file=dot_path,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        max_depth=max_depth # Membatasi kedalaman agar tidak terlalu kompleks
    )

    # 2. Simpan file .html (sebagai panduan pembaca di MLflow/DagsHub)
    html_path = os.path.join(output_dir, "estimator.html")
    with open(html_path, "w") as f:
        f.write(
            f"""
            <!DOCTYPE html>
            <html>
              <head>
                <title>RandomForest Estimator Info</title>
                <style>
                  body {{ font-family: sans-serif; line-height: 1.6; padding: 20px; }}
                  code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
                </style>
              </head>
              <body>
                <h1>RandomForest Tree Visualization</h1>
                <p>Estimator disimpan sebagai file DOT di: <code>artifacts/{dot_filename}</code></p>
                
                <h3>Cara Melihat Pohon:</h3>
                <ul>
                  <li>Gunakan <b>Graphviz</b> secara lokal.</li>
                  <li>Gunakan VSCode extension <b>'Graphviz Preview'</b> untuk melihat langsung di editor.</li>
                  <li>Atau salin isi file <code>.dot</code> ke <a href="http://webgraphviz.com/" target="_blank">WebGraphviz.com</a>.</li>
                </ul>

                <p><i>Catatan: Kedalaman pohon dibatasi hingga <b>max_depth={max_depth}</b> agar struktur logika model mudah dipahami dan divisualisasi.</i></p>
              </body>
            </html>
            """
        )

# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    print("modelling.py berhasil dijalankan")
        