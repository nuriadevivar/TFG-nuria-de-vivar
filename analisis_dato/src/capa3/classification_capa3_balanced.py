import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# Rutas
# =====================================================
INPUT_PATH = "analisis_dato/data/analytic/capa3/capa3_supervised_balanced_generation.csv"
OUTPUT_DIR = "analisis_dato/data/analytic/capa3"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# =====================================================
# Carga de datos
# =====================================================
df = pd.read_csv(INPUT_PATH)

print("\n--- INFO GENERAL ---")
print(df.info())

print("\n--- PRIMERAS FILAS ---")
print(df.head())

print("\n--- NULOS POR COLUMNA ---")
print(df.isnull().sum())

# =====================================================
# Target y features
# =====================================================
target_col = "target_seguira_comprando_bin"

drop_cols = [
    "id_respuesta",
    "target_recomendaria_bin",
    "target_seguira_comprando_bin"
]

X = df.drop(columns=drop_cols)
y = df[target_col].astype(int)

print("\n--- DISTRIBUCIÓN TARGET BALANCED ---")
print(y.value_counts())
print(y.value_counts(normalize=True))

categorical_features = ["grupo_edad", "sexo", "gasto_mensual_moda"]
numeric_features = [col for col in X.columns if col not in categorical_features]

# =====================================================
# Preprocesado
# =====================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# =====================================================
# Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- TRAIN / TEST BALANCED ---")
print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")

# =====================================================
# Modelos
# =====================================================
logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_split=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ))
])

models = {
    "logistic_regression_balanced": logistic_model,
    "random_forest_balanced": rf_model
}

# =====================================================
# Función métricas
# =====================================================
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

results = []

# =====================================================
# Entrenamiento y evaluación
# =====================================================
for model_name, model in models.items():
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob)

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)

    row = {
        "model": model_name,
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1": train_metrics["f1_score"],
        "train_auc": train_metrics["roc_auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1_score"],
        "test_auc": test_metrics["roc_auc"],
        "gap_f1": train_metrics["f1_score"] - test_metrics["f1_score"],
        "gap_auc": train_metrics["roc_auc"] - test_metrics["roc_auc"]
    }
    results.append(row)

    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Matriz de confusión - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{model_name}_confusion_matrix.png"), dpi=300)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_test_prob)
    plt.title(f"Curva ROC - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{model_name}_roc_curve.png"), dpi=300)
    plt.close()

# =====================================================
# Guardar métricas
# =====================================================
results_df = pd.DataFrame(results).sort_values(by="test_auc", ascending=False)
results_df.to_csv(os.path.join(METRICS_DIR, "capa3_supervised_metrics_balanced.csv"), index=False)

with open(os.path.join(METRICS_DIR, "capa3_supervised_metrics_balanced.json"), "w", encoding="utf-8") as f:
    json.dump(results_df.to_dict(orient="records"), f, ensure_ascii=False, indent=4)

print("\n--- MÉTRICAS TRAIN / TEST BALANCED ---")
print(results_df)

# =====================================================
# Importancia variables RF
# =====================================================
rf_fitted = rf_model.fit(X_train, y_train)

preprocessor_fitted = rf_fitted.named_steps["preprocessor"]
rf_classifier = rf_fitted.named_steps["classifier"]

feature_names = preprocessor_fitted.get_feature_names_out()
importances = rf_classifier.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

importance_df.to_csv(os.path.join(REPORTS_DIR, "capa3_rf_feature_importance_balanced.csv"), index=False)

top_n = 15
plt.figure(figsize=(10, 7))
plt.barh(
    importance_df["feature"].head(top_n)[::-1],
    importance_df["importance"].head(top_n)[::-1]
)
plt.title("Top variables más importantes - Random Forest Capa 3 BALANCED")
plt.xlabel("Importancia")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa3_rf_feature_importance_balanced.png"), dpi=300)
plt.close()

print("\nScript de clasificación capa 3 BALANCED completado.")