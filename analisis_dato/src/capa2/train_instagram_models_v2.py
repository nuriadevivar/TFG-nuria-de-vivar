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
INPUT_PATH = "analisis_dato/data/analytic/capa2/instagram_model_input.csv"
OUTPUT_DIR = "analisis_dato/data/analytic/capa2"
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

# =====================================================
# Transformación cíclica del mes
# =====================================================
df["mes_sin"] = np.sin(2 * np.pi * df["mes_num"] / 12)
df["mes_cos"] = np.cos(2 * np.pi * df["mes_num"] / 12)

# =====================================================
# Features / target
# =====================================================
target_col = "alto_engagement"

feature_cols = ["marca", "anio", "tipo_post", "mes_sin", "mes_cos"]

X = df[feature_cols].copy()
y = df[target_col].astype(int)

categorical_features = ["marca", "tipo_post"]
numeric_features = ["anio", "mes_sin", "mes_cos"]

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
# Split estratificado
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- TRAIN / TEST ---")
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
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    ))
])

models = {
    "logistic_regression_v2": logistic_model,
    "random_forest_v2": rf_model
}

results = []

# =====================================================
# Entrenamiento + evaluación
# =====================================================
for model_name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    results.append(metrics)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Matriz de confusión - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{model_name}_confusion_matrix.png"), dpi=300)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f"Curva ROC - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{model_name}_roc_curve.png"), dpi=300)
    plt.close()

# =====================================================
# Guardar métricas
# =====================================================
results_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)
results_df.to_csv(os.path.join(METRICS_DIR, "instagram_models_metrics_v2.csv"), index=False)

with open(os.path.join(METRICS_DIR, "instagram_models_metrics_v2.json"), "w", encoding="utf-8") as f:
    json.dump(results_df.to_dict(orient="records"), f, ensure_ascii=False, indent=4)

print("\n--- MÉTRICAS MODELOS V2 ---")
print(results_df)

# =====================================================
# Importancia de variables RF
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

importance_df.to_csv(os.path.join(REPORTS_DIR, "instagram_rf_feature_importance_v2.csv"), index=False)

top_n = 12
plt.figure(figsize=(10, 6))
plt.barh(
    importance_df["feature"].head(top_n)[::-1],
    importance_df["importance"].head(top_n)[::-1]
)
plt.title("Top variables más importantes - Random Forest V2")
plt.xlabel("Importancia")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "instagram_rf_feature_importance_v2.png"), dpi=300)
plt.close()

print("\nScript de modelado Instagram V2 completado.")