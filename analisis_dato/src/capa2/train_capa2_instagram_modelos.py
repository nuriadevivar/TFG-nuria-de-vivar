"""
train_capa2_instagram_modelos.py — Modelos supervisados: predicción de alto engagement
========================================================================================

MARCO TEÓRICO
-------------
El engagement en redes sociales se define como el conjunto de interacciones que
un usuario realiza con un contenido publicado (Likes + Comentarios). Predecir qué
publicaciones generarán mayor engagement es un problema de clasificación binaria:
dado un post con sus características observables (marca, tipo de contenido, contexto
temporal), ¿obtendrá un engagement por encima del percentil 75 de la distribución?

DEFINICIÓN DEL TARGET:
  alto_engagement = 1 si engagement_total_post > P75(engagement_total_post)
  alto_engagement = 0 en caso contrario
  Justificación del umbral P75: se define "alto engagement" como los posts que
  superan el tercer cuartil de la distribución, lo que equivale al 25% superior.
  Este umbral es robusto a outliers y corresponde a la práctica habitual en
  marketing digital (Jaakonmäki et al., 2017).

FEATURES USADAS:
  - marca:    identidad de la marca (variable categórica — proxy de audiencia base)
  - anio:     año de publicación (captura tendencias temporales)
  - tipo_post: tipo de contenido (Image/Video/Sidecar — impacto diferencial documentado)
  - mes_sin / mes_cos: codificación cíclica del mes (preserva la continuidad
    diciembre→enero que una variable numérica lineal no captura).
    mes_sin = sin(2π·mes/12), mes_cos = cos(2π·mes/12)
    Referencia: Cerda et al. (2018) — "Encoding Cyclical Features".

MODELOS COMPARADOS:
  MODELO A — Regresión Logística (baseline interpretable):
    Modelo lineal generalizado con función logit. Asume separabilidad lineal
    de las clases en el espacio de features. Ventajas: interpretabilidad directa
    de coeficientes, rapidez de entrenamiento, buena calibración de probabilidades.
    Parámetro class_weight='balanced': compensa el desequilibrio de clases 75/25.
    Referencia: Cox (1958).

  MODELO B — Random Forest (ensemble avanzado):
    Ensemble de árboles de decisión con bagging y selección aleatoria de features
    (Breiman, 2001). Captura no-linealidades e interacciones entre variables sin
    necesidad de especificación explícita. La importancia de variables (Gini
    importance) permite interpretar qué features son más relevantes para la
    predicción.
    Hiperparámetros: n_estimators=300, max_depth=None, class_weight='balanced'.

MÉTRICAS DE EVALUACIÓN:
  - Accuracy:  proporción de predicciones correctas. Puede ser engañosa con
               clases desbalanceadas — se complementa con F1.
  - Precision: de todos los predichos como "alto engagement", ¿cuántos lo son?
  - Recall:    de todos los posts con alto engagement real, ¿cuántos detectamos?
  - F1-Score:  media armónica de Precision y Recall. Métrica principal para
               clases desbalanceadas.
  - ROC-AUC:   capacidad discriminativa del modelo. AUC=0.5 equivale a azar;
               AUC=1.0 es clasificación perfecta.

SPLIT: 80% train / 20% test estratificado por clase (preserva distribución).
GAP TRAIN-TEST: se calcula la diferencia F1 y AUC entre train y test para
  detectar sobreajuste (gap > 0.1 indica overfitting relevante).

VISUALIZACIONES:
  - Matriz de confusión (TP, TN, FP, FN)
  - Curva ROC con AUC para ambos modelos
  - Feature importance del Random Forest
  - Tabla comparativa de métricas
"""

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
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# Rutas
# =====================================================
INPUT_PATH  = "data/analytic/capa2/instagram_model_input.csv"
OUTPUT_DIR  = "data/analytic/capa2"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

for d in [FIGURES_DIR, METRICS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# =====================================================
# Carga y preparación
# =====================================================
print("=" * 65)
print("CAPA 2 — MODELOS SUPERVISADOS: PREDICCIÓN ALTO ENGAGEMENT")
print("=" * 65)

df = pd.read_csv(INPUT_PATH)

# Codificación cíclica del mes (captura continuidad diciembre→enero)
df["mes_sin"] = np.sin(2 * np.pi * df["mes_num"] / 12)
df["mes_cos"] = np.cos(2 * np.pi * df["mes_num"] / 12)

TARGET = "alto_engagement"
FEATURES = ["marca", "anio", "tipo_post", "mes_sin", "mes_cos"]

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

print(f"\n[Dataset]")
print(f"  Total posts:  {len(df)}")
print(f"  Alto engagement (clase 1): {y.sum()} ({y.mean()*100:.1f}%)")
print(f"  Bajo engagement (clase 0): {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")
print(f"  Features: {FEATURES}")

CAT_FEATURES = ["marca", "tipo_post"]
NUM_FEATURES = ["anio", "mes_sin", "mes_cos"]

# =====================================================
# Preprocesado
# =====================================================
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer,    NUM_FEATURES),
    ("cat", categorical_transformer, CAT_FEATURES),
])

# =====================================================
# Split estratificado
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[Split 80/20 estratificado]")
print(f"  Train: {X_train.shape[0]} posts | Test: {X_test.shape[0]} posts")

# =====================================================
# Modelos
# =====================================================
models = {
    "A_Logistic_Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )),
    ]),
    "B_Random_Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", random_state=42
        )),
    ]),
}

# =====================================================
# Entrenamiento, evaluación y visualizaciones
# =====================================================
def eval_model(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)

    y_tr_pred = model.predict(X_tr)
    y_tr_prob = model.predict_proba(X_tr)[:, 1]
    y_te_pred = model.predict(X_te)
    y_te_prob = model.predict_proba(X_te)[:, 1]

    return {
        "model":         name,
        "train_accuracy":  accuracy_score(y_tr, y_tr_pred),
        "train_precision": precision_score(y_tr, y_tr_pred, zero_division=0),
        "train_recall":    recall_score(y_tr, y_tr_pred, zero_division=0),
        "train_f1":        f1_score(y_tr, y_tr_pred, zero_division=0),
        "train_auc":       roc_auc_score(y_tr, y_tr_prob),
        "test_accuracy":   accuracy_score(y_te, y_te_pred),
        "test_precision":  precision_score(y_te, y_te_pred, zero_division=0),
        "test_recall":     recall_score(y_te, y_te_pred, zero_division=0),
        "test_f1":         f1_score(y_te, y_te_pred, zero_division=0),
        "test_auc":        roc_auc_score(y_te, y_te_prob),
        "gap_f1":          f1_score(y_tr, y_tr_pred, zero_division=0) - f1_score(y_te, y_te_pred, zero_division=0),
        "gap_auc":         roc_auc_score(y_tr, y_tr_prob) - roc_auc_score(y_te, y_te_prob),
        "_y_te_pred": y_te_pred,
        "_y_te_prob":  y_te_prob,
    }

all_results = []
fitted_models = {}

for mname, mpipe in models.items():
    print(f"\n[{mname}]")
    res = eval_model(mname, mpipe, X_train, y_train, X_test, y_test)
    fitted_models[mname] = mpipe

    print(f"  Train → F1={res['train_f1']:.3f} | AUC={res['train_auc']:.3f}")
    print(f"  Test  → Accuracy={res['test_accuracy']:.3f} | Precision={res['test_precision']:.3f} | "
          f"Recall={res['test_recall']:.3f} | F1={res['test_f1']:.3f} | AUC={res['test_auc']:.3f}")
    gap_flag = "⚠ SOBREAJUSTE" if res["gap_f1"] > 0.1 else "✓ OK"
    print(f"  Gap F1={res['gap_f1']:.3f} | Gap AUC={res['gap_auc']:.3f} → {gap_flag}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, res["_y_te_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bajo", "Alto"])
    disp.plot(colorbar=False)
    plt.title(f"Matriz de Confusión — {mname}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"capa2_instagram_{mname}_confusion_matrix.png"), dpi=300)
    plt.close()

    # Curva ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_test, res["_y_te_prob"], ax=ax, name=mname)
    ax.plot([0, 1], [0, 1], "k--", label="Clasificador aleatorio")
    ax.set_title(f"Curva ROC — {mname}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"capa2_instagram_{mname}_roc_curve.png"), dpi=300)
    plt.close()

    all_results.append({k: v for k, v in res.items() if not k.startswith("_")})

# =====================================================
# Feature importance del Random Forest
# =====================================================
rf_pipe = fitted_models["B_Random_Forest"]
feat_names = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
importances = rf_pipe.named_steps["classifier"].feature_importances_

imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}) \
           .sort_values("importance", ascending=False)
imp_df.to_csv(os.path.join(REPORTS_DIR, "capa2_instagram_rf_feature_importance.csv"), index=False)

plt.figure(figsize=(10, 6))
top = imp_df.head(12)
plt.barh(top["feature"][::-1], top["importance"][::-1], color="#2196F3")
plt.title("Feature Importance — Random Forest | Predicción Alto Engagement Instagram")
plt.xlabel("Importancia (Gini)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa2_instagram_rf_feature_importance.png"), dpi=300)
plt.close()

# =====================================================
# Tabla comparativa y selección del modelo ganador
# =====================================================
results_df = pd.DataFrame(all_results).sort_values("test_auc", ascending=False)
results_df.to_csv(os.path.join(METRICS_DIR, "capa2_instagram_modelos_comparativa.csv"), index=False)
with open(os.path.join(METRICS_DIR, "capa2_instagram_modelos_comparativa.json"), "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

winner = results_df.iloc[0]
print("\n" + "=" * 50)
print(f"MODELO GANADOR: {winner['model']}")
print(f"  Test AUC={winner['test_auc']:.3f} | F1={winner['test_f1']:.3f} | "
      f"Accuracy={winner['test_accuracy']:.3f}")
print(f"  Gap AUC={winner['gap_auc']:.3f} | Gap F1={winner['gap_f1']:.3f}")

# Gráfico comparativo ROC
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], "k--", label="Clasificador aleatorio (AUC=0.50)")
for mname, mpipe in fitted_models.items():
    y_prob = mpipe.predict_proba(X_test)[:, 1]
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name=mname)
ax.set_title("Curva ROC Comparativa — Predicción Alto Engagement Instagram")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa2_instagram_roc_comparativa.png"), dpi=300)
plt.close()

# Interpretación del modelo ganador
interpretation = {
    "modelo_ganador":    winner["model"],
    "criterio":          "mayor_test_AUC",
    "test_auc":          winner["test_auc"],
    "test_f1":           winner["test_f1"],
    "test_accuracy":     winner["test_accuracy"],
    "gap_f1":            winner["gap_f1"],
    "gap_auc":           winner["gap_auc"],
    "sobreajuste":       "si" if winner["gap_f1"] > 0.1 else "no",
    "interpretacion": (
        f"El modelo {winner['model']} obtiene el mayor AUC en test ({winner['test_auc']:.3f}), "
        f"indicando buena capacidad discriminativa entre posts de alto y bajo engagement. "
        f"Un AUC > 0.7 es considerado aceptable para clasificación de comportamiento "
        f"en redes sociales (He et al., 2014). "
        f"La feature importance del Random Forest revela qué características de los posts "
        f"(marca, tipo de contenido, estacionalidad) son más determinantes para predecir "
        f"alta respuesta del público."
    ),
    "limitaciones": (
        "El dataset de posts es pequeño (n=2369) y cubre solo ~15 meses de actividad "
        "(2025), lo que limita la generalización del modelo. Las features disponibles "
        "son básicas (no incluyen texto del caption, número de seguidores en el momento "
        "de la publicación, ni información del algoritmo de distribución de Instagram). "
        "El engagement también depende fuertemente de la base de seguidores, que difiere "
        "sustancialmente entre marcas (Zara vs Massimo Dutti)."
    ),
}

with open(os.path.join(REPORTS_DIR, "capa2_instagram_modelo_ganador.json"), "w", encoding="utf-8") as f:
    json.dump(interpretation, f, ensure_ascii=False, indent=4)

print("\n" + "=" * 65)
print("CAPA 2 INSTAGRAM — COMPLETADO")
print(f"  Modelo A: Logistic Regression | AUC={all_results[0]['test_auc']:.3f}")
print(f"  Modelo B: Random Forest       | AUC={all_results[1]['test_auc']:.3f}")
print(f"  Ganador:  {winner['model']}")
print("Outputs:")
print("  figures/ → confusion_matrix, roc_curve, feature_importance, roc_comparativa")
print("  metrics/ → capa2_instagram_modelos_comparativa.csv/.json")
print("  reports/ → capa2_instagram_modelo_ganador.json")
print("=" * 65)