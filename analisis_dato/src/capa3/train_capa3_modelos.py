"""
train_capa3_modelos.py — Modelos no supervisados y supervisados Capa 3: Encuesta
=================================================================================

MARCO TEÓRICO
-------------
La Capa 3 integra dos bloques analíticos sobre los datos de la encuesta de
comportamiento de consumo de moda en RRSS (n=318):

═══════════════════════════════════════════════════════
BLOQUE A — SEGMENTACIÓN NO SUPERVISADA (Clustering)
═══════════════════════════════════════════════════════

Objetivo: identificar perfiles naturales de consumidores según su actitud y
comportamiento ante la moda en redes sociales, sin etiquetas previas.

ALGORITMO: K-Means (MacQueen, 1967)
  K-Means minimiza la inercia intraclúster (suma de distancias cuadráticas al
  centroide). La función objetivo es:
    J = Σ_k Σ_{x∈C_k} ||x - μ_k||²
  donde μ_k es el centroide del clúster k.
  Hiperparámetro n_init=20: reinicializa los centroides 20 veces para evitar
  soluciones locales subóptimas.

SELECCIÓN DEL NÚMERO DE CLÚSTERES (k):
  - Método del codo (Elbow): se busca el punto donde la inercia deja de
    decrecer significativamente al añadir clústeres.
  - Silhouette Score: mide la cohesión intra-clúster y la separación
    inter-clúster. S(i) = (b(i) - a(i)) / max(a(i), b(i)), donde a(i) es
    la distancia media al resto de puntos del mismo clúster y b(i) es la
    distancia media al clúster más cercano. Rango [-1, 1]; mayor es mejor.
  - Se reportan ambas métricas para k ∈ {2, 3, 4, 5, 6}.

PREPROCESADO PARA CLUSTERING:
  - El clustering se construye exclusivamente sobre 7 índices compuestos que
    resumen dimensiones latentes del comportamiento del consumidor.
  - Variables demográficas y conductuales (grupo_edad, sexo, gasto, compra reciente,
    frecuencia de uso de RRSS, etc.) no se utilizan para construir los clústeres,
    sino únicamente para caracterizar los segmentos obtenidos.
  - Los índices compuestos se estandarizan con StandardScaler para evitar que
    diferencias de escala alteren las distancias euclídeas de K-Means.
  Referencia: Kaufman & Rousseeuw (1990).

INTERPRETACIÓN DE CLÚSTERES:
  Tras el ajuste se calculan los perfiles medios por clúster para cada índice
  compuesto, permitiendo nombrar e interpretar cada segmento de consumidor.

═══════════════════════════════════════════════════════
BLOQUE B — CLASIFICACIÓN SUPERVISADA
═══════════════════════════════════════════════════════

Objetivo: predecir si un consumidor seguirá comprando moda influido por RRSS
(target_seguira_comprando_bin = 1 si puntuó ≥ 4 en escala Likert 1-5).

MODELOS COMPARADOS:
  MODELO A — Regresión Logística (baseline interpretable):
    Modelo lineal con función logit. Los coeficientes son interpretables como
    log-odds: un coeficiente positivo indica que al aumentar la variable aumenta
    la probabilidad de clase 1. Se usa class_weight='balanced' para compensar el
    desbalance de clases (clase 1 = 29.6% del total).
    Referencia: Cox (1958).

  MODELO B — Random Forest (ensemble avanzado):
    Ensemble de árboles CART con bagging. Captura no-linealidades e interacciones
    implícitas entre los índices compuestos. La feature importance revela qué
    constructos psicológicos son más predictivos de la intención de compra futura.
    Referencia: Breiman (2001).

MÉTRICAS (≥3 requeridas por rúbrica):
  - Accuracy:  proporción total de predicciones correctas.
  - F1-Score:  media armónica de Precision y Recall. Métrica principal dado el
               desbalance de clases (clase 1 = 29.6% → accuracy inflado).
  - ROC-AUC:   capacidad discriminativa. Indispensable con clases desbalanceadas.
  - Gap F1/AUC: diferencia train-test. Gap > 0.1 indica sobreajuste relevante.

SPLIT: 80/20 estratificado por clase (preserva distribución 70.4/29.6).
"""

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# =====================================================
# Rutas
# =====================================================
INPUT_CLUSTER = "data/input/capa3/capa3_clustering_ready.csv"
INPUT_SUPERV = "data/input/capa3/capa3_supervised_ready.csv"
OUTPUT_DIR = "data/analytic/capa3"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

for d in [FIGURES_DIR, METRICS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# =====================================================
# Constantes clustering y perfilado
# =====================================================
CLUSTER_FEATURES = [
    "indice_influencia_rrss",
    "indice_impulso_tendencia",
    "indice_confianza_influencers",
    "indice_escepticismo_influencers",
    "indice_difusion_fastfashion",
    "indice_postcompra",
    "indice_riesgo_arrepentimiento",
]

PROFILE_NUMERIC_VARS = [
    "freq_compra_anual",
    "tiempo_rrss_dia",
    "freq_contenido_moda_rrss",
    "sigue_influencers_moda",
    "compra_ult_6m_por_rrss_bin",
]

PROFILE_CATEGORICAL_VARS = [
    "grupo_edad",
    "sexo",
    "gasto_mensual_moda",
    "canal_compra_moda",
]

CLUSTER_LABELS = {
    0: "Perfil_baja_influencia_digital",
    1: "Perfil_alta_influencia_digital",
}

def relabel_clusters_by_influence(df_base: pd.DataFrame, cluster_labels, influence_col: str = "indice_influencia_rrss"):
    """
    Re-etiqueta los clústeres para que:
      cluster 0 = menor influencia RRSS
      cluster 1 = mayor influencia RRSS
    """
    df_tmp = df_base.copy()
    df_tmp["cluster_tmp"] = cluster_labels

    ordered_labels = (
        df_tmp.groupby("cluster_tmp")[influence_col]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    label_map = {old_label: new_label for new_label, old_label in enumerate(ordered_labels)}
    relabeled = pd.Series(cluster_labels).map(label_map).values
    return relabeled, label_map

# ══════════════════════════════════════════════════
# BLOQUE A — CLUSTERING K-MEANS
# ══════════════════════════════════════════════════
print("=" * 65)
print("CAPA 3 — BLOQUE A: CLUSTERING K-MEANS")
print("=" * 65)

df_cl_raw = pd.read_csv(INPUT_CLUSTER)

missing_cluster_cols = [c for c in CLUSTER_FEATURES if c not in df_cl_raw.columns]
if missing_cluster_cols:
    raise KeyError(f"Faltan variables de clustering en el input: {missing_cluster_cols}")

df_cl = df_cl_raw[["id_respuesta"] + CLUSTER_FEATURES].dropna().copy()
X_cl = df_cl[CLUSTER_FEATURES].copy()

scaler_cl = StandardScaler()
X_cl_dense = scaler_cl.fit_transform(X_cl)

print(f"\n[Dataset clustering]")
print(f"  n={len(df_cl)} | features={len(CLUSTER_FEATURES)}")
print(f"  Variables usadas para segmentación: {CLUSTER_FEATURES}")
print(f"  Tras escalado: {X_cl_dense.shape[1]} dimensiones")

k_values = range(2, 7)
k_results = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_cl_dense)
    sil = silhouette_score(X_cl_dense, labels)
    k_results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    print(f"  k={k} | Inercia={km.inertia_:.1f} | Silhouette={sil:.3f}")

k_df = pd.DataFrame(k_results)
k_df.to_csv(os.path.join(REPORTS_DIR, "capa3_clustering_k_selection.csv"), index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(k_df["k"], k_df["inertia"], marker="o", color="#1f77b4")
ax1.set_title("Método del Codo — Selección de k")
ax1.set_xlabel("Número de clústeres (k)")
ax1.set_ylabel("Inercia (suma distancias²)")
ax1.grid(True, alpha=0.3)

ax2.plot(k_df["k"], k_df["silhouette"], marker="o", color="#d62728")
ax2.set_title("Silhouette Score — Selección de k")
ax2.set_xlabel("Número de clústeres (k)")
ax2.set_ylabel("Silhouette Score")
ax2.grid(True, alpha=0.3)

plt.suptitle(
    "K-Means sobre índices compuestos — Selección del número óptimo de clústeres",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa3_kmeans_k_selection.png"), dpi=300)
plt.close()

best_row = k_df.sort_values("silhouette", ascending=False).iloc[0]
best_k = int(best_row["k"])
best_sil = float(best_row["silhouette"])

print(f"\n[Modelo final] k={best_k} (mejor Silhouette={best_sil:.3f})")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
cluster_labels = km_final.fit_predict(X_cl_dense)

cluster_labels, label_map = relabel_clusters_by_influence(
    df_base=df_cl,
    cluster_labels=cluster_labels,
    influence_col="indice_influencia_rrss",
)

df_cl["cluster"] = cluster_labels

profile = df_cl.groupby("cluster")[CLUSTER_FEATURES].mean().round(3)
profile.to_csv(os.path.join(REPORTS_DIR, "capa3_cluster_profiles.csv"))

print("\n[Perfil medio por clúster — índices de segmentación]")
print(profile.T.to_string())

df_profile = df_cl_raw.merge(
    df_cl[["id_respuesta", "cluster"]],
    on="id_respuesta",
    how="inner",
)

profile_numeric_cols = [c for c in PROFILE_NUMERIC_VARS if c in df_profile.columns]
if profile_numeric_cols:
    profile_numeric = df_profile.groupby("cluster")[profile_numeric_cols].mean().round(3)
    profile_numeric.to_csv(
        os.path.join(REPORTS_DIR, "capa3_cluster_profile_numeric_context.csv")
    )
    print("\n[Perfilado posterior — variables conductuales]")
    print(profile_numeric.T.to_string())

for cat_col in PROFILE_CATEGORICAL_VARS:
    if cat_col in df_profile.columns:
        ctab = pd.crosstab(
            df_profile["cluster"],
            df_profile[cat_col],
            normalize="index",
        ).round(3)
        ctab.to_csv(os.path.join(REPORTS_DIR, f"capa3_cluster_profile_{cat_col}.csv"))

sizes = df_cl["cluster"].value_counts().sort_index()
sizes_df = sizes.reset_index()
sizes_df.columns = ["cluster", "n_obs"]
sizes_df["pct"] = (sizes_df["n_obs"] / len(df_cl) * 100).round(1)
sizes_df.to_csv(os.path.join(REPORTS_DIR, "capa3_cluster_sizes.csv"), index=False)

print("\n[Tamaño de clústeres]")
print(sizes_df)

plot_profile = df_cl.groupby("cluster")[CLUSTER_FEATURES].mean()
plot_profile.T.plot(kind="bar", figsize=(12, 6), colormap="Set1")
plt.title(f"Perfil medio de índices por clúster (k={best_k})")
plt.xlabel("Índice compuesto")
plt.ylabel("Valor medio (escala 1-5)")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Clúster")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa3_cluster_profiles_indices.png"), dpi=300)
plt.close()

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cl_dense)
explained = pca.explained_variance_ratio_
centroids_pca = pca.transform(km_final.cluster_centers_)

plt.figure(figsize=(10, 7))
colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

for c in range(best_k):
    mask = cluster_labels == c
    label_name = CLUSTER_LABELS.get(c, f"Clúster {c}")
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        color=colors[c % len(colors)],
        alpha=0.6,
        label=f"C{c}: {label_name}",
        s=40,
    )

plt.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    marker="X",
    s=300,
    color="black",
    zorder=5,
    label="Centroides",
)

plt.title(
    f"K-Means k={best_k} — PCA 2D sobre índices compuestos\n"
    f"Varianza explicada: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}"
)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "capa3_kmeans_pca_2d.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

df_profile.to_csv(os.path.join(REPORTS_DIR, "capa3_cluster_assignments.csv"), index=False)

cluster_metrics = {
    "k_optimo": best_k,
    "silhouette_score": best_sil,
    "inercia": float(km_final.inertia_),
    "n_obs": len(df_cl),
    "variables_segmentacion": CLUSTER_FEATURES,
    "variables_solo_perfilado": PROFILE_NUMERIC_VARS + PROFILE_CATEGORICAL_VARS,
    "distribucion_clusters": sizes_df.to_dict(orient="records"),
    "interpretacion_clusters": CLUSTER_LABELS,
    "interpretacion": (
        f"El análisis K-Means identifica {best_k} perfiles diferenciados de consumidor "
        f"construidos exclusivamente a partir de 7 índices compuestos que sintetizan la "
        f"influencia de las RRSS, el impulso de compra, la confianza en influencers, la "
        f"difusión del fast fashion, la experiencia postcompra y el riesgo de arrepentimiento. "
        f"El Silhouette Score de {best_sil:.3f} indica una separación "
        f"{'buena' if best_sil > 0.3 else 'moderada'} entre clústeres. "
        f"Las variables demográficas y de conducta se reservan para la caracterización posterior "
        f"de los segmentos, evitando que dominen artificialmente la segmentación."
    ),
    "limitaciones": (
        "K-Means asume clústeres aproximadamente esféricos y de tamaño relativamente homogéneo. "
        "Con n=318 la solución sigue siendo exploratoria. La muestra de conveniencia limita la "
        "generalización de los perfiles identificados."
    ),
}

with open(
    os.path.join(METRICS_DIR, "capa3_clustering_metrics.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(cluster_metrics, f, ensure_ascii=False, indent=4)

# ══════════════════════════════════════════════════
# BLOQUE B — CLASIFICACIÓN SUPERVISADA
# ══════════════════════════════════════════════════
print("\n" + "=" * 65)
print("CAPA 3 — BLOQUE B: CLASIFICACIÓN SUPERVISADA")
print("=" * 65)

df_sv = pd.read_csv(INPUT_SUPERV)

TARGET = "target_seguira_comprando_bin"
DROP = ["id_respuesta", "target_recomendaria_bin", TARGET]

X_sv = df_sv.drop(columns=DROP)
y_sv = df_sv[TARGET].astype(int)

CAT_SV = ["grupo_edad", "sexo", "gasto_mensual_moda"]
NUM_SV = [c for c in X_sv.columns if c not in CAT_SV]

print(f"\n[Dataset supervisado]")
print(f"  n={len(df_sv)} | features={X_sv.shape[1]}")
print(f"  Clase 1 (seguirá comprando): {y_sv.sum()} ({y_sv.mean()*100:.1f}%)")
print(f"  Clase 0 (no seguirá):         {(y_sv == 0).sum()} ({(1-y_sv.mean())*100:.1f}%)")

numeric_tr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_tr = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor_sv = ColumnTransformer([
    ("num", numeric_tr, NUM_SV),
    ("cat", cat_tr, CAT_SV),
])

X_tr, X_te, y_tr, y_te = train_test_split(
    X_sv,
    y_sv,
    test_size=0.2,
    random_state=42,
    stratify=y_sv,
)

print(f"\n[Split 80/20 estratificado]")
print(f"  Train: {X_tr.shape[0]} | Test: {X_te.shape[0]}")

sv_models = {
    "A_Logistic_Regression": Pipeline([
        ("preprocessor", preprocessor_sv),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ]),
    "B_Random_Forest": Pipeline([
        ("preprocessor", preprocessor_sv),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_split=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        )),
    ]),
}

sv_results = []

for mname, mpipe in sv_models.items():
    mpipe.fit(X_tr, y_tr)

    y_tr_pred = mpipe.predict(X_tr)
    y_tr_prob = mpipe.predict_proba(X_tr)[:, 1]
    y_te_pred = mpipe.predict(X_te)
    y_te_prob = mpipe.predict_proba(X_te)[:, 1]

    res = {
        "model": mname,
        "train_accuracy": accuracy_score(y_tr, y_tr_pred),
        "train_f1": f1_score(y_tr, y_tr_pred, zero_division=0),
        "train_auc": roc_auc_score(y_tr, y_tr_prob),
        "test_accuracy": accuracy_score(y_te, y_te_pred),
        "test_precision": precision_score(y_te, y_te_pred, zero_division=0),
        "test_recall": recall_score(y_te, y_te_pred, zero_division=0),
        "test_f1": f1_score(y_te, y_te_pred, zero_division=0),
        "test_auc": roc_auc_score(y_te, y_te_prob),
        "gap_f1": f1_score(y_tr, y_tr_pred, zero_division=0) - f1_score(y_te, y_te_pred, zero_division=0),
        "gap_auc": roc_auc_score(y_tr, y_tr_prob) - roc_auc_score(y_te, y_te_prob),
    }
    sv_results.append(res)

    print(f"\n  [{mname}]")
    print(f"  Train → F1={res['train_f1']:.3f} | AUC={res['train_auc']:.3f}")
    print(
        f"  Test  → Acc={res['test_accuracy']:.3f} | Prec={res['test_precision']:.3f} | "
        f"Rec={res['test_recall']:.3f} | F1={res['test_f1']:.3f} | AUC={res['test_auc']:.3f}"
    )
    gap_flag = "⚠ SOBREAJUSTE" if res["gap_f1"] > 0.1 else "✓ OK"
    print(f"  Gap F1={res['gap_f1']:.3f} | Gap AUC={res['gap_auc']:.3f} → {gap_flag}")

    cm = confusion_matrix(y_te, y_te_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No seguirá", "Seguirá"],
    )
    disp.plot(colorbar=False)
    plt.title(f"Matriz de Confusión — {mname}\n(target: seguirá comprando influido por RRSS)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"capa3_{mname}_confusion_matrix.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_te, y_te_prob, ax=ax, name=mname)
    ax.plot([0, 1], [0, 1], "k--", label="Clasificador aleatorio")
    ax.set_title(f"Curva ROC — {mname}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"capa3_{mname}_roc_curve.png"), dpi=300)
    plt.close()

rf_pipe = sv_models["B_Random_Forest"]
feat_names_sv = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
importances_sv = rf_pipe.named_steps["classifier"].feature_importances_

imp_sv = pd.DataFrame({
    "feature": feat_names_sv,
    "importance": importances_sv,
}).sort_values("importance", ascending=False)

imp_sv.to_csv(os.path.join(REPORTS_DIR, "capa3_rf_feature_importance.csv"), index=False)

plt.figure(figsize=(10, 7))
top_sv = imp_sv.head(12)
plt.barh(top_sv["feature"][::-1], top_sv["importance"][::-1], color="#2196F3")
plt.title("Feature Importance — Random Forest | Predicción Intención de Compra Futura")
plt.xlabel("Importancia (Gini)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa3_rf_feature_importance.png"), dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], "k--", label="Clasificador aleatorio (AUC=0.50)")
for mname, mpipe in sv_models.items():
    y_prob = mpipe.predict_proba(X_te)[:, 1]
    RocCurveDisplay.from_predictions(y_te, y_prob, ax=ax, name=mname)
ax.set_title("Curva ROC Comparativa — Intención de Compra Futura")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa3_roc_comparativa.png"), dpi=300)
plt.close()

sv_results_df = pd.DataFrame(sv_results).sort_values("test_auc", ascending=False)
sv_results_df.to_csv(os.path.join(METRICS_DIR, "capa3_supervised_metrics.csv"), index=False)

with open(
    os.path.join(METRICS_DIR, "capa3_supervised_metrics.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(sv_results, f, ensure_ascii=False, indent=4)

winner = sv_results_df.iloc[0]
top_feature = imp_sv.iloc[0]["feature"] if not imp_sv.empty else "n/d"

sv_interpretation = {
    "modelo_ganador": winner["model"],
    "criterio": "mayor_test_AUC",
    "test_auc": winner["test_auc"],
    "test_f1": winner["test_f1"],
    "test_accuracy": winner["test_accuracy"],
    "gap_f1": winner["gap_f1"],
    "gap_auc": winner["gap_auc"],
    "sobreajuste": "si" if winner["gap_f1"] > 0.1 else "no",
    "feature_mas_importante": top_feature,
    "interpretacion": (
        f"El modelo {winner['model']} obtiene el mayor AUC en test ({winner['test_auc']:.3f}). "
        f"La variable más predictiva de la intención de compra futura es '{top_feature}'. "
        f"El F1-Score de {winner['test_f1']:.3f} refleja la capacidad del modelo para "
        f"detectar consumidores con alta intención de compra (clase minoritaria, 29.6%). "
        f"La curva ROC y la matriz de confusión permiten visualizar el equilibrio entre "
        f"precision y recall en el umbral de clasificación por defecto (0.5)."
    ),
    "limitaciones": (
        "El target es una pregunta declarativa ('seguiré comprando'), lo que introduce "
        "sesgo de deseabilidad social. Con n=318 y target desbalanceado (29.6% clase 1), "
        "los modelos pueden ser sensibles al conjunto de test (64 observaciones). "
        "Se recomienda validación cruzada en trabajos futuros con mayor muestra."
    ),
}

with open(
    os.path.join(REPORTS_DIR, "capa3_supervisado_interpretacion.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(sv_interpretation, f, ensure_ascii=False, indent=4)

print("\n" + "=" * 65)
print("CAPA 3 — COMPLETADO")
print(f"\n  CLUSTERING:")
print(f"    k óptimo={best_k} | Silhouette={best_sil:.3f}")
print(f"  SUPERVISADO:")
print(f"    Modelo A: Logistic Regression | AUC={sv_results[0]['test_auc']:.3f} | F1={sv_results[0]['test_f1']:.3f}")
print(f"    Modelo B: Random Forest       | AUC={sv_results[1]['test_auc']:.3f} | F1={sv_results[1]['test_f1']:.3f}")
print(f"    Ganador:  {winner['model']}")
print("Outputs:")
print("  figures/ → kmeans_k_selection, pca_2d, cluster_profiles,")
print("             confusion_matrix, roc_curve, feature_importance, roc_comparativa")
print("  metrics/ → capa3_clustering_metrics.json, capa3_supervised_metrics.csv/.json")
print("  reports/ → cluster_assignments, profiles, supervisado_interpretacion")
print("=" * 65)