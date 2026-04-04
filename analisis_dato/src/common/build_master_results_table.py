"""
build_master_results_table.py — Tabla maestra de resultados del análisis del dato
==================================================================================

Consolida en una única tabla comparativa todos los modelos entrenados en las
tres capas del TFG. Permite presentar una visión global del análisis para la
memoria y la defensa oral.

ESTRUCTURA DE MODELOS:
  Capa 1: Series temporales retail moda
    - Holt-Winters ETS-AAA (baseline) vs SARIMA(0,1,1)(0,1,0,12)
  Capa 2A: Clasificación engagement Instagram
    - Logistic Regression vs Random Forest (features estructuradas)
  Capa 2B: Text Mining captions Instagram
    - TF-IDF + Logistic Regression vs TF-IDF + Random Forest
  Capa 2C: Series temporales Google Trends Zara
    - ARIMA(2,1,2) (único modelo — el tune ya seleccionó el mejor)
  Capa 3A: Clustering consumidores
    - K-Means k óptimo (seleccionado por Silhouette)
  Capa 3B: Clasificación intención de compra futura
    - Logistic Regression vs Random Forest
"""

import os
import json
import pandas as pd
import re
import numpy as np

# =====================================================
# Rutas
# =====================================================
BASE_DIR   = "data/analytic"
OUTPUT_DIR = os.path.join(BASE_DIR, "summary")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_read_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

def safe_read_json(path):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

rows = []

# =====================================================
# CAPA 1 — Comparativa Holt-Winters vs SARIMA
# =====================================================
c1_comp = safe_read_csv(os.path.join(BASE_DIR, "capa1", "metrics", "capa1_modelos_comparativa.csv"))
if c1_comp is not None:
    for _, r in c1_comp.iterrows():
        rows.append({
            "capa":              "Capa 1",
            "bloque":            "Series temporales — Retail moda",
            "tipo_modelo":       "regresion_temporal",
            "objetivo":          "Predecir evolución del índice retail moda (base 2015=100)",
            "dataset":           "capa1_master_mensual_analysis.csv",
            "modelo":            r.get("model", ""),
            "tipo":              "Baseline" if "Holt" in str(r.get("model","")) else "Avanzado",
            "metricas_clave":    f"MAE={r.get('mae',0):.3f} | RMSE={r.get('rmse',0):.3f} | MAPE={r.get('mape',0):.2f}%",
            "mae":               r.get("mae", None),
            "rmse":              r.get("rmse", None),
            "mape":              r.get("mape", None),
            "test_f1":           None,
            "test_auc":          None,
            "gap_f1":            None,
            "silhouette":        None,
            "interpretacion":    "Serie temporal con estacionalidad mensual. Captura patrón rebajas y navidad.",
            "estado":            "Final",
        })
else:
    # Fallback al SARIMA original si no se ejecutó el nuevo script
    c1_old = safe_read_csv(os.path.join(BASE_DIR, "capa1", "metrics", "capa1_sarima_metrics.csv"))
    if c1_old is not None and len(c1_old) > 0:
        r = c1_old.iloc[0]
        rows.append({
            "capa": "Capa 1", "bloque": "Series temporales — Retail moda",
            "tipo_modelo": "regresion_temporal",
            "objetivo": "Predecir evolución del índice retail moda",
            "dataset": "capa1_master_mensual_analysis.csv",
            "modelo": r.get("model", "SARIMA"),
            "tipo": "Avanzado",
            "metricas_clave": f"MAE={r.get('mae',0):.3f} | RMSE={r.get('rmse',0):.3f} | MAPE={r.get('mape',0):.2f}%",
            "mae": r.get("mae"), "rmse": r.get("rmse"), "mape": r.get("mape"),
            "test_f1": None, "test_auc": None, "gap_f1": None, "silhouette": None,
            "interpretacion": "Modelo SARIMA ajustado por grid search.",
            "estado": "Final",
        })

# =====================================================
# CAPA 2A — Clasificación engagement Instagram
# =====================================================
c2a = safe_read_csv(os.path.join(BASE_DIR, "capa2", "metrics", "capa2_instagram_modelos_comparativa.csv"))
if c2a is not None:
    for _, r in c2a.iterrows():
        rows.append({
            "capa":           "Capa 2",
            "bloque":         "2A — Clasificación engagement Instagram",
            "tipo_modelo":    "clasificacion_supervisada",
            "objetivo":       "Predecir publicaciones con alto engagement (> P75)",
            "dataset":        "instagram_model_input.csv",
            "modelo":         r.get("model", ""),
            "tipo":           "Baseline" if "Logistic" in str(r.get("model","")) else "Avanzado",
            "metricas_clave": f"Acc={r.get('test_accuracy',0):.3f} | F1={r.get('test_f1',0):.3f} | AUC={r.get('test_auc',0):.3f} | Gap_F1={r.get('gap_f1',0):.3f}",
            "mae": None, "rmse": None, "mape": None,
            "test_f1":    r.get("test_f1"),
            "test_auc":   r.get("test_auc"),
            "gap_f1":     r.get("gap_f1"),
            "silhouette": None,
            "interpretacion": "Predice alto engagement a partir de marca, tipo de post y estacionalidad.",
            "estado": "Final",
        })
else:
    # Fallback a versión original
    for fname in ["instagram_models_metrics.csv", "instagram_models_metrics_v2.csv"]:
        old = safe_read_csv(os.path.join(BASE_DIR, "capa2", "metrics", fname))
        if old is not None:
            for _, r in old.iterrows():
                rows.append({
                    "capa": "Capa 2", "bloque": "2A — Clasificación engagement Instagram",
                    "tipo_modelo": "clasificacion_supervisada",
                    "objetivo": "Predecir publicaciones con alto engagement",
                    "dataset": "instagram_model_input.csv",
                    "modelo": r.get("model", ""), "tipo": "Original",
                    "metricas_clave": f"F1={r.get('f1_score',0):.3f} | AUC={r.get('roc_auc',0):.3f}",
                    "mae": None, "rmse": None, "mape": None,
                    "test_f1": r.get("f1_score"), "test_auc": r.get("roc_auc"),
                    "gap_f1": None, "silhouette": None,
                    "interpretacion": "Versión original.", "estado": "Legacy",
                })

# =====================================================
# CAPA 2B — Text Mining captions Instagram
# =====================================================
c2b_text = safe_read_csv(os.path.join(BASE_DIR, "capa2", "metrics", "capa2_text_modelos_comparativa.csv"))
if c2b_text is not None:
    for _, r in c2b_text.iterrows():
        rows.append({
            "capa":           "Capa 2",
            "bloque":         "2B — Text Mining captions Instagram",
            "tipo_modelo":    "text_mining_clasificacion",
            "objetivo":       "Predecir alto engagement a partir del texto del caption (TF-IDF)",
            "dataset":        "instagram_posts_clean.csv (captions)",
            "modelo":         r.get("model", ""),
            "tipo":           "Baseline" if "Logistic" in str(r.get("model","")) else "Avanzado",
            "metricas_clave": f"Acc={r.get('test_accuracy',0):.3f} | F1={r.get('test_f1',0):.3f} | AUC={r.get('test_auc',0):.3f}",
            "mae": None, "rmse": None, "mape": None,
            "test_f1":    r.get("test_f1"),
            "test_auc":   r.get("test_auc"),
            "gap_f1":     r.get("gap_f1"),
            "silhouette": None,
            "interpretacion": "TF-IDF + modelo. Identifica términos predictivos del engagement.",
            "estado": "Final",
        })

# =====================================================
# CAPA 2C — ARIMA Google Trends Zara
# =====================================================
c2c = safe_read_csv(os.path.join(BASE_DIR, "capa2", "metrics", "zara_arima_final_metrics.csv"))
if c2c is not None and len(c2c) > 0:
    r = c2c.iloc[0]
    rows.append({
        "capa":           "Capa 2",
        "bloque":         "2C — Series temporales Google Trends Zara",
        "tipo_modelo":    "regresion_temporal",
        "objetivo":       "Predecir interés de búsqueda de Zara (Google Trends)",
        "dataset":        "trends_marcas_clean.csv",
        "modelo":         r.get("model", "ARIMA"),
        "tipo":           "Final",
        "metricas_clave": f"MAE={r.get('mae',0):.3f} | RMSE={r.get('rmse',0):.3f} | MAPE={r.get('mape',0):.2f}%",
        "mae": r.get("mae"), "rmse": r.get("rmse"), "mape": r.get("mape"),
        "test_f1": None, "test_auc": None, "gap_f1": None, "silhouette": None,
        "interpretacion": "ARIMA seleccionado por menor RMSE en test (2025). Captura tendencia del interés digital.",
        "estado": "Final",
    })

# =====================================================
# CAPA 3A — Clustering K-Means
# =====================================================
c3a_json = safe_read_json(os.path.join(BASE_DIR, "capa3", "metrics", "capa3_clustering_metrics.json"))
c3a_old  = safe_read_csv(os.path.join(BASE_DIR, "capa3", "reports", "capa3_clustering_k_selection_main.csv"))

if c3a_json is not None:
    rows.append({
        "capa":           "Capa 3",
        "bloque":         "3A — Clustering consumidores",
        "tipo_modelo":    "clustering_no_supervisado",
        "objetivo":       "Identificar perfiles naturales de consumidor de moda en RRSS",
        "dataset":        "capa3_clustering_ready.csv",
        "modelo":         f"KMeans k={c3a_json.get('k_optimo', '?')}",
        "tipo":           "No supervisado",
        "metricas_clave": f"Silhouette={c3a_json.get('silhouette_score',0):.3f} | Inercia={c3a_json.get('inercia',0):.1f}",
        "mae": None, "rmse": None, "mape": None,
        "test_f1": None, "test_auc": None, "gap_f1": None,
        "silhouette": c3a_json.get("silhouette_score"),
        "interpretacion": c3a_json.get("interpretacion", ""),
        "estado": "Final",
    })
elif c3a_old is not None and len(c3a_old) > 0:
    best = c3a_old.sort_values("silhouette", ascending=False).iloc[0]
    rows.append({
        "capa": "Capa 3", "bloque": "3A — Clustering consumidores",
        "tipo_modelo": "clustering_no_supervisado",
        "objetivo": "Identificar perfiles de consumidor",
        "dataset": "capa3_clustering_ready.csv",
        "modelo": f"KMeans k={int(best['k'])}", "tipo": "No supervisado",
        "metricas_clave": f"Silhouette={best['silhouette']:.3f} | Inercia={best['inertia']:.1f}",
        "mae": None, "rmse": None, "mape": None,
        "test_f1": None, "test_auc": None, "gap_f1": None,
        "silhouette": best["silhouette"],
        "interpretacion": "Segmentación K-Means. Perfiles diferenciados por índices de influencia digital.",
        "estado": "Final",
    })

# =====================================================
# CAPA 3B — Clasificación supervisada
# =====================================================
c3b = safe_read_csv(os.path.join(BASE_DIR, "capa3", "metrics", "capa3_supervised_metrics.csv"))
if c3b is None:
    c3b = safe_read_csv(os.path.join(BASE_DIR, "capa3", "metrics", "capa3_supervised_metrics_main.csv"))

if c3b is not None:
    for _, r in c3b.iterrows():
        rows.append({
            "capa":           "Capa 3",
            "bloque":         "3B — Clasificación intención de compra futura",
            "tipo_modelo":    "clasificacion_supervisada",
            "objetivo":       "Predecir si el consumidor seguirá comprando influido por RRSS",
            "dataset":        "capa3_supervised_ready.csv",
            "modelo":         r.get("model", ""),
            "tipo":           "Baseline" if "Logistic" in str(r.get("model","")) else "Avanzado",
            "metricas_clave": f"Acc={r.get('test_accuracy',0):.3f} | F1={r.get('test_f1',0):.3f} | AUC={r.get('test_auc',0):.3f} | Gap_F1={r.get('gap_f1',0):.3f}",
            "mae": None, "rmse": None, "mape": None,
            "test_f1":    r.get("test_f1"),
            "test_auc":   r.get("test_auc"),
            "gap_f1":     r.get("gap_f1"),
            "silhouette": None,
            "interpretacion": "Predice intención de compra futura a partir de índices psicológicos e índices de influencia RRSS.",
            "estado": "Final",
        })

# =====================================================
# Guardar tabla maestra
# =====================================================
def extract_metric(metricas_clave: str, metric_name: str):
    """
    Extrae una métrica numérica desde strings del tipo:
    'Acc=0.822 | F1=0.656 | AUC=0.852'
    'MAE=5.978 | RMSE=8.953 | MAPE=4.75%'
    'Silhouette=0.353 | Inercia=1333.0'
    """
    if pd.isna(metricas_clave):
        return np.nan

    pattern = rf"{metric_name}=(-?\d+(?:\.\d+)?)"
    match = re.search(pattern, str(metricas_clave))
    return float(match.group(1)) if match else np.nan

master_df = pd.DataFrame(rows)

# =====================================================
# Ordenar ganadores por bloque
# =====================================================
master_df["auc"] = master_df["metricas_clave"].apply(lambda x: extract_metric(x, "AUC"))
master_df["f1"] = master_df["metricas_clave"].apply(lambda x: extract_metric(x, "F1"))
master_df["rmse"] = master_df["metricas_clave"].apply(lambda x: extract_metric(x, "RMSE"))
master_df["mape"] = master_df["metricas_clave"].apply(lambda x: extract_metric(x, "MAPE"))
master_df["silhouette"] = master_df["metricas_clave"].apply(lambda x: extract_metric(x, "Silhouette"))

master_df["sort_score"] = np.nan

# Clasificación y text mining -> mayor AUC es mejor
mask_auc = master_df["bloque"].str.contains("Clasificación|Text Mining", case=False, na=False)
master_df.loc[mask_auc, "sort_score"] = master_df.loc[mask_auc, "auc"]

# Series temporales -> menor RMSE es mejor
mask_rmse = master_df["bloque"].str.contains("Series temporales", case=False, na=False)
master_df.loc[mask_rmse, "sort_score"] = -master_df.loc[mask_rmse, "rmse"]

# Clustering -> mayor Silhouette es mejor
mask_sil = master_df["bloque"].str.contains("Clustering", case=False, na=False)
master_df.loc[mask_sil, "sort_score"] = master_df.loc[mask_sil, "silhouette"]

# Marcar ganador dentro de cada bloque
master_df["es_ganador"] = (
    master_df.groupby(["capa", "bloque"])["sort_score"]
    .transform(lambda s: s == s.max())
    .fillna(False)
)

# Marcar visualmente el ganador en la columna tipo
master_df["tipo"] = np.where(
    master_df["es_ganador"] & ~master_df["tipo"].astype(str).str.contains("Ganador", case=False, na=False),
    master_df["tipo"].astype(str) + " (Ganador)",
    master_df["tipo"]
)

# Orden final: ganador arriba dentro de cada bloque
master_df = master_df.sort_values(
    by=["capa", "bloque", "es_ganador", "sort_score", "modelo"],
    ascending=[True, True, False, False, True]
).reset_index(drop=True)

# Exportar sin columnas auxiliares
export_df = master_df.drop(
    columns=["auc", "f1", "rmse", "mape", "silhouette", "sort_score"],
    errors="ignore"
)

export_df.to_csv(os.path.join(OUTPUT_DIR, "master_results_table.csv"), index=False)
export_df.to_excel(os.path.join(OUTPUT_DIR, "master_results_table.xlsx"), index=False)

# Resumen ejecutivo
print("\n" + "=" * 65)
print("TABLA MAESTRA DE RESULTADOS — ANÁLISIS DEL DATO")
print("=" * 65)
display_cols = ["capa", "bloque", "modelo", "tipo", "metricas_clave", "estado"]
print(export_df[display_cols].to_string(index=False))

print(f"\nTotal de modelos documentados: {len(export_df)}")
print(f"\nGuardada en:")
print(f"  {os.path.join(OUTPUT_DIR, 'master_results_table.csv')}")
print(f"  {os.path.join(OUTPUT_DIR, 'master_results_table.xlsx')}")