import os
import pandas as pd

# =====================================================
# Rutas
# =====================================================
BASE_DIR = "analisis_dato/data/analytic"
OUTPUT_DIR = os.path.join(BASE_DIR, "summary")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# Helpers
# =====================================================
def safe_read_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

rows = []

# =====================================================
# CAPA 1
# =====================================================
c1_metrics = safe_read_csv(os.path.join(BASE_DIR, "capa1", "metrics", "capa1_sarima_metrics.csv"))
if c1_metrics is not None and len(c1_metrics) > 0:
    r = c1_metrics.iloc[0]
    rows.append({
        "capa": "Capa 1",
        "subcapa": "Macro temporal retail moda",
        "objetivo": "Predecir la evolución del índice retail moda",
        "dataset": "capa1_master_mensual_analysis.csv",
        "modelo": r.get("model", "SARIMA"),
        "muestra": "Principal",
        "metricas_clave": f"MAE={r.get('mae', ''):.3f} | RMSE={r.get('rmse', ''):.3f} | MAPE={r.get('mape', ''):.3f}",
        "resultado_principal": "Modelo temporal final seleccionado tras tuning",
        "interpretacion_breve": "Capta la evolución temporal del retail moda y sirve como marco macro del consumo.",
        "estado": "Final"
    })

# =====================================================
# CAPA 2A - Instagram
# =====================================================
c2a_main = safe_read_csv(os.path.join(BASE_DIR, "capa2", "metrics", "instagram_models_metrics.csv"))
if c2a_main is not None:
    for _, r in c2a_main.iterrows():
        rows.append({
            "capa": "Capa 2",
            "subcapa": "2A - Instagram engagement",
            "objetivo": "Predecir publicaciones con alto engagement",
            "dataset": "instagram_model_input.csv",
            "modelo": r.get("model", ""),
            "muestra": "Principal",
            "metricas_clave": f"Accuracy={r.get('accuracy', ''):.3f} | F1={r.get('f1_score', ''):.3f} | ROC_AUC={r.get('roc_auc', ''):.3f}",
            "resultado_principal": "Comparación de modelos supervisados para engagement alto",
            "interpretacion_breve": "Permite identificar patrones de publicación asociados a mayor respuesta del público.",
            "estado": "Final"
        })

c2a_v2 = safe_read_csv(os.path.join(BASE_DIR, "capa2", "metrics", "instagram_models_metrics_v2.csv"))
if c2a_v2 is not None:
    for _, r in c2a_v2.iterrows():
        rows.append({
            "capa": "Capa 2",
            "subcapa": "2A - Instagram engagement",
            "objetivo": "Refinar la predicción de publicaciones con alto engagement",
            "dataset": "instagram_model_input.csv",
            "modelo": r.get("model", ""),
            "muestra": "Iteración adicional",
            "metricas_clave": f"Accuracy={r.get('accuracy', ''):.3f} | F1={r.get('f1_score', ''):.3f} | ROC_AUC={r.get('roc_auc', ''):.3f}",
            "resultado_principal": "Segunda iteración de comparación",
            "interpretacion_breve": "Sirve para contrastar estabilidad del rendimiento y robustez del enfoque.",
            "estado": "Complementario"
        })

# =====================================================
# CAPA 2B - Google Trends
# =====================================================
c2b_metrics = safe_read_csv(os.path.join(BASE_DIR, "capa2", "metrics", "zara_arima_final_metrics.csv"))
if c2b_metrics is not None and len(c2b_metrics) > 0:
    r = c2b_metrics.iloc[0]
    rows.append({
        "capa": "Capa 2",
        "subcapa": "2B - Google Trends Zara",
        "objetivo": "Predecir la evolución temporal del interés de búsqueda de Zara",
        "dataset": "trends_marcas_clean.csv",
        "modelo": r.get("model", "ARIMA"),
        "muestra": "Principal",
        "metricas_clave": f"MAE={r.get('mae', ''):.3f} | RMSE={r.get('rmse', ''):.3f} | MAPE={r.get('mape', ''):.3f}",
        "resultado_principal": "Modelo ARIMA final para Zara",
        "interpretacion_breve": "Aproxima la persistencia del interés digital por una marca líder de moda.",
        "estado": "Final"
    })

# =====================================================
# CAPA 3 - Supervisado main
# =====================================================
c3_main = safe_read_csv(os.path.join(BASE_DIR, "capa3", "metrics", "capa3_supervised_metrics_main.csv"))
if c3_main is not None:
    for _, r in c3_main.iterrows():
        rows.append({
            "capa": "Capa 3",
            "subcapa": "Supervisado consumidor",
            "objetivo": "Predecir si el consumidor seguirá comprando",
            "dataset": "capa3_supervised_ready.csv",
            "modelo": r.get("model", ""),
            "muestra": "Principal",
            "metricas_clave": f"Test_F1={r.get('test_f1', ''):.3f} | Test_AUC={r.get('test_auc', ''):.3f} | Gap_F1={r.get('gap_f1', ''):.3f}",
            "resultado_principal": "Evaluación principal sobre muestra observada",
            "interpretacion_breve": "Contrasta capacidad predictiva y estabilidad de modelos sobre comportamiento futuro declarado.",
            "estado": "Final"
        })

# =====================================================
# CAPA 3 - Supervisado balanced
# =====================================================
c3_bal = safe_read_csv(os.path.join(BASE_DIR, "capa3", "metrics", "capa3_supervised_metrics_balanced.csv"))
if c3_bal is not None:
    for _, r in c3_bal.iterrows():
        rows.append({
            "capa": "Capa 3",
            "subcapa": "Supervisado consumidor",
            "objetivo": "Comprobar robustez del modelo al equilibrar generaciones",
            "dataset": "capa3_supervised_balanced_generation.csv",
            "modelo": r.get("model", ""),
            "muestra": "Balanceada por generación",
            "metricas_clave": f"Test_F1={r.get('test_f1', ''):.3f} | Test_AUC={r.get('test_auc', ''):.3f} | Gap_F1={r.get('gap_f1', ''):.3f}",
            "resultado_principal": "Análisis de sensibilidad por estructura generacional",
            "interpretacion_breve": "Permite comprobar cuánto dependen los resultados del peso de Gen Z en la muestra.",
            "estado": "Robustez"
        })

# =====================================================
# CAPA 3 - Clustering main
# =====================================================
c3_k_main = safe_read_csv(os.path.join(BASE_DIR, "capa3", "reports", "capa3_clustering_k_selection_main.csv"))
if c3_k_main is not None and len(c3_k_main) > 0:
    best = c3_k_main.sort_values("silhouette", ascending=False).iloc[0]
    rows.append({
        "capa": "Capa 3",
        "subcapa": "Clustering consumidor",
        "objetivo": "Identificar perfiles de consumidor",
        "dataset": "capa3_clustering_ready.csv",
        "modelo": "KMeans",
        "muestra": "Principal",
        "metricas_clave": f"k={int(best['k'])} | Silhouette={best['silhouette']:.3f} | Inercia={best['inertia']:.3f}",
        "resultado_principal": "Solución final de 2 clústeres",
        "interpretacion_breve": "Distingue entre consumidor digital influenciado y perfil más moderado de baja influencia digital.",
        "estado": "Final"
    })

# =====================================================
# CAPA 3 - Clustering balanced
# =====================================================
c3_k_bal = safe_read_csv(os.path.join(BASE_DIR, "capa3", "reports", "capa3_clustering_k_selection_balanced.csv"))
if c3_k_bal is not None and len(c3_k_bal) > 0:
    best = c3_k_bal.sort_values("silhouette", ascending=False).iloc[0]
    rows.append({
        "capa": "Capa 3",
        "subcapa": "Clustering consumidor",
        "objetivo": "Comprobar robustez de la segmentación al equilibrar generaciones",
        "dataset": "capa3_clustering_balanced_generation.csv",
        "modelo": "KMeans",
        "muestra": "Balanceada por generación",
        "metricas_clave": f"k={int(best['k'])} | Silhouette={best['silhouette']:.3f} | Inercia={best['inertia']:.3f}",
        "resultado_principal": "La solución k=2 se mantiene al equilibrar la muestra",
        "interpretacion_breve": "Refuerza la solidez de la segmentación y reduce el riesgo de sesgo por sobrerrepresentación generacional.",
        "estado": "Robustez"
    })

# =====================================================
# CAPA 3 - K=3 exploratorio
# =====================================================
k3_main_sizes = safe_read_csv(os.path.join(BASE_DIR, "capa3", "reports", "capa3_clustering_sizes_k3_exploratory_main.csv"))
if k3_main_sizes is not None:
    rows.append({
        "capa": "Capa 3",
        "subcapa": "Clustering consumidor",
        "objetivo": "Explorar una subdivisión adicional de perfiles",
        "dataset": "capa3_clustering_ready.csv",
        "modelo": "KMeans k=3",
        "muestra": "Exploratorio",
        "metricas_clave": "Análisis visual en PCA 2D + tamaños de clúster",
        "resultado_principal": "Sugiere perfiles bajo, intermedio y alto de influencia digital",
        "interpretacion_breve": "Útil como exploración, aunque se mantiene k=2 como solución final por parsimonia y respaldo cuantitativo.",
        "estado": "Exploratorio"
    })

# =====================================================
# Guardado
# =====================================================
master_df = pd.DataFrame(rows)

master_df.to_csv(os.path.join(OUTPUT_DIR, "master_results_table.csv"), index=False)
master_df.to_excel(os.path.join(OUTPUT_DIR, "master_results_table.xlsx"), index=False)

print("\n--- TABLA MAESTRA DE RESULTADOS ---")
print(master_df)

print("\nGuardada en:")
print(os.path.join(OUTPUT_DIR, "master_results_table.csv"))
print(os.path.join(OUTPUT_DIR, "master_results_table.xlsx"))