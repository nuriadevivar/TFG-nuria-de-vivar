import sqlite3
import pandas as pd
import numpy as np

from src.common.config import (
    DB_CAPA3,
    PROCESSED_CAPA3,
    TABLES_CAPA3,
    TABLES_CAPA3_CONTROL,
    TABLES_CAPA3_MASTERS,
)


# =========================
# HELPERS
# =========================

def _ensure_dirs() -> None:
    TABLES_CAPA3.mkdir(parents=True, exist_ok=True)
    TABLES_CAPA3_CONTROL.mkdir(parents=True, exist_ok=True)
    TABLES_CAPA3_MASTERS.mkdir(parents=True, exist_ok=True)
    (PROCESSED_CAPA3 / "integrated").mkdir(parents=True, exist_ok=True)
    DB_CAPA3.parent.mkdir(parents=True, exist_ok=True)


# =========================
# 0. VARIABLE SELECTION MATRIX
# =========================

def build_capa3_variable_selection_matrix() -> pd.DataFrame:
    _ensure_dirs()

    records = [
        {
            "dataset": "capa3_master_encuesta",
            "variable": "id_respuesta",
            "descripcion": "Identificador único de respuesta",
            "tipo_variable": "tecnica",
            "rol_analitico": "identificador",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Clave técnica necesaria para trazabilidad y cruces con tablas long",
        },
        {
            "dataset": "capa3_master_encuesta",
            "variable": "timestamp",
            "descripcion": "Marca temporal de respuesta",
            "tipo_variable": "tecnica",
            "rol_analitico": "trazabilidad",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener_solo_trazabilidad",
            "justificacion": "Útil para control de captura, no es variable central del análisis sustantivo",
        },
        {
            "dataset": "capa3_master_encuesta",
            "variable": "grupo_edad",
            "descripcion": "Grupo generacional del encuestado",
            "tipo_variable": "nucleo",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Variable central para el análisis comparativo por generaciones",
        },
        {
            "dataset": "capa3_master_encuesta",
            "variable": "sexo",
            "descripcion": "Sexo del encuestado",
            "tipo_variable": "auxiliar",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Variable de segmentación complementaria para perfilar respuestas",
        },
        {
            "dataset": "capa3_master_encuesta",
            "variable": "rrss_habituales / marcas_ve_frecuencia_rrss / marcas_que_mas_influyen_compra_rrss",
            "descripcion": "Campos multirrespuesta originales",
            "tipo_variable": "tecnica",
            "rol_analitico": "trazabilidad",
            "nivel_nulos": "medio",
            "redundancia": "alta",
            "decision": "mantener_solo_trazabilidad",
            "justificacion": "Se conservan para auditoría, pero el análisis multirrespuesta opera sobre tablas long normalizadas",
        },
        {
            "dataset": "capa3_clustering_ready",
            "variable": "bloque_sociodemografico + bloque_comportamiento + indices_compuestos",
            "descripcion": "Dataset final para segmentación de consumidores",
            "tipo_variable": "mixta",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener",
            "justificacion": "Resume las variables más útiles para clustering y perfilado de segmentos",
        },
        {
            "dataset": "capa3_supervised_ready",
            "variable": "bloque_features + targets_binarios",
            "descripcion": "Dataset final para modelos supervisados",
            "tipo_variable": "mixta",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener_con_control",
            "justificacion": "Se mantiene como base supervisada, excluyendo variables con fuga de información hacia los targets",
        },
        {
            "dataset": "capa3_generacion_summary",
            "variable": "metricas_agregadas_por_generacion",
            "descripcion": "Resumen agregado por grupo de edad",
            "tipo_variable": "contextual",
            "rol_analitico": "contexto",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener_contexto",
            "justificacion": "Útil para EDA y narrativa comparativa entre generaciones",
        },
        {
            "dataset": "encuesta_rrss_long",
            "variable": "id_respuesta / rrss",
            "descripcion": "Tabla long de redes sociales utilizadas",
            "tipo_variable": "nucleo",
            "rol_analitico": "multirrespuesta",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Base normalizada para ranking y análisis de redes sociales",
        },
        {
            "dataset": "encuesta_marcas_vistas_long",
            "variable": "id_respuesta / marca_vista_rrss",
            "descripcion": "Tabla long de marcas vistas en redes",
            "tipo_variable": "nucleo",
            "rol_analitico": "multirrespuesta",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Base normalizada para ranking de exposición a marcas en redes",
        },
        {
            "dataset": "encuesta_marcas_influyen_long",
            "variable": "id_respuesta / marca_influye_compra_rrss",
            "descripcion": "Tabla long de marcas que influyen en compra",
            "tipo_variable": "nucleo",
            "rol_analitico": "multirrespuesta",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Base normalizada para ranking de influencia de marca en compra",
        },
        {
            "dataset": "capa3_sample_structure",
            "variable": "grupo_edad + sexo + n_respuestas",
            "descripcion": "Resumen de la composición muestral por generación y sexo",
            "tipo_variable": "contextual",
            "rol_analitico": "control_muestral",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener_contexto",
            "justificacion": "Permite documentar la estructura muestral y justificar la composición de la encuesta",
        },
        {
            "dataset": "capa3_target_summary",
            "variable": "target + n_validos + pct_clase_1 + pct_clase_0",
            "descripcion": "Resumen del balance de clases de los targets binarios",
            "tipo_variable": "contextual",
            "rol_analitico": "control_modelizacion",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener_contexto",
            "justificacion": "Permite documentar el balance de clases antes de la fase supervisada",
        },
    ]

    df = pd.DataFrame(records)

    output_path = TABLES_CAPA3_CONTROL / "capa3_variable_selection_matrix.csv"
    df.to_csv(output_path, index=False)

    print("Matriz de selección de variables de capa 3 guardada en:")
    print(output_path)
    print("")
    print(df.head(15))

    return df


# =========================
# 0B. DECISIONES FINALES DE VARIABLES
# =========================

def build_capa3_variable_final_decisions() -> pd.DataFrame:
    _ensure_dirs()

    records = [
        {
            "dataset": "capa3_master_encuesta",
            "variable": "id_respuesta",
            "usar_en_modelos": "no",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Identificador técnico para trazabilidad y cruces.",
        },
        {
            "dataset": "capa3_master_encuesta",
            "variable": "timestamp",
            "usar_en_modelos": "no",
            "usar_en_eda": "si",
            "usar_solo_contexto": "si",
            "comentario": "Útil para control de captura; no forma parte del núcleo analítico.",
        },
        {
            "dataset": "capa3_master_encuesta",
            "variable": "grupo_edad",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Segmentación generacional principal.",
        },
        {
            "dataset": "capa3_master_encuesta",
            "variable": "sexo",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Variable auxiliar de perfilado.",
        },
        {
            "dataset": "capa3_clustering_ready",
            "variable": "bloque_clustering",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Tabla final para segmentación de consumidores.",
        },
        {
            "dataset": "capa3_supervised_ready",
            "variable": "bloque_supervisado_sin_fuga",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Tabla final supervisada sin variables derivadas que contengan los targets.",
        },
        {
            "dataset": "capa3_generacion_summary",
            "variable": "metricas_agregadas",
            "usar_en_modelos": "no",
            "usar_en_eda": "si",
            "usar_solo_contexto": "si",
            "comentario": "Resumen interpretativo por generación.",
        },
        {
            "dataset": "encuesta_rrss_long",
            "variable": "rrss",
            "usar_en_modelos": "no",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Base long para análisis multirrespuesta.",
        },
        {
            "dataset": "encuesta_marcas_vistas_long",
            "variable": "marca_vista_rrss",
            "usar_en_modelos": "no",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Base long para exposición a marcas.",
        },
        {
            "dataset": "encuesta_marcas_influyen_long",
            "variable": "marca_influye_compra_rrss",
            "usar_en_modelos": "no",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Base long para influencia de marcas en compra.",
        },
        {
            "dataset": "capa3_sample_structure",
            "variable": "grupo_edad + sexo + n_respuestas",
            "usar_en_modelos": "no",
            "usar_en_eda": "si",
            "usar_solo_contexto": "si",
            "comentario": "Resume la composición muestral de la encuesta y apoya la interpretación."
        },
        {
            "dataset": "capa3_target_summary",
            "variable": "target + n_validos + pct_clase_1 + pct_clase_0",
            "usar_en_modelos": "no",
            "usar_en_eda": "si",
            "usar_solo_contexto": "si",
            "comentario": "Resume el balance de clases de los targets supervisados."
        },
    ]

    df = pd.DataFrame(records)

    output_path = TABLES_CAPA3_CONTROL / "capa3_variable_final_decisions.csv"
    df.to_csv(output_path, index=False)

    print("Decisiones finales de variables guardadas en:")
    print(output_path)
    print("")
    print(df.head(15))

    return df


# =========================
# 0C. PREVIEWS DE MASTERS
# =========================

def build_capa3_master_previews() -> None:
    _ensure_dirs()

    preview_map = {
        "capa3_inventory_preview.csv": PROCESSED_CAPA3 / "integrated" / "capa3_inventory.csv",
        "capa3_master_encuesta_preview.csv": PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv",
        "capa3_clustering_ready_preview.csv": PROCESSED_CAPA3 / "integrated" / "capa3_clustering_ready.csv",
        "capa3_supervised_ready_preview.csv": PROCESSED_CAPA3 / "integrated" / "capa3_supervised_ready.csv",
        "capa3_generacion_summary_preview.csv": PROCESSED_CAPA3 / "integrated" / "capa3_generacion_summary.csv",
        "capa3_sample_structure_preview.csv": PROCESSED_CAPA3 / "integrated" / "capa3_sample_structure.csv",
        "capa3_target_summary_preview.csv": PROCESSED_CAPA3 / "integrated" / "capa3_target_summary.csv",
        "capa3_dataset_quality_summary_preview.csv": TABLES_CAPA3_CONTROL / "capa3_dataset_quality_summary.csv",
    }

    for output_name, input_path in preview_map.items():
        df = pd.read_csv(input_path)
        df.head(50).to_csv(TABLES_CAPA3_MASTERS / output_name, index=False)

    print("Previews de masters guardados en:")
    print(TABLES_CAPA3_MASTERS)


# =========================
# 1. INVENTARIO
# =========================

def build_capa3_inventory() -> pd.DataFrame:
    _ensure_dirs()

    inventory = pd.DataFrame(
        [
            {
                "dataset": "processed/capa3/survey/encuesta_fastfashion_clean.csv",
                "fuente": "Encuesta propia",
                "frecuencia": "corte transversal",
                "granularidad": "individuo",
                "rol_capa3": "master individual de respuestas",
            },
            {
                "dataset": "processed/capa3/survey/encuesta_rrss_long.csv",
                "fuente": "Encuesta propia",
                "frecuencia": "corte transversal",
                "granularidad": "individuo-rrss",
                "rol_capa3": "multirrespuesta de redes sociales",
            },
            {
                "dataset": "processed/capa3/survey/encuesta_marcas_vistas_long.csv",
                "fuente": "Encuesta propia",
                "frecuencia": "corte transversal",
                "granularidad": "individuo-marca",
                "rol_capa3": "multirrespuesta de marcas vistas en rrss",
            },
            {
                "dataset": "processed/capa3/survey/encuesta_marcas_influyen_long.csv",
                "fuente": "Encuesta propia",
                "frecuencia": "corte transversal",
                "granularidad": "individuo-marca",
                "rol_capa3": "multirrespuesta de marcas que influyen en compra",
            },
        ]
    )

    output_path = PROCESSED_CAPA3 / "integrated" / "capa3_inventory.csv"
    inventory.to_csv(output_path, index=False)

    print("Inventario capa 3 guardado en:")
    print(output_path)
    print("")
    print(inventory)

    return inventory


# =========================
# 2. MASTER ENCUESTA
# =========================

def build_capa3_master_encuesta() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA3 / "survey" / "encuesta_fastfashion_clean.csv"
    df = pd.read_csv(input_path)

    output_path = PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv"
    df.to_csv(output_path, index=False)

    master_table_path = TABLES_CAPA3_MASTERS / "capa3_master_encuesta_preview.csv"
    df.head(50).to_csv(master_table_path, index=False)

    print("Master encuesta guardado en:")
    print(output_path)
    print("")
    print(df.head())

    return df


# =========================
# 3. DATASET CLUSTERING READY
# =========================

def build_capa3_clustering_ready() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv"
    df = pd.read_csv(input_path)

    clustering_cols = [
        "id_respuesta",
        "grupo_edad",
        "sexo",
        "freq_compra_anual",
        "canal_compra_moda",
        "gasto_mensual_moda",
        "tiempo_rrss_dia",
        "freq_contenido_moda_rrss",
        "sigue_influencers_moda",
        "indice_influencia_rrss",
        "indice_impulso_tendencia",
        "indice_confianza_influencers",
        "indice_escepticismo_influencers",
        "indice_difusion_fastfashion",
        "indice_postcompra",
        "indice_riesgo_arrepentimiento",
        "compra_ult_6m_por_rrss_bin",
    ]

    clustering_df = df[clustering_cols].copy()

    output_path = PROCESSED_CAPA3 / "integrated" / "capa3_clustering_ready.csv"
    clustering_df.to_csv(output_path, index=False)

    print("Dataset clustering ready guardado en:")
    print(output_path)
    print("")
    print(clustering_df.head())

    return clustering_df


# =========================
# 4. DATASET SUPERVISADO READY
# =========================

def build_capa3_supervised_ready() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv"
    df = pd.read_csv(input_path)

    supervised_cols = [
        "id_respuesta",
        "grupo_edad",
        "sexo",
        "freq_compra_anual",
        "canal_compra_moda",
        "gasto_mensual_moda",
        "tiempo_rrss_dia",
        "freq_contenido_moda_rrss",
        "sigue_influencers_moda",
        "compra_ult_6m_por_rrss_bin",
        "indice_influencia_rrss",
        "indice_impulso_tendencia",
        "indice_confianza_influencers",
        "indice_escepticismo_influencers",
        "indice_difusion_fastfashion",
        "indice_riesgo_arrepentimiento",
        "target_recomendaria_bin",
        "target_seguira_comprando_bin",
    ]

    supervised_df = df[supervised_cols].copy()

    output_path = PROCESSED_CAPA3 / "integrated" / "capa3_supervised_ready.csv"
    supervised_df.to_csv(output_path, index=False)

    print("Dataset supervised ready guardado en:")
    print(output_path)
    print("")
    print(supervised_df.head())

    return supervised_df


# =========================
# 5. RESUMEN POR GENERACIÓN Y TARGET
# =========================

def build_capa3_generacion_summary() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv"
    df = pd.read_csv(input_path)

    summary = (
        df.groupby("grupo_edad", dropna=False)
        .agg(
            n_respuestas=("id_respuesta", "count"),
            influencia_rrss_media=("indice_influencia_rrss", "mean"),
            impulso_tendencia_medio=("indice_impulso_tendencia", "mean"),
            confianza_influencers_media=("indice_confianza_influencers", "mean"),
            postcompra_media=("indice_postcompra", "mean"),
            arrepentimiento_medio=("indice_riesgo_arrepentimiento", "mean"),
            pct_compra_ult_6m_rrss=("compra_ult_6m_por_rrss_bin", "mean"),
        )
        .reset_index()
    )

    summary["pct_compra_ult_6m_rrss"] = (summary["pct_compra_ult_6m_rrss"] * 100).round(2)

    output_path = PROCESSED_CAPA3 / "integrated" / "capa3_generacion_summary.csv"
    summary.to_csv(output_path, index=False)

    print("Resumen por generación guardado en:")
    print(output_path)
    print("")
    print(summary)

    return summary


def build_capa3_sample_structure() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv"
    df = pd.read_csv(input_path)

    summary = (
        df.groupby(["grupo_edad", "sexo"], dropna=False)
        .size()
        .reset_index(name="n_respuestas")
        .sort_values(["grupo_edad", "sexo"])
    )

    output_path = PROCESSED_CAPA3 / "integrated" / "capa3_sample_structure.csv"
    summary.to_csv(output_path, index=False)

    print("Estructura muestral guardada en:")
    print(output_path)
    print("")
    print(summary)

    return summary


def build_capa3_target_summary() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv"
    df = pd.read_csv(input_path)

    target_cols = [
        "compra_ult_6m_por_rrss_bin",
        "target_recomendaria_bin",
        "target_seguira_comprando_bin",
    ]

    records = []
    for col in target_cols:
        valid = df[col].dropna()
        records.append(
            {
                "target": col,
                "n_validos": int(valid.shape[0]),
                "pct_clase_1": round(float((valid == 1).mean() * 100), 2) if not valid.empty else None,
                "pct_clase_0": round(float((valid == 0).mean() * 100), 2) if not valid.empty else None,
            }
        )

    out = pd.DataFrame(records)
    output_path = PROCESSED_CAPA3 / "integrated" / "capa3_target_summary.csv"
    out.to_csv(output_path, index=False)

    print("Resumen de targets guardado en:")
    print(output_path)
    print("")
    print(out)

    return out

# =========================
# 6. SQLITE
# =========================

def build_capa3_sqlite() -> None:
    _ensure_dirs()

    tables = {
        "encuesta_fastfashion_clean": PROCESSED_CAPA3 / "survey" / "encuesta_fastfashion_clean.csv",
        "encuesta_rrss_long": PROCESSED_CAPA3 / "survey" / "encuesta_rrss_long.csv",
        "encuesta_marcas_vistas_long": PROCESSED_CAPA3 / "survey" / "encuesta_marcas_vistas_long.csv",
        "encuesta_marcas_influyen_long": PROCESSED_CAPA3 / "survey" / "encuesta_marcas_influyen_long.csv",
        "capa3_inventory": PROCESSED_CAPA3 / "integrated" / "capa3_inventory.csv",
        "capa3_master_encuesta": PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv",
        "capa3_clustering_ready": PROCESSED_CAPA3 / "integrated" / "capa3_clustering_ready.csv",
        "capa3_supervised_ready": PROCESSED_CAPA3 / "integrated" / "capa3_supervised_ready.csv",
        "capa3_generacion_summary": PROCESSED_CAPA3 / "integrated" / "capa3_generacion_summary.csv",
        "capa3_sample_structure": PROCESSED_CAPA3 / "integrated" / "capa3_sample_structure.csv",
        "capa3_target_summary": PROCESSED_CAPA3 / "integrated" / "capa3_target_summary.csv",
        "capa3_variable_selection_matrix": TABLES_CAPA3_CONTROL / "capa3_variable_selection_matrix.csv",
        "capa3_variable_final_decisions": TABLES_CAPA3_CONTROL / "capa3_variable_final_decisions.csv",
        "capa3_dataset_quality_summary": TABLES_CAPA3_CONTROL / "capa3_dataset_quality_summary.csv",
        "capa3_cronbach_alpha": TABLES_CAPA3_CONTROL / "capa3_cronbach_alpha.csv",
        "capa3_justificacion_tipo_analisis": TABLES_CAPA3_CONTROL / "capa3_justificacion_tipo_analisis.csv",
    }

    conn = sqlite3.connect(DB_CAPA3)

    for table_name, path in tables.items():
        df = pd.read_csv(path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Tabla cargada: {table_name}")

    conn.close()

    print("")
    print("Base SQLite capa 3 creada en:")
    print(DB_CAPA3)

# =========================
# 7. DATASET QUALITY SUMMARY
# =========================

def build_capa3_dataset_quality_summary() -> pd.DataFrame:
    _ensure_dirs()

    dataset_map = {
        "capa3_master_encuesta": PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv",
        "capa3_clustering_ready": PROCESSED_CAPA3 / "integrated" / "capa3_clustering_ready.csv",
        "capa3_supervised_ready": PROCESSED_CAPA3 / "integrated" / "capa3_supervised_ready.csv",
    }

    records = []

    for dataset_name, path in dataset_map.items():
        df = pd.read_csv(path)

        n_rows = df.shape[0]
        n_cols = df.shape[1]
        complete_cases = int(df.dropna().shape[0])
        pct_complete_cases = round((complete_cases / n_rows) * 100, 2) if n_rows > 0 else 0.0
        total_nulls = int(df.isna().sum().sum())
        avg_nulls_per_row = round(float(df.isna().sum(axis=1).mean()), 4) if n_rows > 0 else 0.0

        records.append(
            {
                "dataset": dataset_name,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "complete_cases": int(df.dropna().shape[0]),
                "pct_complete_cases": round(df.dropna().shape[0] / df.shape[0] * 100, 2),
                "total_nulls": int(df.isna().sum().sum()),
                "avg_nulls_per_row": round(float(df.isna().sum(axis=1).mean()), 4),
            }
        )

    quality_df = pd.DataFrame(records)

    output_path = TABLES_CAPA3_CONTROL / "capa3_dataset_quality_summary.csv"
    quality_df.to_csv(output_path, index=False)

    print("Resumen de calidad de datasets guardado en:")
    print(output_path)
    print("")
    print(quality_df)

    return quality_df


# =========================
# CRONBACH
# =========================

def build_capa3_cronbach_alpha() -> pd.DataFrame:
    """
    Calcula el alpha de Cronbach para cada índice compuesto de la encuesta.

    El alpha de Cronbach mide la consistencia interna de un conjunto de ítems
    que se supone miden el mismo constructo. Formula:
        α = (k / (k-1)) * (1 - Σvar_items / var_total)
    donde k = número de ítems, var_items = varianza de cada ítem,
    var_total = varianza de la suma de los ítems.

    Interpretación:
        α < 0.6  → inaceptable (cuestionar la agrupación)
        0.6–0.7  → aceptable en ciencias sociales exploratorias
        0.7–0.8  → bueno
        0.8–0.9  → muy bueno
        > 0.9    → excelente (posible redundancia entre ítems)

    Referencia: Cronbach, L. J. (1951). Coefficient alpha and the internal
    structure of tests. Psychometrika, 16(3), 297-334.

    NOTA: Para índices de 1 ítem (indice_escepticismo_influencers) el alpha
    no es aplicable — se documenta como N/A con nota metodológica.
    Para índices de 2 ítems el alpha puede ser bajo pero es aceptable si la
    correlación entre ítems es alta (r > 0.4).
    """
    _ensure_dirs()

    input_path = PROCESSED_CAPA3 / "survey" / "encuesta_fastfashion_clean.csv"
    df = pd.read_csv(input_path)

    INDICES_ITEMS = {
        "indice_influencia_rrss": [
            "rs_influyen_compra", "ha_comprado_por_ver_en_rrss",
            "descubre_marcas_por_rrss", "contenido_moda_aumenta_ganas",
            "rrss_mas_que_publicidad",
        ],
        "indice_impulso_tendencia": [
            "compra_por_repeticion_rrss", "atraen_prendas_de_moda",
            "compra_impulso", "influye_prueba_social", "interes_productos_virales",
        ],
        "indice_confianza_influencers": [
            "confia_influencers", "autenticidad_influencer_compra",
        ],
        "indice_escepticismo_influencers": [
            "influencers_interes_comercial",
        ],
        "indice_difusion_fastfashion": [
            "rrss_aceleran_tendencias", "rrss_favorecen_consumo_rapido",
        ],
        "indice_postcompra": [
            "satisfaccion_compra_rrss", "seguira_comprando_influido_rrss",
            "recomendaria_productos_descubiertos_rrss",
        ],
        "indice_riesgo_arrepentimiento": [
            "ha_comprado_y_apenas_usado", "arrepentimiento_compra_rrss",
        ],
    }

    def cronbach_alpha(data: pd.DataFrame) -> float:
        """Calcula alpha de Cronbach para un DataFrame de ítems (filas=sujetos, cols=ítems)."""
        data_clean = data.dropna()
        if len(data_clean) < 2 or data_clean.shape[1] < 2:
            return float("nan")
        k = data_clean.shape[1]
        var_items = data_clean.var(axis=0, ddof=1).sum()
        var_total = data_clean.sum(axis=1).var(ddof=1)
        if var_total == 0:
            return float("nan")
        return round((k / (k - 1)) * (1 - var_items / var_total), 4)

    def interpretar_alpha(alpha: float, k: int) -> str:
        if k == 1:
            return "N/A — indice de 1 item, alpha no aplicable"
        if pd.isna(alpha):
            return "N/A — datos insuficientes"
        if alpha >= 0.9:
            return "excelente (posible redundancia entre items)"
        if alpha >= 0.8:
            return "muy_bueno"
        if alpha >= 0.7:
            return "bueno"
        if alpha >= 0.6:
            return "aceptable"
        return "inaceptable — revisar agrupacion de items"

    records = []
    for indice_name, items in INDICES_ITEMS.items():
        items_presentes = [c for c in items if c in df.columns]
        k = len(items_presentes)

        if k == 1:
            alpha = float("nan")
            # Para 1 item: calcular estadísticos descriptivos del item
            s = pd.to_numeric(df[items_presentes[0]], errors="coerce").dropna()
            correlacion_media_items = float("nan")
        else:
            data = df[items_presentes].apply(pd.to_numeric, errors="coerce")
            alpha = cronbach_alpha(data)
            # Correlación media inter-items (complementa el alpha para n=2)
            corr_matrix = data.corr()
            mask = ~np.eye(k, dtype=bool)
            correlacion_media_items = (
                round(float(corr_matrix.where(mask).stack().mean()), 4)
                if k > 1 else np.nan
            ) if k > 1 else float("nan")

        records.append({
            "indice": indice_name,
            "n_items": k,
            "items": ", ".join(items_presentes),
            "n_respondentes": int(df[items_presentes].dropna().shape[0]) if k > 0 else 0,
            "alpha_cronbach": alpha if not pd.isna(alpha) else None,
            "correlacion_media_items": correlacion_media_items if not pd.isna(correlacion_media_items) else None,
            "interpretacion": interpretar_alpha(alpha, k),
            "decision": (
                "validar_agrupacion" if not pd.isna(alpha) and alpha < 0.6
                else "agrupacion_aceptada"
            ),
        })

    alpha_df = pd.DataFrame(records)
    out_path = TABLES_CAPA3_CONTROL / "capa3_cronbach_alpha.csv"
    alpha_df.to_csv(out_path, index=False)

    print(f"  Alpha de Cronbach calculado para {len(alpha_df)} índices → {out_path.name}")
    print()
    for _, row in alpha_df.iterrows():
        alpha_str = f"α={row['alpha_cronbach']:.3f}" if row["alpha_cronbach"] is not None else "N/A"
        print(f"    {row['indice']:45s} {alpha_str:12s} → {row['interpretacion']}")

    return alpha_df


def build_capa3_justificacion_analitica() -> pd.DataFrame:
    """
    Documenta la justificación del tipo de análisis descriptivo aplicado
    a la encuesta de fast fashion.

    La rúbrica exige "justificación del tipo de análisis descriptivo":
    esta función responde a ese requisito para los datos de encuesta Likert.
    """
    _ensure_dirs()

    records = [
        {
            "dataset": "capa3_master_encuesta",
            "tipo_dato": "encuesta_likert_transversal",
            "n_respondentes": "~200",
            "tipo_analisis_elegido": "analisis_descriptivo_transversal_con_indices_compuestos",
            "justificacion": (
                "Los datos son de corte transversal (una sola recogida) con variables "
                "de actitud y comportamiento en escala Likert 1-5. El analisis aplica: "
                "(1) estadisticos descriptivos por variable (media, mediana, desv. tipica, "
                "percentiles); (2) analisis de distribucion de respuestas para detectar "
                "sesgos (efecto techo/suelo en items Likert); (3) analisis de indices "
                "compuestos con validacion de consistencia interna (alpha de Cronbach); "
                "(4) analisis segmentado por generacion (Millennials, Gen-Z, etc.) para "
                "detectar diferencias de actitud entre cohortes; "
                "(5) tablas de contingencia para preguntas categoricas."
            ),
            "alternativa_descartada": (
                "Series temporales: no aplicable, datos de corte transversal. "
                "Analisis factorial confirmatorio (CFA): requiere n >> 200 y software "
                "especializado; el alpha de Cronbach es suficiente para validacion basica. "
                "Tests de hipotesis parametricos (t-test, ANOVA): requieren normalidad "
                "— los datos Likert 1-5 son ordinales, se usan descriptivos y pruebas "
                "no parametricas si procede."
            ),
        },
        {
            "dataset": "capa3_clustering_ready",
            "tipo_dato": "datos_numericos_normalizados_para_clustering",
            "n_respondentes": "~200",
            "tipo_analisis_elegido": "analisis_descriptivo_pre_clustering",
            "justificacion": (
                "Dataset con indices compuestos normalizados (min-max) listo para "
                "algoritmos de clustering no supervisado. El EDA descriptivo verifica: "
                "ausencia de nulos, distribucion de variables, outliers IQR. "
                "Los boxplots por generacion anticipan si existen grupos naturales "
                "en el espacio de caracteristicas de los indices."
            ),
            "alternativa_descartada": "Datos raw Likert: requieren normalizacion antes de clustering por distancias.",
        },
        {
            "dataset": "capa3_supervised_ready",
            "tipo_dato": "datos_para_clasificacion_supervisada",
            "n_respondentes": "~200",
            "tipo_analisis_elegido": "analisis_balance_clases_y_distribucion_features",
            "justificacion": (
                "Dataset con targets binarios (target_recomendaria_bin, "
                "target_seguira_comprando_bin) para modelos supervisados. "
                "El EDA verifica el balance de clases (% clase 1 vs 0), "
                "la distribucion de features por clase para anticipar separabilidad, "
                "y la correlacion entre features para detectar multicolinealidad."
            ),
            "alternativa_descartada": "Regresion ordinal sobre escala 1-5: viable pero fuera del alcance del TFG.",
        },
        {
            "dataset": "tablas_long_multirrespuesta",
            "tipo_dato": "datos_categoricos_multirrespuesta",
            "n_respondentes": "~200",
            "tipo_analisis_elegido": "analisis_frecuencias_ranking",
            "justificacion": (
                "Las preguntas de seleccion multiple (RRSS habituales, marcas) "
                "se expanden a formato long para analisis de frecuencias. "
                "Se calcula: ranking de RRSS por frecuencia de uso, "
                "ranking de marcas por visibilidad e influencia en compra, "
                "y segmentacion del ranking por generacion."
            ),
            "alternativa_descartada": "Variables dummy por opcion: genera alta dimensionalidad con n=200.",
        },
    ]

    df = pd.DataFrame(records)
    out_path = TABLES_CAPA3_CONTROL / "capa3_justificacion_tipo_analisis.csv"
    df.to_csv(out_path, index=False)
    print(f"  Justificación tipo análisis capa3: {len(df)} datasets → {out_path.name}")
    return df


# =========================
# RUN ALL
# =========================

def run_all_builds() -> None:
    build_capa3_inventory()
    build_capa3_cronbach_alpha()
    build_capa3_justificacion_analitica()
    build_capa3_master_encuesta()
    build_capa3_clustering_ready()
    build_capa3_supervised_ready()
    build_capa3_generacion_summary()
    build_capa3_sample_structure()
    build_capa3_target_summary()
    build_capa3_variable_selection_matrix()
    build_capa3_variable_final_decisions()
    build_capa3_master_previews()
    build_capa3_dataset_quality_summary()
    build_capa3_sqlite()
    print("Todos los builds de capa 3 completados.")


if __name__ == "__main__":
    run_all_builds()