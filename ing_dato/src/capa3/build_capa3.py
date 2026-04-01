import sqlite3
import pandas as pd

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
# RUN ALL
# =========================

def run_all_builds() -> None:
    build_capa3_inventory()
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