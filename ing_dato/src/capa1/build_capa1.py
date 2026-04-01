import sqlite3

import pandas as pd

from src.common.config import (
    DB_CAPA1,
    PROCESSED_CAPA1,
    TABLES_CAPA1,
    TABLES_CAPA1_CONTROL,
    TABLES_CAPA1_MASTERS,
)


# =========================
# HELPERS
# =========================

def _ensure_dirs() -> None:
    (PROCESSED_CAPA1 / "comercio_electronico").mkdir(parents=True, exist_ok=True)
    (PROCESSED_CAPA1 / "integrated").mkdir(parents=True, exist_ok=True)

    TABLES_CAPA1.mkdir(parents=True, exist_ok=True)
    TABLES_CAPA1_CONTROL.mkdir(parents=True, exist_ok=True)
    TABLES_CAPA1_MASTERS.mkdir(parents=True, exist_ok=True)

    DB_CAPA1.parent.mkdir(parents=True, exist_ok=True)


# =========================
# 0. VARIABLE SELECTION MATRIX
# =========================

def build_variable_selection_matrix_capa1() -> pd.DataFrame:
    _ensure_dirs()

    records = [
        # =========================
        # contexto_digitalizacion_clean
        # =========================
        {
            "dataset": "contexto_digitalizacion_clean",
            "variable": "anio",
            "descripcion": "Año de observación",
            "tipo_variable": "tecnica",
            "rol_analitico": "join_temporal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Variable temporal principal para agregaciones y cruces con otras fuentes anuales",
        },
        {
            "dataset": "contexto_digitalizacion_clean",
            "variable": "pct_usuarios_rrss",
            "descripcion": "Porcentaje de usuarios de redes sociales en España",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Indicador clave del contexto de digitalización social",
        },
        {
            "dataset": "contexto_digitalizacion_clean",
            "variable": "pct_personas_compra_online",
            "descripcion": "Porcentaje de personas que realizan compras online",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Indicador clave del contexto de digitalización del consumo",
        },

        # =========================
        # contexto_digitalizacion_extended / documentado
        # =========================
        {
            "dataset": "contexto_digitalizacion_extended",
            "variable": "pct_personas_compra_ropa_online",
            "descripcion": "Porcentaje de personas que compran ropa online",
            "tipo_variable": "auxiliar",
            "rol_analitico": "contexto",
            "nivel_nulos": "alto",
            "redundancia": "baja",
            "decision": "mantener_solo_trazabilidad",
            "justificacion": "Variable relevante temáticamente, pero con cobertura insuficiente para el master analítico principal",
        },
        {
            "dataset": "contexto_digitalizacion_documentado",
            "variable": "comentarios_hitos",
            "descripcion": "Comentarios contextuales sobre hitos del periodo",
            "tipo_variable": "contextual",
            "rol_analitico": "trazabilidad",
            "nivel_nulos": "medio",
            "redundancia": "baja",
            "decision": "mantener_solo_trazabilidad",
            "justificacion": "Útil para documentación y narrativa, no para análisis cuantitativo",
        },
        {
            "dataset": "contexto_digitalizacion_documentado",
            "variable": "fuente_usuarios_redes / fuente_compra_online / fuente_compra_ropa_online",
            "descripcion": "Fuentes originales de cada variable",
            "tipo_variable": "tecnica",
            "rol_analitico": "trazabilidad",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener_solo_trazabilidad",
            "justificacion": "Necesarias para auditoría metodológica, no para el master analítico",
        },

        # =========================
        # comercio_electronico_core_std
        # =========================
        {
            "dataset": "comercio_electronico_core_std",
            "variable": "anio",
            "descripcion": "Año de observación",
            "tipo_variable": "tecnica",
            "rol_analitico": "join_temporal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Necesaria para series anuales y cruces con el resto de fuentes",
        },
        {
            "dataset": "comercio_electronico_core_std",
            "variable": "indicador",
            "descripcion": "Etiqueta original del indicador",
            "tipo_variable": "tecnica",
            "rol_analitico": "trazabilidad",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener_solo_trazabilidad",
            "justificacion": "Útil para conservar la referencia original del INE, pero el análisis opera sobre el indicador estandarizado",
        },
        {
            "dataset": "comercio_electronico_core_std",
            "variable": "indicador_std",
            "descripcion": "Indicador estandarizado común entre años",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Permite comparar entre años indicadores equivalentes con distinta codificación original",
        },
        {
            "dataset": "comercio_electronico_core_std",
            "variable": "tamano_empresa",
            "descripcion": "Tamaño de empresa",
            "tipo_variable": "auxiliar",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Permite análisis por tamaño empresarial y selección de total para masters integrados",
        },
        {
            "dataset": "comercio_electronico_core_std",
            "variable": "valor",
            "descripcion": "Valor numérico del indicador",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Variable cuantitativa principal del bloque de ecommerce empresarial",
        },

        # =========================
        # capa1_master_anual_analysis
        # =========================
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "anio",
            "descripcion": "Año de observación",
            "tipo_variable": "tecnica",
            "rol_analitico": "join_temporal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Variable temporal principal del master anual",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_usuarios_rrss",
            "descripcion": "Uso de redes sociales",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Indicador esencial del contexto digital",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_personas_compra_online",
            "descripcion": "Compra online de personas",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Indicador esencial del consumo digital",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_facturacion_empresas_online",
            "descripcion": "Facturación empresarial procedente de ventas online",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Aporta la perspectiva empresarial del canal online",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_empresas_venden_ecommerce",
            "descripcion": "Porcentaje de empresas que venden por ecommerce",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Mide adopción empresarial del canal ecommerce",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_empresas_venden_web_apps",
            "descripcion": "Porcentaje de empresas que venden por web o apps",
            "tipo_variable": "auxiliar",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener",
            "justificacion": "Refina la medición de adopción del ecommerce por canal concreto",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_ventas_ecommerce_sobre_total",
            "descripcion": "Peso del ecommerce sobre ventas totales",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Mide la importancia relativa del canal online en el total de ventas",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_ventas_ecommerce_sobre_total_empresas_que_venden",
            "descripcion": "Peso del ecommerce sobre ventas de empresas que ya venden online",
            "tipo_variable": "auxiliar",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener",
            "justificacion": "Aporta una lectura más fina del peso del canal dentro del subconjunto de empresas activas en ecommerce",
        },

        # =========================
        # capa1_master_mensual_analysis
        # =========================
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "fecha",
            "descripcion": "Mes de observación",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Variable temporal principal del master mensual",
        },
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "anio",
            "descripcion": "Año derivado",
            "tipo_variable": "tecnica",
            "rol_analitico": "join_temporal",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener",
            "justificacion": "Facilita agregaciones anuales y lectura resumida",
        },
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "mes",
            "descripcion": "Mes numérico derivado",
            "tipo_variable": "tecnica",
            "rol_analitico": "join_temporal",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener",
            "justificacion": "Facilita análisis estacional y heatmaps",
        },
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "indice_retail_moda",
            "descripcion": "Índice mensual del retail de moda",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Serie principal para estudiar la evolución del sector moda",
        },
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "indice_retail_total",
            "descripcion": "Índice mensual del retail total",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Serie de referencia para comparar el desempeño de la moda frente al retail agregado",
        },
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "ratio_moda_vs_total",
            "descripcion": "Ratio entre retail de moda y retail total",
            "tipo_variable": "auxiliar",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Variable derivada con alto valor interpretativo para comparar rendimiento relativo",
        },
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "dif_moda_vs_total",
            "descripcion": "Diferencia entre índice de moda e índice retail total",
            "tipo_variable": "auxiliar",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Complementa el ratio con una lectura absoluta de la brecha entre ambas series",
        },
    ]

    df = pd.DataFrame(records)

    output_path = TABLES_CAPA1_CONTROL / "capa1_variable_selection_matrix.csv"
    df.to_csv(output_path, index=False)

    print("Matriz de selección de variables de capa 1 guardada en:")
    print(output_path)
    print("")
    print(df.head(15))

    return df


# =========================
# 1. BUILD COMERCIO CORE
# =========================

def build_comercio_core() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA1 / "comercio_electronico" / "comercio_electronico_clean.csv"
    df = pd.read_csv(input_path)

    patterns = [
        r"% de empresas que han realizado ventas por comercio electrónico",
        r"% ventas mediante comercio electrónico sobre el total de ventas$",
        r"% ventas mediante comercio electrónico sobre el total de ventas de las empresas que venden por comercio electrónico",
        r"% de empresas que han realizado ventas mediante páginas web o apps",
    ]

    mask = df["indicador"].str.contains("|".join(patterns), case=False, na=False, regex=True)
    core = df[mask].copy().reset_index(drop=True)

    output_path = PROCESSED_CAPA1 / "comercio_electronico" / "comercio_electronico_core.csv"
    core.to_csv(output_path, index=False)

    print("Comercio core guardado en:")
    print(output_path)
    print("")
    print("Dimensiones core:", core.shape)
    print("")
    print("Indicadores incluidos:")
    print(core["indicador"].dropna().unique().tolist())

    return core


# =========================
# 2. STANDARDIZE COMERCIO CORE
# =========================

def standardize_comercio_core() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA1 / "comercio_electronico" / "comercio_electronico_core.csv"
    df = pd.read_csv(input_path)

    def map_indicator(text: str) -> str | None:
        if pd.isna(text):
            return None

        t = str(text).lower()

        if "empresas que han realizado ventas por comercio electrónico" in t:
            return "pct_empresas_venden_ecommerce"

        if "ventas mediante comercio electrónico sobre el total de ventas de las empresas que venden por comercio electrónico" in t:
            return "pct_ventas_ecommerce_sobre_total_empresas_que_venden"

        if "ventas mediante comercio electrónico sobre el total de ventas" in t:
            return "pct_ventas_ecommerce_sobre_total"

        if "empresas que han realizado ventas mediante páginas web o apps" in t:
            return "pct_empresas_venden_web_apps"

        return None

    df["indicador_std"] = df["indicador"].apply(map_indicator)
    df = df[df["indicador_std"].notna()].copy().reset_index(drop=True)

    output_path = PROCESSED_CAPA1 / "comercio_electronico" / "comercio_electronico_core_std.csv"
    df.to_csv(output_path, index=False)

    print("Comercio core estandarizado guardado en:")
    print(output_path)
    print("")
    print("Dimensiones:", df.shape)
    print("")
    print("Indicadores estándar detectados:")
    print(df["indicador_std"].value_counts())

    return df


# =========================
# 3. BUILD INVENTORY
# =========================

def build_capa1_inventory() -> pd.DataFrame:
    _ensure_dirs()

    records = [
        {
            "dataset": "processed/capa1/contexto_digitalizacion/contexto_digitalizacion_clean.csv",
            "fuente": "DataReportal + Eurostat",
            "frecuencia": "anual",
            "periodo_inicio": 2020,
            "periodo_fin": 2025,
            "granularidad": "España",
            "rol_capa1": "contexto digital",
        },
        {
            "dataset": "processed/capa1/eurostat/eurostat_moda_mensual_clean.csv",
            "fuente": "Eurostat",
            "frecuencia": "mensual",
            "periodo_inicio": "2010-01",
            "periodo_fin": "2023-12",
            "granularidad": "España",
            "rol_capa1": "contexto sectorial moda",
        },
        {
            "dataset": "processed/capa1/eurostat/eurostat_retail_total_mensual_clean.csv",
            "fuente": "Eurostat",
            "frecuencia": "mensual",
            "periodo_inicio": "2010-01",
            "periodo_fin": "2025-09",
            "granularidad": "España",
            "rol_capa1": "contexto retail general",
        },
        {
            "dataset": "processed/capa1/eurostat/eurostat_online_empresas_clean.csv",
            "fuente": "Eurostat",
            "frecuencia": "anual",
            "periodo_inicio": 2015,
            "periodo_fin": 2023,
            "granularidad": "España",
            "rol_capa1": "adopcion online empresarial",
        },
        {
            "dataset": "processed/capa1/comercio_electronico/comercio_electronico_core_std.csv",
            "fuente": "INE / Encuesta TIC y Comercio Electrónico",
            "frecuencia": "anual",
            "periodo_inicio": 2015,
            "periodo_fin": 2023,
            "granularidad": "España por tamaño de empresa",
            "rol_capa1": "contexto ecommerce empresarial",
        },
    ]

    inventory = pd.DataFrame(records)

    output_path = PROCESSED_CAPA1 / "integrated" / "capa1_inventory.csv"
    inventory.to_csv(output_path, index=False)

    print("Inventario guardado en:")
    print(output_path)
    print("")
    print(inventory)

    return inventory


# =========================
# 4. BUILD MASTER ANUAL
# =========================

def build_capa1_master_anual() -> tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()

    contexto_path = PROCESSED_CAPA1 / "contexto_digitalizacion" / "contexto_digitalizacion_clean.csv"
    online_empresas_path = PROCESSED_CAPA1 / "eurostat" / "eurostat_online_empresas_clean.csv"
    comercio_path = PROCESSED_CAPA1 / "comercio_electronico" / "comercio_electronico_core_std.csv"

    contexto = pd.read_csv(contexto_path)
    online_empresas = pd.read_csv(online_empresas_path)
    comercio = pd.read_csv(comercio_path)

    comercio_total = comercio[comercio["tamano_empresa"] == "total"].copy()

    comercio_pivot = comercio_total.pivot_table(
        index="anio",
        columns="indicador_std",
        values="valor",
        aggfunc="first",
    ).reset_index()

    master = contexto.merge(
        online_empresas[["anio", "valor_pct"]],
        on="anio",
        how="left",
    )

    master = master.rename(columns={"valor_pct": "pct_facturacion_empresas_online"})
    master = master.merge(comercio_pivot, on="anio", how="left")

    output_full = PROCESSED_CAPA1 / "integrated" / "capa1_master_anual_full.csv"
    master.to_csv(output_full, index=False)

    master_analysis = master[master["anio"].between(2020, 2023)].copy().reset_index(drop=True)
    output_analysis = PROCESSED_CAPA1 / "integrated" / "capa1_master_anual_analysis.csv"
    master_analysis.to_csv(output_analysis, index=False)

    print("Master anual full guardado en:")
    print(output_full)
    print("")
    print("Master anual analysis guardado en:")
    print(output_analysis)
    print("")
    print(master_analysis)
    print("")
    print("Columnas:")
    print(master_analysis.columns.tolist())

    return master, master_analysis


# =========================
# 5. BUILD MASTER MENSUAL
# =========================

def build_capa1_master_mensual() -> tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()

    moda_path = PROCESSED_CAPA1 / "eurostat" / "eurostat_moda_mensual_clean.csv"
    retail_path = PROCESSED_CAPA1 / "eurostat" / "eurostat_retail_total_mensual_clean.csv"

    moda = pd.read_csv(moda_path)
    retail = pd.read_csv(retail_path)

    moda["fecha"] = pd.to_datetime(moda["fecha"])
    retail["fecha"] = pd.to_datetime(retail["fecha"])

    moda = moda.rename(columns={"valor_indice": "indice_retail_moda"})
    retail = retail.rename(columns={"valor_indice": "indice_retail_total"})

    moda = moda[["fecha", "indice_retail_moda"]].copy()
    retail = retail[["fecha", "indice_retail_total"]].copy()

    master_mensual = moda.merge(retail, on="fecha", how="inner")

    master_mensual["anio"] = master_mensual["fecha"].dt.year
    master_mensual["mes"] = master_mensual["fecha"].dt.month
    master_mensual["ratio_moda_vs_total"] = (
        master_mensual["indice_retail_moda"] / master_mensual["indice_retail_total"]
    )
    master_mensual["dif_moda_vs_total"] = (
        master_mensual["indice_retail_moda"] - master_mensual["indice_retail_total"]
    )

    master_mensual = master_mensual.sort_values("fecha").reset_index(drop=True)

    output_full = PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_full.csv"
    master_mensual.to_csv(output_full, index=False)

    master_analysis = master_mensual[
        master_mensual["anio"].between(2015, 2023)
    ].copy().reset_index(drop=True)

    output_analysis = PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_analysis.csv"
    master_analysis.to_csv(output_analysis, index=False)

    print("Master mensual full guardado en:")
    print(output_full)
    print("")
    print("Master mensual analysis guardado en:")
    print(output_analysis)
    print("")
    print(master_analysis.head(12))
    print("")
    print("Columnas:")
    print(master_analysis.columns.tolist())

    return master_mensual, master_analysis


# =========================
# 5B. VARIABLE FINAL DECISIONS
# =========================

def build_capa1_variable_final_decisions() -> pd.DataFrame:
    _ensure_dirs()

    records = [
        # =========================
        # CONTEXTO
        # =========================
        {
            "dataset": "contexto_digitalizacion_clean",
            "variable": "anio",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Eje temporal anual del bloque de contexto.",
        },
        {
            "dataset": "contexto_digitalizacion_clean",
            "variable": "pct_usuarios_rrss",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Indicador clave de digitalización social.",
        },
        {
            "dataset": "contexto_digitalizacion_clean",
            "variable": "pct_personas_compra_online",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Indicador clave del consumo digital.",
        },
        {
            "dataset": "contexto_digitalizacion_extended",
            "variable": "pct_personas_compra_ropa_online",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "no",
            "usar_solo_contexto": "si",
            "comentario": "Variable temática relevante, pero con cobertura insuficiente para el master principal.",
        },
        {
            "dataset": "contexto_digitalizacion_documentado",
            "variable": "comentarios_hitos",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "no",
            "usar_solo_contexto": "si",
            "comentario": "Útil para narrativa e interpretación, no para análisis cuantitativo.",
        },
        {
            "dataset": "contexto_digitalizacion_documentado",
            "variable": "fuentes_documentales",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "no",
            "usar_solo_contexto": "si",
            "comentario": "Trazabilidad metodológica de los datos.",
        },

        # =========================
        # COMERCIO ELECTRÓNICO
        # =========================
        {
            "dataset": "comercio_electronico_core_std",
            "variable": "anio",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Eje temporal anual del ecommerce empresarial.",
        },
        {
            "dataset": "comercio_electronico_core_std",
            "variable": "indicador_std",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Variable clave para unificar indicadores entre años.",
        },
        {
            "dataset": "comercio_electronico_core_std",
            "variable": "tamano_empresa",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Permite segmentación empresarial; en el master integrado se usa principalmente total.",
        },
        {
            "dataset": "comercio_electronico_core_std",
            "variable": "valor",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Magnitud principal del bloque ecommerce.",
        },
        {
            "dataset": "comercio_electronico_core_std",
            "variable": "indicador",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "no",
            "usar_solo_contexto": "si",
            "comentario": "Se conserva por trazabilidad del indicador original.",
        },

        # =========================
        # MASTER ANUAL
        # =========================
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "anio",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Eje temporal del master anual.",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_usuarios_rrss",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Indicador principal de digitalización.",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_personas_compra_online",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Indicador principal de consumo online.",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_facturacion_empresas_online",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Indicador principal de penetración económica del canal online.",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_empresas_venden_ecommerce",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Indicador de adopción empresarial del ecommerce.",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_empresas_venden_web_apps",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Indicador complementario de canal.",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_ventas_ecommerce_sobre_total",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Mide el peso del ecommerce sobre las ventas totales.",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_ventas_ecommerce_sobre_total_empresas_que_venden",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Refina el peso del ecommerce sobre empresas activas en el canal.",
        },

        # =========================
        # MASTER MENSUAL
        # =========================
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "fecha",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Eje temporal mensual.",
        },
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "anio",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Variable auxiliar temporal.",
        },
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "mes",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Variable auxiliar temporal para estacionalidad.",
        },
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "indice_retail_moda",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Serie mensual principal del sector moda.",
        },
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "indice_retail_total",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Serie de referencia del retail agregado.",
        },
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "ratio_moda_vs_total",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Indicador derivado de rendimiento relativo.",
        },
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "dif_moda_vs_total",
            "mantener_en_base": "si",
            "usar_en_analisis_principal": "si",
            "usar_solo_contexto": "no",
            "comentario": "Indicador derivado de brecha absoluta.",
        },
    ]

    df = pd.DataFrame(records)

    output_path = TABLES_CAPA1_CONTROL / "capa1_variable_final_decisions.csv"
    df.to_csv(output_path, index=False)

    print("Decisiones finales de variables guardadas en:")
    print(output_path)
    print("")
    print(df.head(15))

    return df


# =========================
# 5C. EXPORT MASTER PREVIEWS
# =========================

def export_capa1_master_previews() -> None:
    _ensure_dirs()

    inventory_path = PROCESSED_CAPA1 / "integrated" / "capa1_inventory.csv"
    anual_path = PROCESSED_CAPA1 / "integrated" / "capa1_master_anual_analysis.csv"
    mensual_path = PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_analysis.csv"

    inventory = pd.read_csv(inventory_path)
    master_anual = pd.read_csv(anual_path)
    master_mensual = pd.read_csv(mensual_path)

    inventory.head(10).to_csv(TABLES_CAPA1_MASTERS / "capa1_inventory_head.csv", index=False)
    master_anual.head(10).to_csv(TABLES_CAPA1_MASTERS / "capa1_master_anual_head.csv", index=False)
    master_mensual.head(12).to_csv(TABLES_CAPA1_MASTERS / "capa1_master_mensual_head.csv", index=False)

    print("Previews de masters guardados en:")
    print(TABLES_CAPA1_MASTERS)


# =========================
# 6. BUILD SQLITE
# =========================

def build_capa1_sqlite() -> None:
    _ensure_dirs()

    tables = {
        "contexto_digitalizacion_clean": PROCESSED_CAPA1 / "contexto_digitalizacion" / "contexto_digitalizacion_clean.csv",
        "contexto_digitalizacion_documentado": PROCESSED_CAPA1 / "contexto_digitalizacion" / "contexto_digitalizacion_documentado.csv",
        "contexto_digitalizacion_extended": PROCESSED_CAPA1 / "contexto_digitalizacion" / "contexto_digitalizacion_extended.csv",
        "eurostat_moda_mensual_clean": PROCESSED_CAPA1 / "eurostat" / "eurostat_moda_mensual_clean.csv",
        "eurostat_retail_total_mensual_clean": PROCESSED_CAPA1 / "eurostat" / "eurostat_retail_total_mensual_clean.csv",
        "eurostat_online_empresas_clean": PROCESSED_CAPA1 / "eurostat" / "eurostat_online_empresas_clean.csv",
        "comercio_electronico_core_std": PROCESSED_CAPA1 / "comercio_electronico" / "comercio_electronico_core_std.csv",
        "capa1_inventory": PROCESSED_CAPA1 / "integrated" / "capa1_inventory.csv",
        "capa1_master_anual_full": PROCESSED_CAPA1 / "integrated" / "capa1_master_anual_full.csv",
        "capa1_master_anual_analysis": PROCESSED_CAPA1 / "integrated" / "capa1_master_anual_analysis.csv",
        "capa1_master_mensual_full": PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_full.csv",
        "capa1_master_mensual_analysis": PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_analysis.csv",
    }

    conn = sqlite3.connect(DB_CAPA1)

    for table_name, path in tables.items():
        df = pd.read_csv(path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Tabla cargada: {table_name}")

    conn.close()

    print("")
    print("Base SQLite creada en:")
    print(DB_CAPA1)


# =========================
# RUN ALL
# =========================

def run_all_builds() -> None:
    build_variable_selection_matrix_capa1()
    build_comercio_core()
    standardize_comercio_core()
    build_capa1_inventory()
    build_capa1_master_anual()
    build_capa1_master_mensual()
    build_capa1_variable_final_decisions()
    export_capa1_master_previews()
    build_capa1_sqlite()
    print("Todos los builds de capa 1 completados.")


if __name__ == "__main__":
    run_all_builds()