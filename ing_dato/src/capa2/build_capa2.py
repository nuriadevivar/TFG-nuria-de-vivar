import sqlite3
import pandas as pd
import numpy as np

from src.common.config import (
    DB_CAPA2,
    PROCESSED_CAPA2,
    TABLES_CAPA2,
    TABLES_CAPA2_CONTROL,
    TABLES_CAPA2_MASTERS,
)


# =========================
# HELPERS
# =========================

def _ensure_dirs() -> None:
    (PROCESSED_CAPA2 / "googletrends").mkdir(parents=True, exist_ok=True)
    (PROCESSED_CAPA2 / "eventos").mkdir(parents=True, exist_ok=True)
    (PROCESSED_CAPA2 / "apify").mkdir(parents=True, exist_ok=True)
    (PROCESSED_CAPA2 / "integrated").mkdir(parents=True, exist_ok=True)

    TABLES_CAPA2.mkdir(parents=True, exist_ok=True)
    TABLES_CAPA2_CONTROL.mkdir(parents=True, exist_ok=True)
    TABLES_CAPA2_MASTERS.mkdir(parents=True, exist_ok=True)

    DB_CAPA2.parent.mkdir(parents=True, exist_ok=True)


def _normalize_month_start(df: pd.DataFrame, fecha_col: str = "fecha") -> pd.DataFrame:
    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col], errors="coerce")
    df[fecha_col] = df[fecha_col].dt.to_period("M").dt.to_timestamp()
    return df


# =========================
# TAXONOMÍA DE TÉRMINOS
# =========================

# Mapa de clasificación semántica de términos de Google Trends.
# Asigna a cada término su subgrupo, tipo y familia analítica.
# Permite segmentar el análisis por naturaleza del término
# (marca, estética o comportamiento de consumo) sin depender
# del grupo de extracción original de pytrends.
TERM_MAP = {
    "zara": {"subgrupo": "marca_fast_fashion", "tipo_termino": "marca", "familia_analitica": "marca"},
    "mango": {"subgrupo": "marca_mainstream", "tipo_termino": "marca", "familia_analitica": "marca"},
    "pull and bear": {"subgrupo": "marca_fast_fashion", "tipo_termino": "marca", "familia_analitica": "marca"},
    "shein": {"subgrupo": "marca_ultra_fast_fashion", "tipo_termino": "marca", "familia_analitica": "marca"},
    "stradivarius": {"subgrupo": "marca_fast_fashion", "tipo_termino": "marca", "familia_analitica": "marca"},
    "hm": {"subgrupo": "marca_fast_fashion", "tipo_termino": "marca", "familia_analitica": "marca"},
    "massimo_dutti": {"subgrupo": "marca_premium_mainstream", "tipo_termino": "marca", "familia_analitica": "marca"},
    "cayetana": {"subgrupo": "estetica_sofisticada", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "pija": {"subgrupo": "estetica_sofisticada", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "old money": {"subgrupo": "estetica_sofisticada", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "moda preppy": {"subgrupo": "estetica_sofisticada", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "minimalista elegante": {"subgrupo": "estetica_sofisticada", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "choni": {"subgrupo": "estetica_urbana", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "trap style": {"subgrupo": "estetica_urbana", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "y2k outfit": {"subgrupo": "estetica_urbana", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "streetwear": {"subgrupo": "estetica_urbana", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "moda sostenible": {"subgrupo": "sostenibilidad", "tipo_termino": "comportamiento", "familia_analitica": "sostenibilidad"},
    "slow fashion": {"subgrupo": "sostenibilidad", "tipo_termino": "comportamiento", "familia_analitica": "sostenibilidad"},
    "ropa vintage": {"subgrupo": "segunda_mano_vintage", "tipo_termino": "comportamiento", "familia_analitica": "segunda_mano_vintage"},
    "comprar ropa online": {"subgrupo": "canal_compra_online", "tipo_termino": "comportamiento", "familia_analitica": "consumo_online"},
    "zara online": {"subgrupo": "canal_compra_online", "tipo_termino": "comportamiento", "familia_analitica": "consumo_online"},
}


def classify_term(termino: str, grupo: str) -> tuple[str, str, str]:
    # Clasifica un término usando TERM_MAP como primera opción.
    # Si no está en el mapa, aplica reglas de fallback por grupo
    # para garantizar que todos los términos tengan clasificación.
    termino_norm = str(termino).strip().lower()
    grupo_norm = str(grupo).strip().lower()

    if termino_norm in TERM_MAP:
        return (
            TERM_MAP[termino_norm]["subgrupo"],
            TERM_MAP[termino_norm]["tipo_termino"],
            TERM_MAP[termino_norm]["familia_analitica"],
        )

    # Fallback por grupo cuando el término no está en el mapa explícito
    if grupo_norm == "marcas":
        return "marca_otra", "marca", "marca"
    if grupo_norm in ["sofisticado", "urbano"]:
        return "estetica_otra", "estetica", "estetica"
    if grupo_norm == "consciente_compra":
        return "consumo_otro", "comportamiento", "comportamiento"

    return "sin_clasificar", "sin_clasificar", "sin_clasificar"


# =========================
# TAXONOMÍA DE EVENTOS
# =========================

def normalize_platform(value: str) -> str:
    # Normaliza el nombre de plataforma a un identificador canónico.
    # El campo plataforma en eventos_moda puede tener variantes de texto libre
    # ("Instagram", "IG", "instagram") que se unifican para facilitar agrupaciones.
    text = str(value).strip().lower()
    if "instagram" in text:
        return "instagram"
    if "tiktok" in text:
        return "tiktok"
    if "youtube" in text:
        return "youtube"
    if "app" in text:
        return "app"
    return "general"


def classify_event_category(row: pd.Series) -> str:
    # Clasifica un evento en una categoría analítica a partir de palabras clave
    # en el texto combinado de marca, tipo y descripción del evento.
    # El orden de evaluación importa: las categorías más específicas
    # (lanzamiento, estética) se evalúan antes que las más genéricas.
    text = " ".join(
        [
            str(row.get("marca_o_tendencia", "")),
            str(row.get("tipo_evento", "")),
            str(row.get("descripcion_evento", "")),
        ]
    ).lower()

    if any(k in text for k in ["colección", "coleccion", "campaña", "campana", "desfile", "lanzamiento", "drop", "colaboración", "colaboracion"]):
        return "lanzamiento_campana"

    if any(k in text for k in ["old money", "preppy", "pija", "cayetana", "streetwear", "trap", "y2k", "choni"]):
        return "estetica_tendencia"

    if any(k in text for k in ["haul", "hauls"]):
        return "haul_viralidad"

    if any(k in text for k in ["segunda mano", "vintage", "resale", "thrift"]):
        return "segunda_mano_vintage"

    if any(k in text for k in ["slow fashion", "moda sostenible", "sostenibilidad", "sostenible"]):
        return "sostenibilidad"

    return "otro"


# =========================
# 0. VARIABLE SELECTION MATRIX
# =========================

def build_variable_selection_matrix_capa2() -> pd.DataFrame:
    _ensure_dirs()

    # Documenta la decisión de incluir o excluir cada variable de cada dataset
    # en el master analítico. Cubre todas las fuentes de la Capa 2: Google Trends,
    # eventos, Instagram y masters integrados. Es la evidencia documental de
    # la selección de variables para el corrector y el tribunal.
    records = [
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "fecha",
            "descripcion": "Mes normalizado de observación",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Variable temporal principal para el análisis de tendencias mensuales",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "anio",
            "descripcion": "Año derivado de fecha",
            "tipo_variable": "tecnica",
            "rol_analitico": "join_temporal",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener",
            "justificacion": "Facilita agregaciones anuales y visualizaciones sin recalcular",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "mes_num",
            "descripcion": "Número de mes derivado de fecha",
            "tipo_variable": "tecnica",
            "rol_analitico": "join_temporal",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener",
            "justificacion": "Facilita análisis de estacionalidad y pivotes mensuales",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "termino",
            "descripcion": "Término consultado en Google Trends",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Unidad semántica central del análisis digital",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "grupo",
            "descripcion": "Grupo temático del término",
            "tipo_variable": "nucleo",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Permite separar marcas, estéticas y comportamientos de consumo",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "subgrupo",
            "descripcion": "Subclasificación analítica del término",
            "tipo_variable": "auxiliar",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Aporta segmentación semántica más rica que el grupo original",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "tipo_termino",
            "descripcion": "Naturaleza del término: marca, estética o comportamiento",
            "tipo_variable": "auxiliar",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Facilita comparaciones entre tipos de interés digital",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "familia_analitica",
            "descripcion": "Agrupación analítica simplificada del término",
            "tipo_variable": "auxiliar",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Simplifica el EDA y la interpretación comparativa entre familias",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "valor_trends",
            "descripcion": "Intensidad mensual de búsqueda",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "medio",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Variable principal de intensidad de búsqueda en la capa 2",
        },
        {
            "dataset": "capa2_term_coverage",
            "variable": "grupo / termino / n_obs / n_non_null / avg_trends / pct_non_null / quality_flag",
            "descripcion": "Bloque de control de calidad y cobertura temporal por término",
            "tipo_variable": "tecnica",
            "rol_analitico": "control_calidad",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Permite evaluar qué términos tienen calidad suficiente para análisis principal",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "fecha",
            "descripcion": "Mes de observación",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Variable temporal principal para el bloque de productos",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "anio",
            "descripcion": "Año derivado",
            "tipo_variable": "tecnica",
            "rol_analitico": "join_temporal",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener",
            "justificacion": "Útil para resúmenes anuales",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "mes_num",
            "descripcion": "Mes derivado",
            "tipo_variable": "tecnica",
            "rol_analitico": "join_temporal",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener",
            "justificacion": "Útil para análisis estacional",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "marca",
            "descripcion": "Marca del producto",
            "tipo_variable": "nucleo",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Permite comparar productos entre marcas",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "categoria_producto",
            "descripcion": "Categoría del producto",
            "tipo_variable": "nucleo",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Permite comparar tipos de producto",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "valor_trends",
            "descripcion": "Intensidad de interés del producto",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Variable principal del análisis de productos",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "termino_busqueda",
            "descripcion": "Texto completo de búsqueda original",
            "tipo_variable": "tecnica",
            "rol_analitico": "trazabilidad",
            "nivel_nulos": "bajo",
            "redundancia": "alta",
            "decision": "mantener_solo_trazabilidad",
            "justificacion": "Se conserva en el transformado por auditoría, pero no en el master analítico por redundancia",
        },
        {
            "dataset": "capa2_master_eventos",
            "variable": "fecha_aprox / fecha / anio / mes_num / marca_o_tendencia / plataforma / plataforma_std / tipo_evento / categoria_evento / descripcion_evento / fuente",
            "descripcion": "Bloque contextual de hitos y eventos de moda",
            "tipo_variable": "contextual",
            "rol_analitico": "contexto",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener_contexto",
            "justificacion": "Sirve como soporte interpretativo y de trazabilidad, no como núcleo cuantitativo principal",
        },
        {
            "dataset": "capa2_master_eventos_mensual",
            "variable": "fecha / anio / mes_num / n_eventos_total / n_plataformas / n_tipos_evento",
            "descripcion": "Resumen mensual agregado de hitos",
            "tipo_variable": "contextual",
            "rol_analitico": "contexto",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener_contexto",
            "justificacion": "Aporta una señal contextual general del mes, sin pretensión de análisis causal fuerte",
        },
        {
            "dataset": "capa2_master_integrated",
            "variable": "bloque_base_terminos + bloque_contextual_eventos_mensuales",
            "descripcion": "Dataset enriquecido con contexto mensual general",
            "tipo_variable": "mixta",
            "rol_analitico": "analisis_principal + contexto",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "revisar_tras_eda",
            "justificacion": "Debe mantenerse solo si las variables contextuales de eventos aportan valor interpretativo real",
        },
        {
            "dataset": "capa2_master_social",
            "variable": "fecha / anio / mes_num / marca / plataforma / n_posts / likes_totales / comentarios_totales / engagement_total / likes_medios_post / comentarios_medios_post / engagement_medio_post",
            "descripcion": "Resumen mensual de actividad y engagement en Instagram por marca",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Aporta la señal social agregada de marca necesaria para integrar Apify en la capa 2",
        },
        {
            "dataset": "capa2_master_brand_digital",
            "variable": "bloque_trends_marcas + bloque_social + bloque_contextual_eventos",
            "descripcion": "Tabla final integrada marca-mes con Google Trends, Instagram y contexto mensual",
            "tipo_variable": "mixta",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener",
            "justificacion": "Es la tabla final más útil para analizar conjuntamente interés digital y actividad social de marca",
        },
    ]

    df = pd.DataFrame(records)

    output_path = TABLES_CAPA2_CONTROL / "capa2_variable_selection_matrix.csv"
    df.to_csv(output_path, index=False)

    print("Matriz de selección de variables guardada en:")
    print(output_path)
    print("")
    print(df.head(15))

    return df


# =========================
# 1. INVENTARIO
# =========================

def build_capa2_inventory() -> pd.DataFrame:
    _ensure_dirs()

    # Inventario de todos los datasets procesados de la Capa 2.
    # Documenta fuente, frecuencia, cobertura temporal y rol analítico
    # de cada dataset. Punto de entrada para entender qué alimenta el master.
    inventory = pd.DataFrame(
        [
            {
                "dataset": "processed/capa2/googletrends/trends_moda_total_clean.csv",
                "fuente": "Google Trends",
                "frecuencia": "mensual",
                "periodo_inicio": "2014-12",
                "periodo_fin": "2025-12",
                "granularidad": "España - término/mes",
                "rol_capa2": "base agregada de tendencias",
            },
            {
                "dataset": "processed/capa2/googletrends/trends_marcas_clean.csv",
                "fuente": "Google Trends",
                "frecuencia": "mensual",
                "periodo_inicio": "2014-12",
                "periodo_fin": "2025-12",
                "granularidad": "España - marca/mes",
                "rol_capa2": "interés digital por marcas",
            },
            {
                "dataset": "processed/capa2/googletrends/trends_sofisticado_clean.csv",
                "fuente": "Google Trends",
                "frecuencia": "mensual",
                "periodo_inicio": "2014-12",
                "periodo_fin": "2025-12",
                "granularidad": "España - estética/mes",
                "rol_capa2": "estéticas sofisticadas",
            },
            {
                "dataset": "processed/capa2/googletrends/trends_urbano_clean.csv",
                "fuente": "Google Trends",
                "frecuencia": "mensual",
                "periodo_inicio": "2014-12",
                "periodo_fin": "2025-12",
                "granularidad": "España - estética/mes",
                "rol_capa2": "estéticas urbanas",
            },
            {
                "dataset": "processed/capa2/googletrends/trends_consciente_compra_clean.csv",
                "fuente": "Google Trends",
                "frecuencia": "mensual",
                "periodo_inicio": "2014-12",
                "periodo_fin": "2025-12",
                "granularidad": "España - término/mes",
                "rol_capa2": "consumo y sostenibilidad",
            },
            {
                "dataset": "processed/capa2/googletrends/trends_productos_clean.csv",
                "fuente": "Google Trends",
                "frecuencia": "mensual",
                "periodo_inicio": "2015-01",
                "periodo_fin": "2025-12",
                "granularidad": "España - producto/mes",
                "rol_capa2": "interés por productos",
            },
            {
                "dataset": "processed/capa2/googletrends/trends_grupos_unificados_clean.csv",
                "fuente": "Google Trends",
                "frecuencia": "mensual",
                "periodo_inicio": "2014-12",
                "periodo_fin": "2025-12",
                "granularidad": "España - término/grupo/mes",
                "rol_capa2": "base unificada de términos",
            },
            {
                "dataset": "processed/capa2/eventos/eventos_moda_clean.csv",
                "fuente": "Curación manual a partir de fuentes públicas",
                "frecuencia": "puntual agregada a mes",
                "periodo_inicio": "2015-10",
                "periodo_fin": "2025-01",
                "granularidad": "evento",
                "rol_capa2": "contextualización de tendencias",
            },
            {
                "dataset": "processed/capa2/apify/instagram_posts_clean.csv",
                "fuente": "Apify - Instagram Scraper",
                "frecuencia": "post",
                "periodo_inicio": "2025-01",
                "periodo_fin": "2026-03",
                "granularidad": "post",
                "rol_capa2": "actividad social por post",
            },
            {
                "dataset": "processed/capa2/apify/instagram_brand_monthly.csv",
                "fuente": "Apify - Instagram Scraper",
                "frecuencia": "mensual",
                "periodo_inicio": "2025-01",
                "periodo_fin": "2026-03",
                "granularidad": "marca-mes",
                "rol_capa2": "actividad social agregada por marca",
            },
        ]
    )

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_inventory.csv"
    inventory.to_csv(output_path, index=False)

    print("Inventario guardado en:")
    print(output_path)
    print("")
    print(inventory)

    return inventory


# =========================
# 2. MASTER TÉRMINOS MENSUAL
# =========================

def build_capa2_master_terminos() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "googletrends" / "trends_grupos_unificados_clean.csv"
    df = pd.read_csv(input_path)

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = _normalize_month_start(df, "fecha")
    # Restricción temporal: solo se incluyen datos desde 2015,
    # ventana analítica del TFG que garantiza comparabilidad con la Capa 1
    df = df[df["fecha"].dt.year.between(2015, 2025)].copy().reset_index(drop=True)

    df["termino"] = df["termino"].astype(str).str.strip().str.lower()
    df["grupo"] = df["grupo"].astype(str).str.strip().str.lower()
    df["valor_trends"] = pd.to_numeric(df["valor_trends"], errors="coerce")

    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month

    # Enriquecimiento semántico: clasifica cada término con subgrupo,
    # tipo y familia analítica usando el TERM_MAP definido en este script
    df[["subgrupo", "tipo_termino", "familia_analitica"]] = df.apply(
        lambda row: pd.Series(classify_term(row["termino"], row["grupo"])),
        axis=1,
    )

    # Cálculo de cobertura por término: métricas de calidad que determinan
    # si el término tiene suficiente señal para el análisis principal
    coverage = (
        df.groupby(["grupo", "termino"], dropna=False)
        .agg(
            n_obs=("valor_trends", "size"),
            n_non_null=("valor_trends", lambda s: s.notna().sum()),
            avg_trends=("valor_trends", "mean"),
        )
        .reset_index()
    )
    coverage["pct_non_null"] = (coverage["n_non_null"] / coverage["n_obs"] * 100).round(2)

    def quality_flag(pct: float) -> str:
        # Clasifica la calidad de cobertura temporal de cada término:
        # alta (≥90% de meses con dato), media (60-90%), baja (<60%)
        if pct >= 90:
            return "alta"
        if pct >= 60:
            return "media"
        return "baja"

    coverage["quality_flag"] = coverage["pct_non_null"].apply(quality_flag)

    coverage_path = PROCESSED_CAPA2 / "integrated" / "capa2_term_coverage.csv"
    coverage.to_csv(coverage_path, index=False)

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_mensual.csv"
    df.to_csv(output_path, index=False)

    print("Master términos mensual guardado en:")
    print(output_path)
    print("")
    print("Cobertura de términos guardada en:")
    print(coverage_path)
    print("")
    print(df.head())

    return df


# =========================
# 2B. PRIORIDAD ANALÍTICA DE TÉRMINOS
# =========================

def build_term_analysis_priority() -> pd.DataFrame:
    _ensure_dirs()

    coverage_path = PROCESSED_CAPA2 / "integrated" / "capa2_term_coverage.csv"
    coverage = pd.read_csv(coverage_path)

    coverage["grupo"] = coverage["grupo"].astype(str).str.strip().str.lower()
    coverage["termino"] = coverage["termino"].astype(str).str.strip().str.lower()

    # Términos de prioridad principal: incluidos en el análisis central del TFG.
    # Son las marcas principales y los términos estéticos y de comportamiento
    # con mayor relevancia para las preguntas de investigación.
    principal_terms = {
        "ropa vintage",
        "comprar ropa online",
        "zara online",
        "moda sostenible",
        "zara",
        "mango",
        "shein",
        "hm",
        "massimo_dutti",
        "choni",
        "streetwear",
        "old money",
    }

    # Términos secundarios: relevantes pero con menor continuidad temporal
    # o señal más débil. Se incluyen con cautela.
    secondary_terms = {
        "slow fashion",
        "pija",
        "cayetana",
    }

    # Términos exploratorios: señal muy débil o cobertura insuficiente.
    # Se mantienen en el dataset base por trazabilidad pero no en el análisis principal.
    exploratory_terms = {
        "y2k outfit",
        "trap style",
        "moda preppy",
        "minimalista elegante",
    }

    def assign_priority(row: pd.Series) -> str:
        # Asigna prioridad analítica combinando la lista explícita de términos
        # con criterios de quality_flag y avg_trends para términos no listados
        termino = row["termino"]
        avg_trends = row.get("avg_trends", np.nan)
        quality_flag = row.get("quality_flag", "")

        if termino in principal_terms:
            return "principal"
        if termino in secondary_terms:
            return "secundario"
        if termino in exploratory_terms:
            return "exploratorio"

        # Criterios automáticos para términos no explícitamente clasificados
        if quality_flag == "alta" and pd.notna(avg_trends) and avg_trends >= 20:
            return "principal"
        if quality_flag in ["alta", "media"] and pd.notna(avg_trends) and avg_trends >= 8:
            return "secundario"
        return "exploratorio"

    coverage["prioridad_analitica"] = coverage.apply(assign_priority, axis=1)

    def assign_decision(priority: str) -> str:
        # Traduce la prioridad analítica a una decisión de uso operativa
        if priority == "principal":
            return "usar_en_analisis_principal"
        if priority == "secundario":
            return "usar_en_analisis_secundario"
        return "usar_solo_exploratorio"

    coverage["decision_uso"] = coverage["prioridad_analitica"].apply(assign_decision)

    def assign_comment(row: pd.Series) -> str:
        # Genera un comentario metodológico específico para términos con
        # particularidades relevantes para el corrector o el analista
        termino = row["termino"]
        quality_flag = row["quality_flag"]
        avg_trends = row["avg_trends"]

        if termino == "slow fashion":
            return "Cobertura alta pero señal débil; mantener como término exploratorio o secundario según contexto."
        if termino in {"y2k outfit", "trap style", "moda preppy", "minimalista elegante"}:
            return "Cobertura y señal débiles; mantener solo para exploración, no para análisis principal."
        if termino in {"pija", "cayetana"}:
            return "Interés conceptual alto, pero continuidad temporal limitada; usar con prudencia."
        if quality_flag == "alta" and pd.notna(avg_trends) and avg_trends >= 20:
            return "Término robusto por cobertura y señal media; apto para análisis principal."
        if quality_flag == "media":
            return "Término útil, pero con robustez intermedia; conviene interpretarlo con cautela."
        return "Término de soporte o exploración; no conviene sobrerrepresentarlo en conclusiones."

    coverage["comentario"] = coverage.apply(assign_comment, axis=1)

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_term_analysis_priority.csv"
    coverage.to_csv(output_path, index=False)

    control_output_path = TABLES_CAPA2_CONTROL / "capa2_term_analysis_priority.csv"
    coverage.to_csv(control_output_path, index=False)

    print("Prioridad analítica de términos guardada en:")
    print(output_path)
    print("")
    print(
        coverage[
            [
                "grupo",
                "termino",
                "avg_trends",
                "pct_non_null",
                "quality_flag",
                "prioridad_analitica",
                "decision_uso",
            ]
        ]
        .sort_values(["prioridad_analitica", "avg_trends"], ascending=[True, False])
        .head(20)
    )

    return coverage


# =========================
# 2C. DECISIONES DE CALIDAD DE TÉRMINOS
# =========================

def build_term_quality_decisions() -> pd.DataFrame:
    _ensure_dirs()

    priority_path = PROCESSED_CAPA2 / "integrated" / "capa2_term_analysis_priority.csv"
    df = pd.read_csv(priority_path)

    # Selecciona las columnas relevantes de la tabla de prioridad
    # y añade columnas operativas de uso para cada término
    decisions = df[
        [
            "grupo",
            "termino",
            "n_obs",
            "n_non_null",
            "avg_trends",
            "pct_non_null",
            "quality_flag",
            "prioridad_analitica",
            "decision_uso",
            "comentario",
        ]
    ].copy()

    # Todos los términos se mantienen en el dataset base independientemente de su prioridad.
    # La columna usar_en_analisis_principal filtra los que entran en el análisis central.
    decisions["mantener_en_base"] = "si"
    decisions["usar_en_analisis_principal"] = decisions["prioridad_analitica"].apply(
        lambda x: "si" if x == "principal" else "no"
    )
    decisions["usar_solo_exploratorio"] = decisions["prioridad_analitica"].apply(
        lambda x: "si" if x == "exploratorio" else "no"
    )

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_term_quality_decisions.csv"
    decisions.to_csv(output_path, index=False)

    control_output_path = TABLES_CAPA2_CONTROL / "capa2_term_quality_decisions.csv"
    decisions.to_csv(control_output_path, index=False)

    print("Decisiones de calidad de términos guardadas en:")
    print(output_path)
    print("")
    print(decisions.head(20))

    return decisions


# =========================
# 2D. MASTER TÉRMINOS MAIN
# =========================

def build_capa2_master_terminos_main() -> pd.DataFrame:
    _ensure_dirs()

    terms_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_mensual.csv"
    priority_path = PROCESSED_CAPA2 / "integrated" / "capa2_term_analysis_priority.csv"

    terms = pd.read_csv(terms_path)
    priority = pd.read_csv(priority_path)

    terms["termino"] = terms["termino"].astype(str).str.strip().str.lower()
    priority["termino"] = priority["termino"].astype(str).str.strip().str.lower()

    # Subset de términos de prioridad principal y secundaria para el análisis central.
    # Los términos exploratorios se excluyen aquí pero permanecen en el master completo.
    keep_terms = priority[
        priority["prioridad_analitica"].isin(["principal", "secundario"])
    ]["termino"].unique()

    main_df = terms[terms["termino"].isin(keep_terms)].copy()

    # Enriquece el subset con la prioridad y decisión de uso de cada término
    main_df = main_df.merge(
        priority[["termino", "prioridad_analitica", "decision_uso"]],
        on="termino",
        how="left",
    )

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_main.csv"
    main_df.to_csv(output_path, index=False)

    print("Master términos main guardado en:")
    print(output_path)
    print("")
    print(main_df.head())

    return main_df


# =========================
# 3. MASTER PRODUCTOS
# =========================

def build_capa2_master_productos() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "googletrends" / "trends_productos_clean.csv"
    df = pd.read_csv(input_path)

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = _normalize_month_start(df, "fecha")
    df = df[df["fecha"].dt.year.between(2015, 2025)].copy().reset_index(drop=True)

    df["marca"] = df["marca"].astype(str).str.strip().str.lower()
    df["categoria_producto"] = df["categoria_producto"].astype(str).str.strip().str.lower()
    df["valor_trends"] = pd.to_numeric(df["valor_trends"], errors="coerce")

    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month

     # Se excluye termino_busqueda del master analítico por redundancia con
    # categoria_producto; se mantiene en el dataset transformado por trazabilidad
    df = df[
        [
            "fecha",
            "anio",
            "mes_num",
            "marca",
            "categoria_producto",
            "valor_trends",
        ]
    ].copy()

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_productos_mensual.csv"
    df.to_csv(output_path, index=False)

    print("Master productos mensual guardado en:")
    print(output_path)
    print("")
    print(df.head())

    return df


# =========================
# 4. MASTER EVENTOS
# =========================

def build_capa2_master_eventos() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "eventos" / "eventos_moda_clean.csv"
    df = pd.read_csv(input_path)

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = _normalize_month_start(df, "fecha")

    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month

    text_cols = ["marca_o_tendencia", "plataforma", "tipo_evento", "descripcion_evento", "fuente"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    # Enriquecimiento: normaliza la plataforma a identificador canónico
    # y clasifica cada evento en una categoría analítica usando palabras clave
    df["plataforma_std"] = df["plataforma"].apply(normalize_platform)
    df["categoria_evento"] = df.apply(classify_event_category, axis=1)

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos.csv"
    df.to_csv(output_path, index=False)

    print("Master eventos guardado en:")
    print(output_path)
    print("")
    print(df.head())

    return df


# =========================
# 5. MASTER EVENTOS MENSUAL
# =========================

def build_capa2_master_eventos_mensual() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos.csv"
    df = pd.read_csv(input_path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # Agrega los eventos a nivel mensual: cuenta el total de eventos,
    # el número de plataformas distintas y los tipos de evento por mes.
    # Estas métricas se usan como señal contextual en el master integrado.
    monthly = (
        df.groupby("fecha")
        .agg(
            n_eventos_total=("fecha", "size"),
            n_plataformas=("plataforma_std", "nunique"),
            n_tipos_evento=("tipo_evento", "nunique"),
        )
        .reset_index()
    )

    monthly["anio"] = monthly["fecha"].dt.year
    monthly["mes_num"] = monthly["fecha"].dt.month

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos_mensual.csv"
    monthly.to_csv(output_path, index=False)

    print("Master eventos mensual guardado en:")
    print(output_path)
    print("")
    print(monthly.head())

    return monthly


# =========================
# 6. MASTER INTEGRATED
# =========================

def build_capa2_master_integrated() -> pd.DataFrame:
    _ensure_dirs()

    terms_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_mensual.csv"
    eventos_monthly_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos_mensual.csv"

    terms = pd.read_csv(terms_path)
    eventos = pd.read_csv(eventos_monthly_path)

    terms["fecha"] = pd.to_datetime(terms["fecha"], errors="coerce")
    eventos["fecha"] = pd.to_datetime(eventos["fecha"], errors="coerce")

    # Left join: todos los términos se conservan aunque no haya eventos ese mes.
    # Los meses sin eventos reciben 0 en las columnas de contexto (no NaN),
    # lo que facilita el análisis sin necesidad de gestionar nulos adicionales.
    master = terms.merge(eventos, on=["fecha", "anio", "mes_num"], how="left")

    # Rellena con 0 los meses sin eventos: la ausencia de eventos es información
    # válida (mes sin hitos documentados), no un dato faltante
    context_cols = ["n_eventos_total", "n_plataformas", "n_tipos_evento"]
    for col in context_cols:
        master[col] = master[col].fillna(0)

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_integrated.csv"
    master.to_csv(output_path, index=False)

    print("Master integrated guardado en:")
    print(output_path)
    print("")
    print(master.head())

    return master


# =========================
# 7. MASTER SOCIAL
# =========================

def build_capa2_master_social() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "apify" / "instagram_brand_monthly.csv"
    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_social.csv"

    df = pd.read_csv(input_path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    df["marca"] = df["marca"].astype(str).str.strip().str.lower()
    df["plataforma"] = df["plataforma"].astype(str).str.strip().str.lower()

    # Conversión explícita a numérico para garantizar tipos correctos
    # antes de exportar al master social
    numeric_cols = [
        "n_posts",
        "likes_totales",
        "comentarios_totales",
        "engagement_total",
        "likes_medios_post",
        "comentarios_medios_post",
        "engagement_medio_post",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["marca", "fecha"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print("Master social guardado en:")
    print(output_path)
    print("")
    print(df.head())

    return df


# =========================
# 8. MASTER BRAND DIGITAL
# =========================

def build_capa2_master_brand_digital() -> pd.DataFrame:
    _ensure_dirs()

    trends_path = PROCESSED_CAPA2 / "googletrends" / "trends_marcas_clean.csv"
    social_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_social.csv"
    eventos_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos_mensual.csv"

    trends = pd.read_csv(trends_path)
    social = pd.read_csv(social_path)
    eventos = pd.read_csv(eventos_path)

    trends["fecha"] = pd.to_datetime(trends["fecha"], errors="coerce")
    social["fecha"] = pd.to_datetime(social["fecha"], errors="coerce")
    eventos["fecha"] = pd.to_datetime(eventos["fecha"], errors="coerce")

    trends["termino"] = trends["termino"].astype(str).str.strip().str.lower()
    social["marca"] = social["marca"].astype(str).str.strip().str.lower()

    # Renombra 'termino' a 'marca' en trends para poder hacer el join con social
    trends_brand = trends.rename(columns={"termino": "marca"}).copy()
    trends_brand = trends_brand[["fecha", "anio", "mes_num", "marca", "valor_trends"]].copy()

    social = social[
        [
            "fecha",
            "anio",
            "mes_num",
            "marca",
            "plataforma",
            "n_posts",
            "likes_totales",
            "comentarios_totales",
            "engagement_total",
            "likes_medios_post",
            "comentarios_medios_post",
            "engagement_medio_post",
        ]
    ].copy()

    # Join 1: Google Trends (histórico largo) + Instagram (ventana reciente).
    # Left join desde trends: la cobertura temporal de Google Trends es mayor,
    # por lo que los meses anteriores a la ventana de Instagram tendrán NaN en social.
    master = trends_brand.merge(
        social,
        on=["fecha", "anio", "mes_num", "marca"],
        how="left",
    )

    # Join 2: añade contexto mensual de eventos
    master = master.merge(
        eventos[["fecha", "n_eventos_total", "n_plataformas", "n_tipos_evento"]],
        on="fecha",
        how="left",
    )

    # Los meses fuera de la ventana de Instagram o sin eventos reciben 0.
    # Esto es metodológicamente correcto: los meses previos al scraping
    # no tienen actividad social registrada (no es un dato faltante).
    fill_zero_cols = [
        "n_posts",
        "likes_totales",
        "comentarios_totales",
        "engagement_total",
        "likes_medios_post",
        "comentarios_medios_post",
        "engagement_medio_post",
        "n_eventos_total",
        "n_plataformas",
        "n_tipos_evento",
    ]
    for col in fill_zero_cols:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce").fillna(0)

    if "plataforma" in master.columns:
        master["plataforma"] = master["plataforma"].fillna("instagram")

    master = master.sort_values(["marca", "fecha"]).reset_index(drop=True)

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_brand_digital.csv"
    master.to_csv(output_path, index=False)

    print("Master brand digital guardado en:")
    print(output_path)
    print("")
    print(master.head())

    return master


# =========================
# 9. DECISION VARIABLES FINALES
# =========================

def build_variable_final_decisions_capa2() -> pd.DataFrame:
    _ensure_dirs()

    # Tabla operativa de decisiones finales: qué variables se usan en modelos,
    # en EDA y cuáles son solo de contexto. Complementa la variable_selection_matrix
    # con una vista más directamente orientada a la ejecución del análisis.
    records = [
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "fecha",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Eje temporal principal de la capa 2.",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "termino",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Unidad semántica principal del análisis de tendencias.",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "grupo",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Permite segmentar marcas, estéticas y comportamiento.",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "subgrupo",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Aporta una capa semántica adicional útil.",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "tipo_termino",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Facilita lectura analítica por naturaleza del término.",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "familia_analitica",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Muy útil para síntesis e interpretación.",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "valor_trends",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Variable principal de intensidad de búsqueda.",
        },
        {
            "dataset": "capa2_term_coverage",
            "variable": "quality_flag",
            "usar_en_modelos": "no",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Control de calidad para priorizar términos.",
        },
        {
            "dataset": "capa2_term_analysis_priority",
            "variable": "prioridad_analitica",
            "usar_en_modelos": "no",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Permite justificar inclusión o exclusión de términos.",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "fecha",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Eje temporal del bloque de productos.",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "marca",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Clave de segmentación por marca.",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "categoria_producto",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Clave de segmentación por tipo de producto.",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "valor_trends",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Variable central del bloque de productos.",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "termino_busqueda",
            "usar_en_modelos": "no",
            "usar_en_eda": "no",
            "usar_solo_contexto": "si",
            "comentario": "Se mantiene solo por trazabilidad.",
        },
        {
            "dataset": "capa2_master_eventos",
            "variable": "bloque_eventos",
            "usar_en_modelos": "no",
            "usar_en_eda": "si",
            "usar_solo_contexto": "si",
            "comentario": "Dataset contextual para interpretar cambios, no núcleo predictivo.",
        },
        {
            "dataset": "capa2_master_eventos_mensual",
            "variable": "n_eventos_total / n_plataformas / n_tipos_evento",
            "usar_en_modelos": "no",
            "usar_en_eda": "si",
            "usar_solo_contexto": "si",
            "comentario": "Resumen contextual mensual; útil para lectura, no necesariamente para modelización principal.",
        },
        {
            "dataset": "capa2_master_social",
            "variable": "n_posts / likes_totales / comentarios_totales / engagement_total / likes_medios_post / comentarios_medios_post / engagement_medio_post",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Bloque social principal derivado de Apify.",
        },
        {
            "dataset": "capa2_master_brand_digital",
            "variable": "valor_trends + bloque_social + bloque_eventos",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Tabla integrada final para análisis conjunto por marca-mes.",
        },
    ]

    df = pd.DataFrame(records)

    output_path = TABLES_CAPA2_CONTROL / "capa2_variable_final_decisions.csv"
    df.to_csv(output_path, index=False)

    print("Decisiones finales de variables guardadas en:")
    print(output_path)
    print("")
    print(df.head(15))

    return df


def build_capa2_master_previews() -> None:
    _ensure_dirs()

    # Exporta las primeras 15 filas de cada master como vista rápida de verificación.
    # Permite confirmar estructura y contenido sin abrir los ficheros completos.
    preview_map = {
        "capa2_inventory_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_inventory.csv",
        "capa2_master_terminos_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_mensual.csv",
        "capa2_master_terminos_main_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_main.csv",
        "capa2_master_productos_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_master_productos_mensual.csv",
        "capa2_master_eventos_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos.csv",
        "capa2_master_eventos_mensual_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos_mensual.csv",
        "capa2_master_integrated_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_master_integrated.csv",
        "capa2_master_social_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_master_social.csv",
        "capa2_master_brand_digital_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_master_brand_digital.csv",
    }

    for output_name, input_path in preview_map.items():
        df = pd.read_csv(input_path)
        df.head(15).to_csv(TABLES_CAPA2_MASTERS / output_name, index=False)

    print("Previews de masters guardados en:")
    print(TABLES_CAPA2_MASTERS)


# =========================
# 10. SQLITE
# =========================

def build_capa2_sqlite() -> None:
    _ensure_dirs()

    tables = {
        "trends_moda_total_clean": PROCESSED_CAPA2 / "googletrends" / "trends_moda_total_clean.csv",
        "trends_marcas_clean": PROCESSED_CAPA2 / "googletrends" / "trends_marcas_clean.csv",
        "trends_sofisticado_clean": PROCESSED_CAPA2 / "googletrends" / "trends_sofisticado_clean.csv",
        "trends_urbano_clean": PROCESSED_CAPA2 / "googletrends" / "trends_urbano_clean.csv",
        "trends_consciente_compra_clean": PROCESSED_CAPA2 / "googletrends" / "trends_consciente_compra_clean.csv",
        "trends_productos_clean": PROCESSED_CAPA2 / "googletrends" / "trends_productos_clean.csv",
        "trends_grupos_unificados_clean": PROCESSED_CAPA2 / "googletrends" / "trends_grupos_unificados_clean.csv",
        "eventos_moda_clean": PROCESSED_CAPA2 / "eventos" / "eventos_moda_clean.csv",
        "instagram_posts_clean": PROCESSED_CAPA2 / "apify" / "instagram_posts_clean.csv",
        "instagram_brand_monthly": PROCESSED_CAPA2 / "apify" / "instagram_brand_monthly.csv",
        "capa2_inventory": PROCESSED_CAPA2 / "integrated" / "capa2_inventory.csv",
        "capa2_term_coverage": PROCESSED_CAPA2 / "integrated" / "capa2_term_coverage.csv",
        "capa2_master_terminos_mensual": PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_mensual.csv",
        "capa2_master_productos_mensual": PROCESSED_CAPA2 / "integrated" / "capa2_master_productos_mensual.csv",
        "capa2_master_eventos": PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos.csv",
        "capa2_master_eventos_mensual": PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos_mensual.csv",
        "capa2_master_integrated": PROCESSED_CAPA2 / "integrated" / "capa2_master_integrated.csv",
        "capa2_master_social": PROCESSED_CAPA2 / "integrated" / "capa2_master_social.csv",
        "capa2_master_brand_digital": PROCESSED_CAPA2 / "integrated" / "capa2_master_brand_digital.csv",
        "capa2_term_analysis_priority": PROCESSED_CAPA2 / "integrated" / "capa2_term_analysis_priority.csv",
        "capa2_term_quality_decisions": PROCESSED_CAPA2 / "integrated" / "capa2_term_quality_decisions.csv",
        "capa2_master_terminos_main": PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_main.csv",
        "capa2_variable_selection_matrix": TABLES_CAPA2_CONTROL / "capa2_variable_selection_matrix.csv",
        "capa2_variable_final_decisions": TABLES_CAPA2_CONTROL / "capa2_variable_final_decisions.csv",
        "capa2_null_summary": TABLES_CAPA2_CONTROL / "capa2_null_summary.csv",
    }

    conn = sqlite3.connect(DB_CAPA2)

    for table_name, path in tables.items():
        df = pd.read_csv(path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Tabla cargada: {table_name}")

    conn.close()

    print("")
    print("Base SQLite creada en:")
    print(DB_CAPA2)


# =========================
# NULL SUMMARY
# =========================
 
def build_capa2_null_summary() -> pd.DataFrame:
    """
    Tabla única con todos los nulos y valores especiales documentados en capa 2.
 
    Casos documentados:
    1. valor_trends = 0 (Google Trends): no es ausencia de interés sino volumen
       de búsqueda insuficiente para ser representado en escala 0-100.
       Decisión: mantener con flag valor_cero_trends=True.
 
    2. NaN en valor_trends: dato genuinamente no disponible en la API de Google
       Trends para ese término/mes. Decisión: mantener NaN.
 
    3. NaN en likes/comentarios (Instagram Apify): la API devolvió null para ese
       campo. Decisión: imputar a 0 con flag metricas_imputadas=True.
 
    4. NaN en hashtags (Instagram): campo opcional en la API. Sin impacto analítico.
       Decisión: mantener NaN, no se usa como variable cuantitativa.
 
    5. Términos con quality_flag='baja' (cobertura < 60%): se conservan en el
       dataset pero se excluyen del análisis principal (master_terminos_main).
    """
    _ensure_dirs()
 
    # Cada registro documenta un tipo de valor especial con su naturaleza,
    # la decisión adoptada y el impacto de alternativas no elegidas.
    # Esta tabla es la evidencia documental para el corrector del tratamiento
    # de todos los casos no triviales de la Capa 2.
    records = [
        {
            "dataset": "trends_grupos_unificados_clean / trends_marcas_clean",
            "variable": "valor_trends",
            "tipo_valor_especial": "cero_GT",
            "descripcion": (
                "valor_trends=0 en Google Trends indica que el volumen de busquedas "
                "es demasiado bajo para ser representado en la escala 0-100 de la API. "
                "No equivale a ausencia de interes — es un artefacto de la normalizacion "
                "de Google Trends que escala el termino con mayor volumen a 100."
            ),
            "decision": "mantener_con_flag — valor_cero_trends=True permite filtrar en analisis sensibles",
            "impacto_si_se_imputa": "Distorsionaria la comparacion entre terminos con diferente popularidad base",
            "impacto_si_se_elimina": "Perdida de contexto temporal; podria enmascarar periodos de baja actividad real",
        },
        {
            "dataset": "trends_grupos_unificados_clean",
            "variable": "valor_trends",
            "tipo_valor_especial": "NaN",
            "descripcion": (
                "NaN genuino: dato no disponible en la API de Google Trends para ese "
                "termino/mes. Puede deberse a que el termino fue introducido en la "
                "busqueda fuera de su ventana temporal o a limitaciones de la API."
            ),
            "decision": "mantener_NaN — no imputable sin distorsion",
            "impacto_si_se_imputa": "Cualquier imputacion seria arbitraria sin referencia valida",
            "impacto_si_se_elimina": "Perdida de filas que podrian tener otros campos validos",
        },
        {
            "dataset": "instagram_posts_clean",
            "variable": "likes / comentarios",
            "tipo_valor_especial": "NaN_imputado_0",
            "descripcion": (
                "La API de Apify devolvio null para likes/comentarios en algunos posts. "
                "Puede ocurrir por posts con engagement bloqueado, posts muy recientes o "
                "limitaciones de la API en el momento del scraping. "
                "Se imputa a 0 (decision conservadora) con flag metricas_imputadas=True."
            ),
            "decision": "imputar_0_con_flag — el engagement calculado puede estar subestimado para posts afectados",
            "impacto_si_se_imputa": "Subestimacion del engagement real en posts afectados (conservador)",
            "impacto_si_se_elimina": "Perdida de posts validos en otras variables (caption, fecha, tipo_post)",
        },
        {
            "dataset": "instagram_posts_clean",
            "variable": "hashtags",
            "tipo_valor_especial": "NaN",
            "descripcion": (
                "Campo opcional en la API de Apify. Muchos posts no incluyen hashtags "
                "o la API no los extrajo en algunos casos. No se usa como variable "
                "cuantitativa en el analisis principal."
            ),
            "decision": "mantener_NaN — sin impacto analitico en variables cuantitativas",
            "impacto_si_se_imputa": "No aplicable (campo de texto no cuantitativo)",
            "impacto_si_se_elimina": "Perdida de posts validos en otras variables",
        },
        {
            "dataset": "capa2_term_coverage / capa2_term_analysis_priority",
            "variable": "quality_flag / prioridad_analitica",
            "tipo_valor_especial": "baja_cobertura_o_prioridad_exploratoria",
            "descripcion": (
                "Terminos con cobertura insuficiente y/o prioridad analitica exploratoria "
                "no se eliminan del dataset base. Se conservan por trazabilidad, pero pueden "
                "quedar fuera del subconjunto principal usado en el EDA principal."
            ),
            "decision": (
                "mantener_en_base_y_filtrar_en_subset_principal — "
                "la seleccion final depende de prioridad_analitica y no solo de quality_flag"
            ),
            "impacto_si_se_imputa": "Podria crear tendencias artificiales en terminos debiles",
            "impacto_si_se_elimina": "Se pierde trazabilidad sobre terminos de nicho o baja señal",
        },
    ]
 
    df = pd.DataFrame(records)
    out_path = TABLES_CAPA2_CONTROL / "capa2_null_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"  Null summary capa2: {len(df)} casos documentados → {out_path.name}")
    return df
 
 
# =========================
# VARIABLE SELECTION MATRIX — ENTRADAS ADICIONALES
# para valor_cero_trends y metricas_imputadas
# =========================
 
def build_capa2_vsm_flags() -> pd.DataFrame:
    """
    Añade a la variable selection matrix las entradas para los flags
    de calidad introducidos en el transform mejorado.
    """
    _ensure_dirs()

    # Documenta específicamente los flags de calidad que no son variables
    # analíticas en sí mismas sino indicadores del proceso de limpieza.
    # Se añaden a la VSM existente sin sobrescribir las entradas anteriores.
    records = [
        {
            "dataset": "trends_grupos_unificados_clean / trends_marcas_clean",
            "variable": "valor_cero_trends",
            "descripcion": "Flag booleano: True si valor_trends=0 (volumen insuficiente en GT, no interés cero)",
            "tipo_variable": "tecnica",
            "rol_analitico": "trazabilidad",
            "nivel_nulos": "ninguno",
            "redundancia": "baja",
            "decision": "mantener — permite filtrar ceros en analisis sensibles a la distribucion",
            "justificacion": (
                "El valor 0 en Google Trends no es equivalente a ausencia de interes. "
                "La API escala el maximo a 100 y trunca los valores bajos a 0. "
                "Este flag permite decidir si incluir o excluir ceros segun el objetivo del analisis."
            ),
        },
        {
            "dataset": "instagram_posts_clean / instagram_brand_monthly",
            "variable": "metricas_imputadas",
            "descripcion": "Flag booleano: True si likes o comentarios fueron imputados a 0 desde NaN",
            "tipo_variable": "tecnica",
            "rol_analitico": "trazabilidad",
            "nivel_nulos": "ninguno",
            "redundancia": "baja",
            "decision": "mantener — permite evaluar sesgo potencial en calculos de engagement",
            "justificacion": (
                "Los posts con metricas_imputadas=True pueden tener engagement subestimado. "
                "En el agregado mensual permite contextualizar la fiabilidad de likes, comentarios y engagement."
            ),
        },
    ]

    df = pd.DataFrame(records)

    vsm_path = TABLES_CAPA2_CONTROL / "capa2_variable_selection_matrix.csv"

    # Si la VSM ya existe, añade las nuevas entradas evitando duplicados.
    # Si no existe, crea una nueva VSM solo con los flags.
    if vsm_path.exists():
        vsm_existing = pd.read_csv(vsm_path)
        vsm_combined = pd.concat([vsm_existing, df], ignore_index=True)
        vsm_combined = vsm_combined.drop_duplicates(subset=["dataset", "variable"], keep="last")
        vsm_combined.to_csv(vsm_path, index=False)
        print(f"  VSM actualizada: {len(vsm_combined)} variables totales (añadidos flags de calidad)")
    else:
        df.to_csv(vsm_path, index=False)
        print(f"  VSM flags creada: {len(df)} entradas")

    return df


# =========================
# RUN ALL
# =========================

def run_all_builds() -> None:
    build_variable_selection_matrix_capa2()
    build_capa2_vsm_flags()
    build_capa2_null_summary()
    build_capa2_inventory()
    build_capa2_master_terminos()
    build_term_analysis_priority()
    build_term_quality_decisions()
    build_capa2_master_terminos_main()
    build_capa2_master_productos()
    build_capa2_master_eventos()
    build_capa2_master_eventos_mensual()
    build_capa2_master_integrated()
    build_capa2_master_social()
    build_capa2_master_brand_digital()
    build_variable_final_decisions_capa2()
    build_capa2_master_previews()
    build_capa2_sqlite()
    print("Todos los builds de capa 2 completados.")


if __name__ == "__main__":
    run_all_builds()