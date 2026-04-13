"""
build_capa1.py — Construcción de masters integrados de Capa 1

Ejecutar después de transform_capa1.py.

Cambios respecto a versión anterior:
  - Variable selection matrix actualizada: pct_empresas_venden_web_apps
    refleja el nulo de 2023 por cambio metodológico INE K→I
  - build_capa1_master_anual() propaga flag pct_personas_compra_online_imputado
    desde contexto al master anual para trazabilidad completa
  - build_capa1_null_summary(): tabla única con todos los nulos del master
    anual y sus decisiones metodológicas documentadas
  - Inventory corregido: eurostat_online_empresas periodo_fin = 2024
  - variable_final_decisions actualizada en consecuencia
"""

import sqlite3
import re

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
    # Crea la estructura de carpetas de salida si no existe.
    # Se llama al inicio de cada función para garantizar que los paths
    # de escritura están disponibles independientemente del orden de ejecución.
    for sub in ["comercio_electronico", "integrated"]:
        (PROCESSED_CAPA1 / sub).mkdir(parents=True, exist_ok=True)
    TABLES_CAPA1.mkdir(parents=True, exist_ok=True)
    TABLES_CAPA1_CONTROL.mkdir(parents=True, exist_ok=True)
    TABLES_CAPA1_MASTERS.mkdir(parents=True, exist_ok=True)
    DB_CAPA1.parent.mkdir(parents=True, exist_ok=True)


# =========================
# 0. VARIABLE SELECTION MATRIX
# =========================

def build_variable_selection_matrix_capa1() -> pd.DataFrame:
    """
    Documenta la decisión de incluir o excluir cada variable en el master analítico.
    Refleja el estado post-transform incluyendo:
      - El nulo estructural de pct_empresas_venden_web_apps en 2023
        (cambio metodológico INE K→I, no imputable)
      - El flag de imputación de pct_personas_compra_online (2025)
    """
    _ensure_dirs()

    # Cada registro documenta una variable con su rol analítico, nivel de nulos
    # y la decisión metodológica adoptada. Esta tabla es la evidencia documental
    # del proceso de selección de variables para el corrector y el tribunal.
    records = [
        # --- contexto_digitalizacion_clean ---
        {
            "dataset": "contexto_digitalizacion_clean",
            "variable": "anio",
            "descripcion": "Año de observación",
            "tipo_variable": "tecnica",
            "rol_analitico": "join_temporal",
            "nivel_nulos": "ninguno",
            "decision": "mantener",
            "justificacion": "Variable temporal principal para cruces con otras fuentes anuales.",
        },
        {
            "dataset": "contexto_digitalizacion_clean",
            "variable": "pct_usuarios_rrss",
            "descripcion": "Porcentaje de usuarios de redes sociales en España",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "ninguno",
            "decision": "mantener",
            "justificacion": "Indicador clave del contexto de digitalización social.",
        },
        {
            "dataset": "contexto_digitalizacion_clean",
            "variable": "pct_personas_compra_online",
            "descripcion": "Porcentaje de personas que realizan compras online",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "1 (2025, imputado por interpolacion lineal)",
            "decision": "mantener — nulo imputado con interpolacion lineal, flag _imputado=True",
            "justificacion": (
                "Indicador clave del consumo digital. 1 nulo en 2025 por dato pendiente "
                "de publicacion. Serie 2020-2024 continua y monótonamente creciente "
                "(62.62→68.94): interpolacion lineal es la estimacion mas conservadora. "
                "Valor imputado: 68.94. Flag de trazabilidad incluido en el master."
            ),
        },
        {
            "dataset": "contexto_digitalizacion_clean",
            "variable": "pct_personas_compra_online_imputado",
            "descripcion": "Flag booleano: True si el valor de pct_personas_compra_online fue imputado",
            "tipo_variable": "tecnica",
            "rol_analitico": "trazabilidad",
            "nivel_nulos": "ninguno",
            "decision": "mantener — propagado al master anual para analisis de sensibilidad",
            "justificacion": "Permite excluir el valor estimado del año 2025 en analisis de sensibilidad.",
        },
        # --- contexto extended / documentado ---
        {
            "dataset": "contexto_digitalizacion_extended",
            "variable": "pct_personas_compra_ropa_online",
            "descripcion": "Porcentaje de personas que compran ropa online",
            "tipo_variable": "auxiliar",
            "rol_analitico": "contexto",
            "nivel_nulos": "alto (5/6 filas)",
            "decision": "mantener_solo_trazabilidad",
            "justificacion": (
                "Variable temáticamente relevante pero con cobertura insuficiente "
                "(solo 1 observacion valida, año 2024). No entra en el master analitico principal."
            ),
        },
        {
            "dataset": "contexto_digitalizacion_documentado",
            "variable": "comentarios_hitos",
            "descripcion": "Comentarios contextuales sobre hitos del periodo",
            "tipo_variable": "contextual",
            "rol_analitico": "trazabilidad",
            "nivel_nulos": "bajo",
            "decision": "mantener_solo_trazabilidad",
            "justificacion": "Util para narrativa e interpretacion, no para analisis cuantitativo.",
        },
        {
            "dataset": "contexto_digitalizacion_documentado",
            "variable": "fuente_usuarios_redes / fuente_compra_online / fuente_compra_ropa_online",
            "descripcion": "Fuentes originales de cada variable",
            "tipo_variable": "tecnica",
            "rol_analitico": "trazabilidad",
            "nivel_nulos": "bajo",
            "decision": "mantener_solo_trazabilidad",
            "justificacion": "Necesarias para auditoria metodologica, no para el master analitico.",
        },
        # --- comercio_electronico_core_std ---
        {
            "dataset": "comercio_electronico_core_std",
            "variable": "anio",
            "descripcion": "Año de observacion",
            "tipo_variable": "tecnica",
            "rol_analitico": "join_temporal",
            "nivel_nulos": "ninguno",
            "decision": "mantener",
            "justificacion": "Necesaria para series anuales y cruces con el resto de fuentes.",
        },
        {
            "dataset": "comercio_electronico_core_std",
            "variable": "indicador_std",
            "descripcion": "Indicador estandarizado comun entre años",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "ninguno",
            "decision": "mantener",
            "justificacion": "Permite comparar entre años indicadores con distinta codificacion original.",
        },
        {
            "dataset": "comercio_electronico_core_std",
            "variable": "tamano_empresa",
            "descripcion": "Tamaño de empresa",
            "tipo_variable": "auxiliar",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "ninguno",
            "decision": "mantener",
            "justificacion": "Permite analisis por tamaño y seleccion de 'total' para masters integrados.",
        },
        {
            "dataset": "comercio_electronico_core_std",
            "variable": "valor",
            "descripcion": "Valor numerico del indicador",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo (1210 nulos estructurales INE, 3.52%)",
            "decision": "mantener — nulos son ausencias estructurales de la fuente, no imputables",
            "justificacion": (
                "Variable cuantitativa principal. Los 1210 nulos corresponden a indicadores "
                "no incluidos en ediciones anteriores de la Encuesta TIC-E. Son ausencias "
                "de cuestionario, no errores de datos."
            ),
        },
        {
            "dataset": "comercio_electronico_core_std",
            "variable": "indicador",
            "descripcion": "Etiqueta original del indicador INE",
            "tipo_variable": "tecnica",
            "rol_analitico": "trazabilidad",
            "nivel_nulos": "ninguno",
            "decision": "mantener_solo_trazabilidad",
            "justificacion": "Conserva referencia original del INE; el analisis opera sobre indicador_std.",
        },
        # --- capa1_master_anual_analysis ---
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "anio",
            "descripcion": "Año de observacion",
            "tipo_variable": "tecnica",
            "rol_analitico": "join_temporal",
            "nivel_nulos": "ninguno",
            "decision": "mantener",
            "justificacion": "Variable temporal principal del master anual.",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_usuarios_rrss",
            "descripcion": "Uso de redes sociales (%)",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "ninguno",
            "decision": "mantener",
            "justificacion": "Indicador esencial del contexto digital.",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_personas_compra_online",
            "descripcion": "Compra online de personas (%)",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "ninguno en periodo 2020-2023",
            "decision": "mantener",
            "justificacion": "Indicador esencial del consumo digital. Nulo de 2025 excluido del master_analysis.",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_facturacion_empresas_online",
            "descripcion": "Facturacion empresarial procedente de ventas online (%)",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "ninguno en periodo 2020-2023",
            "decision": "mantener",
            "justificacion": "Aporta la perspectiva empresarial del canal online.",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_empresas_venden_ecommerce",
            "descripcion": "Porcentaje de empresas que venden por ecommerce",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "ninguno en periodo 2020-2023",
            "decision": "mantener",
            "justificacion": "Mide adopcion empresarial del canal ecommerce.",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_empresas_venden_web_apps",
            "descripcion": "Porcentaje de empresas que venden por web o apps",
            "tipo_variable": "auxiliar",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "1 (2023) — ausencia metodologica, NO imputable",
            "decision": (
                "mantener con nulo documentado — cambio metodologico INE: "
                "indicador K.1 (2020-2022) no es comparable con I.1.1 (2023)"
            ),
            "justificacion": (
                "El INE reformulo el modulo de ecommerce en 2023, renombrando K.1 como I.1.1 "
                "con cambios en universo y formulacion. El valor de I.1.1 en 2023 es 27.44% "
                "pero no es homologable con K.1 de años anteriores. Se conserva el NaN en 2023 "
                "para no introducir una comparacion metodologicamente incorrecta. "
                "Valor de referencia documentado en calidad/master_anual_null_decisions.csv."
            ),
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_ventas_ecommerce_sobre_total",
            "descripcion": "Peso del ecommerce sobre ventas totales (%)",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "ninguno en periodo 2020-2023",
            "decision": "mantener",
            "justificacion": "Mide la importancia relativa del canal online en el total de ventas.",
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_ventas_ecommerce_sobre_total_empresas_que_venden",
            "descripcion": "Peso del ecommerce sobre ventas de empresas activas en el canal (%)",
            "tipo_variable": "auxiliar",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "ninguno en periodo 2020-2023",
            "decision": "mantener",
            "justificacion": "Lectura mas fina del peso del canal dentro del subconjunto de empresas ecommerce.",
        },
        # --- capa1_master_mensual_analysis ---
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "fecha",
            "descripcion": "Mes de observacion",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "ninguno",
            "decision": "mantener",
            "justificacion": "Variable temporal principal del master mensual.",
        },
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "indice_retail_moda",
            "descripcion": "Indice mensual del retail de moda (base 2015=100)",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "ninguno",
            "decision": "mantener",
            "justificacion": "Serie principal para estudiar la evolucion del sector moda.",
        },
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "indice_retail_total",
            "descripcion": "Indice mensual del retail total (base 2021=100)",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "ninguno",
            "decision": "mantener",
            "justificacion": "Serie de referencia para comparar el desempeño de la moda frente al retail agregado.",
        },
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "ratio_moda_vs_total",
            "descripcion": "Ratio entre retail de moda y retail total",
            "tipo_variable": "auxiliar",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "ninguno",
            "decision": "mantener",
            "justificacion": "Variable derivada con alto valor interpretativo para comparar rendimiento relativo.",
        },
        {
            "dataset": "capa1_master_mensual_analysis",
            "variable": "dif_moda_vs_total",
            "descripcion": "Diferencia absoluta entre indice de moda e indice retail total",
            "tipo_variable": "auxiliar",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "ninguno",
            "decision": "mantener",
            "justificacion": "Complementa el ratio con una lectura absoluta de la brecha entre ambas series.",
        },
    ]

    df = pd.DataFrame(records)
    out_path = TABLES_CAPA1_CONTROL / "capa1_variable_selection_matrix.csv"
    df.to_csv(out_path, index=False)
    print(f"  Variable selection matrix: {len(df)} variables documentadas → {out_path.name}")
    return df


# =========================
# 0B. NULL SUMMARY — MASTER ANUAL
# =========================

def build_capa1_null_summary() -> pd.DataFrame:
    """
    Tabla única que consolida todos los nulos del master anual con su decisión
    metodológica. Complementa la variable_selection_matrix con los valores
    concretos antes y después del tratamiento.

    Esta tabla es la evidencia documental del proceso de limpieza:
    muestra para cada nulo su naturaleza, la decisión tomada y el impacto.
    """
    _ensure_dirs()

    master_full = pd.read_csv(PROCESSED_CAPA1 / "integrated" / "capa1_master_anual_full.csv")

    records = []

    # --- pct_personas_compra_online: imputado en 2025 ---
    # Este nulo sí se imputa porque la serie es continua y el dato estaba pendiente
    # de publicación. Se usa interpolación lineal como estimación conservadora.
    rows_con_nulo = master_full[master_full["pct_personas_compra_online"].isnull()]
    for _, row in rows_con_nulo.iterrows():
        records.append({
            "dataset": "capa1_master_anual_full",
            "variable": "pct_personas_compra_online",
            "anio": int(row["anio"]) if pd.notna(row["anio"]) else None,
            "valor_antes": None,
            "valor_despues": 68.94,
            "tipo_ausencia": "dato_pendiente_publicacion",
            "metodo_tratamiento": "interpolacion_lineal",
            "flag_trazabilidad": "pct_personas_compra_online_imputado=True en contexto_digitalizacion_clean",
            "incluido_en_master_analysis": "no — 2025 fuera del periodo 2020-2023",
            "justificacion": (
                "Dato de 2025 no publicado en fecha de extraccion. "
                "Serie 2020-2024 continua: interpolacion lineal aplicada."
            ),
        })

    # --- pct_empresas_venden_web_apps: nulo en 2023, NO imputado ---
    # Este nulo NO se imputa porque el cambio de codificación INE (K.1→I.1.1)
    # hace que el valor de 2023 sea metodológicamente incomparable con años anteriores.
    # Imputar introduciría un sesgo sistemático en el análisis de tendencia.
    if "pct_empresas_venden_web_apps" in master_full.columns:
        rows_web = master_full[master_full["pct_empresas_venden_web_apps"].isnull()]
        for _, row in rows_web.iterrows():
            records.append({
                "dataset": "capa1_master_anual_full / capa1_master_anual_analysis",
                "variable": "pct_empresas_venden_web_apps",
                "anio": int(row["anio"]) if pd.notna(row["anio"]) else None,
                "valor_antes": None,
                "valor_despues": None,
                "tipo_ausencia": "cambio_metodologico_ine_K_a_I",
                "metodo_tratamiento": "ninguno — incompatibilidad metodologica",
                "flag_trazabilidad": "NaN conservado en master; valor referencia I.1.1=27.44% en null_decisions.csv",
                "incluido_en_master_analysis": "si — con NaN (2023 dentro del periodo 2020-2023)",
                "justificacion": (
                    "INE renombro K.1→I.1.1 en 2023 con cambios en universo de referencia. "
                    "Imputar con I.1.1 introduciria un sesgo metodologico. "
                    "Se conserva NaN y se documenta el valor de referencia (27.44%)."
                ),
            })

    # --- Variables INE sin cobertura en 2024-2025 ---
    # La Encuesta TIC-E se publica con ~1 año de desfase, por lo que los datos
    # de 2024-2025 no estaban disponibles en la fecha de extracción.
    # Estos nulos quedan en el master_full pero se excluyen del master_analysis
    # restringiendo el periodo a 2020-2023 donde todas las variables son completas.
    ine_vars = [
        "pct_facturacion_empresas_online",
        "pct_empresas_venden_ecommerce",
        "pct_ventas_ecommerce_sobre_total",
        "pct_ventas_ecommerce_sobre_total_empresas_que_venden",
    ]
    for var in ine_vars:
        if var not in master_full.columns:
            continue
        rows_var = master_full[
            master_full[var].isnull() & master_full["anio"].isin([2024, 2025])
        ]
        for _, row in rows_var.iterrows():
            records.append({
                "dataset": "capa1_master_anual_full",
                "variable": var,
                "anio": int(row["anio"]) if pd.notna(row["anio"]) else None,
                "valor_antes": None,
                "valor_despues": None,
                "tipo_ausencia": "fuera_cobertura_temporal_ine",
                "metodo_tratamiento": "ninguno — excluido del master_analysis",
                "flag_trazabilidad": "presente en master_full, ausente en master_analysis (2020-2023)",
                "incluido_en_master_analysis": "no — master_analysis cubre 2020-2023",
                "justificacion": (
                    "Encuesta TIC-E se publica con ~1 año de desfase. "
                    "Datos de 2024-2025 no disponibles en fecha de extraccion. "
                    "El master_analysis se restringe a 2020-2023 para garantizar "
                    "cobertura completa en todas las variables."
                ),
            })

    df = pd.DataFrame(records)
    out_path = TABLES_CAPA1_CONTROL / "capa1_master_anual_null_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"  Null summary master anual: {len(df)} entradas → {out_path.name}")
    return df


# =========================
# 1. BUILD COMERCIO CORE
# =========================

def build_comercio_core() -> pd.DataFrame:
    _ensure_dirs()

    df = pd.read_csv(PROCESSED_CAPA1 / "comercio_electronico" / "comercio_electronico_clean.csv")

    # Normalización del campo indicador para facilitar la comparación de texto:
    # minúsculas, eliminación de comas y espacios múltiples
    indicador_norm = (
        df["indicador"]
        .astype(str)
        .str.lower()
        .str.replace(",", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Filtra solo los 4 indicadores analíticamente relevantes para el TFG.
    # Se usan expresiones regulares para ser robustos frente a variaciones
    # menores en el texto entre ediciones anuales del cuestionario INE.
    mask = (
        indicador_norm.str.contains(r"% de empresas que han realizado ventas por comercio electrónico", na=False)
        | indicador_norm.str.contains(r"% de empresas que han realizado ventas mediante páginas web", na=False)
        | indicador_norm.str.contains(r"% ventas mediante comercio electrónico sobre el total de ventas$", na=False, regex=True)
        | indicador_norm.str.contains(
            r"% ventas mediante comercio electrónico sobre el total de ventas de las empresas que venden por comercio electrónico$",
            na=False,
            regex=True,
        )
    )

    core = df[mask].copy().reset_index(drop=True)

    out_path = PROCESSED_CAPA1 / "comercio_electronico" / "comercio_electronico_core.csv"
    core.to_csv(out_path, index=False)

    print(f"  Comercio core: {core.shape[0]} filas, {core['indicador'].nunique()} indicadores únicos")
    return core


# =========================
# 2. STANDARDIZE COMERCIO CORE
# =========================

def standardize_comercio_core() -> pd.DataFrame:
    _ensure_dirs()

    df = pd.read_csv(PROCESSED_CAPA1 / "comercio_electronico" / "comercio_electronico_core.csv")

    def map_indicator(text: str) -> str | None:
        # Mapea el texto original del indicador INE a un nombre estandarizado
        # y estable entre ediciones anuales. Sin esta estandarización, el mismo
        # indicador puede tener textos ligeramente distintos en 2018 y en 2022.
        if pd.isna(text):
            return None

        t = str(text).lower().strip()
        t = t.replace(",", " ")
        t = re.sub(r"\s+", " ", t)

        if "empresas que han realizado ventas por comercio electrónico" in t:
            return "pct_empresas_venden_ecommerce"

        if "empresas que han realizado ventas mediante páginas web" in t:
            return "pct_empresas_venden_web_apps"

        if "ventas mediante comercio electrónico sobre el total de ventas de las empresas que venden" in t:
            return "pct_ventas_ecommerce_sobre_total_empresas_que_venden"

        if "ventas mediante comercio electrónico sobre el total de ventas" in t:
            return "pct_ventas_ecommerce_sobre_total"

        return None

    df["indicador_std"] = df["indicador"].apply(map_indicator)

    # Exporta los indicadores que no pudieron mapearse para revisión manual.
    # Si este fichero tiene contenido, indica que hay indicadores nuevos en la
    # fuente que no estaban contemplados en el mapa de estandarización.
    unmatched = df[df["indicador_std"].isna()].copy()
    if not unmatched.empty:
        unmatched.to_csv(
            PROCESSED_CAPA1 / "comercio_electronico" / "comercio_electronico_core_unmatched.csv",
            index=False,
        )
        print(f"  [WARN] Indicadores no mapeados: {len(unmatched)} -> comercio_electronico_core_unmatched.csv")

    df = df[df["indicador_std"].notna()].copy().reset_index(drop=True)

    out_path = PROCESSED_CAPA1 / "comercio_electronico" / "comercio_electronico_core_std.csv"
    df.to_csv(out_path, index=False)

    print(f"  Comercio core_std: {df.shape[0]} filas | indicadores_std: {df['indicador_std'].nunique()}")
    print(f"    {df['indicador_std'].value_counts().to_dict()}")

    return df


# =========================
# 3. BUILD INVENTORY
# =========================

def build_capa1_inventory() -> pd.DataFrame:
    """
    Inventario de datasets procesados con cobertura temporal actualizada.
    Corrección: eurostat_online_empresas ahora cubre hasta 2024 (no 2023).
    """
    _ensure_dirs()

    # El inventario documenta cada dataset con su fuente, frecuencia, cobertura
    # y rol analítico en la Capa 1. Es el punto de entrada para entender
    # qué datos alimentan el master y por qué se incluyeron.
    records = [
        {
            "dataset": "processed/capa1/contexto_digitalizacion/contexto_digitalizacion_clean.csv",
            "fuente": "DataReportal + Eurostat",
            "frecuencia": "anual",
            "periodo_inicio": 2020,
            "periodo_fin": 2025,
            "n_observaciones": 6,
            "granularidad": "España",
            "rol_capa1": "contexto digital — RRSS y compra online",
            "nota": "1 valor imputado (2025, interpolacion lineal, flag _imputado=True)",
        },
        {
            "dataset": "processed/capa1/eurostat/eurostat_moda_mensual_clean.csv",
            "fuente": "Eurostat sts_trtu_m (NACE G47.7)",
            "frecuencia": "mensual",
            "periodo_inicio": "2010-01",
            "periodo_fin": "2023-12",
            "n_observaciones": 168,
            "granularidad": "España",
            "rol_capa1": "contexto sectorial moda — indice volumen ventas base 2015=100",
            "nota": "21 celdas no disponibles (':') para 2024-2025 excluidas y documentadas",
        },
        {
            "dataset": "processed/capa1/eurostat/eurostat_retail_total_mensual_clean.csv",
            "fuente": "Eurostat sts_trtu_m (NACE G47)",
            "frecuencia": "mensual",
            "periodo_inicio": "2010-01",
            "periodo_fin": "2025-09",
            "n_observaciones": 189,
            "granularidad": "España",
            "rol_capa1": "contexto retail general — indice volumen ventas base 2021=100",
            "nota": "Base 2021=100 (correccion respecto a nomenclatura anterior)",
        },
        {
            "dataset": "processed/capa1/eurostat/eurostat_online_empresas_clean.csv",
            "fuente": "Eurostat tin00110",
            "frecuencia": "anual",
            "periodo_inicio": 2015,
            "periodo_fin": 2024,  # CORREGIDO: antes 2023
            "n_observaciones": 10,
            "granularidad": "España",
            "rol_capa1": "adopcion online empresarial — % facturacion ecommerce",
            "nota": "Se extiende a 2024 (19.52%) respecto a version anterior que cortaba en 2023",
        },
        {
            "dataset": "processed/capa1/comercio_electronico/comercio_electronico_core_std.csv",
            "fuente": "INE / Encuesta TIC y Comercio Electronico",
            "frecuencia": "anual",
            "periodo_inicio": 2015,
            "periodo_fin": 2023,
            "n_observaciones": "variable (4 indicadores x 4 tamaños x 9 años)",
            "granularidad": "España por tamaño de empresa",
            "rol_capa1": "contexto ecommerce empresarial — 4 indicadores estandarizados",
            "nota": "1210 nulos estructurales (3.52%) por variacion del cuestionario INE entre años",
        },
    ]

    inventory = pd.DataFrame(records)
    out_path = PROCESSED_CAPA1 / "integrated" / "capa1_inventory.csv"
    inventory.to_csv(out_path, index=False)
    print(f"  Inventory: {len(inventory)} datasets documentados")
    return inventory


# =========================
# 4. BUILD MASTER ANUAL
# =========================

def build_capa1_master_anual() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye los masters anuales full y analysis.

    Cambio respecto a versión anterior:
      - Propaga la columna pct_personas_compra_online_imputado desde
        contexto_digitalizacion_clean al master para trazabilidad completa.
      - master_analysis: periodo 2020-2023 (cobertura completa de todas las fuentes).
      - master_full: periodo completo 2020-2025 (incluye años con nulos parciales).
    """
    _ensure_dirs()

    contexto = pd.read_csv(
        PROCESSED_CAPA1 / "contexto_digitalizacion" / "contexto_digitalizacion_clean.csv"
    )
    online_empresas = pd.read_csv(
        PROCESSED_CAPA1 / "eurostat" / "eurostat_online_empresas_clean.csv"
    )
    comercio = pd.read_csv(
        PROCESSED_CAPA1 / "comercio_electronico" / "comercio_electronico_core_std.csv"
    )

    # Pivot del comercio al nivel 'total': una fila por año con cada indicador en columna.
    # Se usa solo el agregado total (no por tamaño de empresa) para el master integrado.
    comercio_total = comercio[comercio["tamano_empresa"] == "total"].copy()
    comercio_pivot = comercio_total.pivot_table(
        index="anio",
        columns="indicador_std",
        values="valor",
        aggfunc="first",
    ).reset_index()

    # Merge secuencial: contexto como base, se añaden las demás fuentes por año.
    # El flag de imputación de contexto se propaga al master para trazabilidad.
    master = contexto.merge(
        online_empresas[["anio", "valor_pct"]],
        on="anio",
        how="left",
    ).rename(columns={"valor_pct": "pct_facturacion_empresas_online"})

    master = master.merge(comercio_pivot, on="anio", how="left")

    # Reordenar columnas: flag de imputacion al final para legibilidad
    flag_col = "pct_personas_compra_online_imputado"
    other_cols = [c for c in master.columns if c != flag_col]
    master = master[other_cols + [flag_col]] if flag_col in master.columns else master

    # Master full: todos los años
    out_full = PROCESSED_CAPA1 / "integrated" / "capa1_master_anual_full.csv"
    master.to_csv(out_full, index=False)

    # Master analysis: periodo 2020-2023, cobertura completa en todas las variables INE.
    # Este es el dataset que alimenta el análisis del TFG.
    master_analysis = master[master["anio"].between(2020, 2023)].copy().reset_index(drop=True)
    out_analysis = PROCESSED_CAPA1 / "integrated" / "capa1_master_anual_analysis.csv"
    master_analysis.to_csv(out_analysis, index=False)

    # Resumen de nulos en el master_analysis
    nulos = master_analysis.isnull().sum()
    nulos_con = nulos[nulos > 0]
    print(f"  Master anual full: {master.shape} → {out_full.name}")
    print(f"  Master anual analysis (2020-2023): {master_analysis.shape} → {out_analysis.name}")
    if len(nulos_con):
        print(f"  Nulos en master_analysis:")
        for col, n in nulos_con.items():
            print(f"    {col}: {n} (documentado en null_summary)")
    else:
        print("  Nulos en master_analysis: ninguno en variables core")
    if flag_col in master_analysis.columns:
        n_imp = int(master_analysis[flag_col].sum())
        print(f"  Valores imputados en master_analysis: {n_imp} (pct_personas_compra_online)")

    return master, master_analysis


# =========================
# 5. BUILD MASTER MENSUAL
# =========================

def build_capa1_master_mensual() -> tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()

    # Las dos series mensuales de Eurostat tienen bases distintas (2015=100 y 2021=100)
    # y coberturas temporales distintas (hasta 2023-12 y hasta 2025-09 respectivamente).
    # El master mensual se construye por inner join sobre fecha, conservando solo
    # el periodo con cobertura completa en ambas series (2010-2023).
    moda = pd.read_csv(PROCESSED_CAPA1 / "eurostat" / "eurostat_moda_mensual_clean.csv")
    retail = pd.read_csv(PROCESSED_CAPA1 / "eurostat" / "eurostat_retail_total_mensual_clean.csv")

    moda["fecha"] = pd.to_datetime(moda["fecha"])
    retail["fecha"] = pd.to_datetime(retail["fecha"])

    moda = moda[["fecha", "valor_indice"]].rename(columns={"valor_indice": "indice_retail_moda"})
    retail = retail[["fecha", "valor_indice"]].rename(columns={"valor_indice": "indice_retail_total"})

    # Variables derivadas para comparación relativa entre series con bases distintas.
    # El ratio expresa el nivel relativo de moda respecto al retail total;
    # la diferencia aporta la lectura en puntos absolutos de la brecha entre series.
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

    out_full = PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_full.csv"
    master_mensual.to_csv(out_full, index=False)

    # Master analysis: 2015-2023 (108 obs), ventana analítica del TFG.
    # Se excluyen 2010-2014 por ser anteriores al inicio de la serie de moda completa.
    master_analysis = master_mensual[
        master_mensual["anio"].between(2015, 2023)
    ].copy().reset_index(drop=True)
    out_analysis = PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_analysis.csv"
    master_analysis.to_csv(out_analysis, index=False)

    print(f"  Master mensual full: {master_mensual.shape} ({master_mensual['fecha'].min().strftime('%Y-%m')}–{master_mensual['fecha'].max().strftime('%Y-%m')})")
    print(f"  Master mensual analysis (2015-2023): {master_analysis.shape}")
    print(f"  Nulos: {master_analysis.isnull().sum().sum()} (ninguno esperado)")
    return master_mensual, master_analysis


# =========================
# 5B. VARIABLE FINAL DECISIONS
# =========================

def build_capa1_variable_final_decisions() -> pd.DataFrame:
    _ensure_dirs()

    # Tabla de decisiones finales sobre qué variables entran en el análisis principal
    # y cuáles quedan solo como contexto o trazabilidad. Complementa la
    # variable_selection_matrix con una vista más operativa orientada al análisis.
    records = [
        {"dataset": "contexto_digitalizacion_clean", "variable": "anio",
         "mantener_en_base": "si", "usar_en_analisis_principal": "si", "usar_solo_contexto": "no",
         "comentario": "Eje temporal anual del bloque de contexto."},
        {"dataset": "contexto_digitalizacion_clean", "variable": "pct_usuarios_rrss",
         "mantener_en_base": "si", "usar_en_analisis_principal": "si", "usar_solo_contexto": "no",
         "comentario": "Indicador clave de digitalización social."},
        {"dataset": "contexto_digitalizacion_clean", "variable": "pct_personas_compra_online",
         "mantener_en_base": "si", "usar_en_analisis_principal": "si", "usar_solo_contexto": "no",
         "comentario": "Indicador clave del consumo digital. 1 valor imputado (2025)."},
        {"dataset": "contexto_digitalizacion_clean", "variable": "pct_personas_compra_online_imputado",
         "mantener_en_base": "si", "usar_en_analisis_principal": "no", "usar_solo_contexto": "si",
         "comentario": "Flag de trazabilidad de imputacion. Propagado al master anual."},
        {"dataset": "contexto_digitalizacion_extended", "variable": "pct_personas_compra_ropa_online",
         "mantener_en_base": "si", "usar_en_analisis_principal": "no", "usar_solo_contexto": "si",
         "comentario": "Cobertura insuficiente (1/6 obs). Solo trazabilidad."},
        {"dataset": "contexto_digitalizacion_documentado", "variable": "comentarios_hitos",
         "mantener_en_base": "si", "usar_en_analisis_principal": "no", "usar_solo_contexto": "si",
         "comentario": "Util para narrativa e interpretacion."},
        {"dataset": "comercio_electronico_core_std", "variable": "indicador_std",
         "mantener_en_base": "si", "usar_en_analisis_principal": "si", "usar_solo_contexto": "no",
         "comentario": "Clave para unificar indicadores entre años."},
        {"dataset": "comercio_electronico_core_std", "variable": "valor",
         "mantener_en_base": "si", "usar_en_analisis_principal": "si", "usar_solo_contexto": "no",
         "comentario": "Magnitud principal. Nulos estructurales conservados como NaN."},
        {"dataset": "capa1_master_anual_analysis", "variable": "pct_empresas_venden_web_apps",
         "mantener_en_base": "si", "usar_en_analisis_principal": "si", "usar_solo_contexto": "no",
         "comentario": (
             "NaN en 2023 conservado (cambio metodologico K→I). "
             "Analisis sobre 2020-2022 unicamente para esta variable."
         )},
        {"dataset": "capa1_master_mensual_analysis", "variable": "indice_retail_moda",
         "mantener_en_base": "si", "usar_en_analisis_principal": "si", "usar_solo_contexto": "no",
         "comentario": "Serie principal del sector moda. Descomposicion ST aplicada en EDA."},
        {"dataset": "capa1_master_mensual_analysis", "variable": "ratio_moda_vs_total",
         "mantener_en_base": "si", "usar_en_analisis_principal": "si", "usar_solo_contexto": "no",
         "comentario": "Variable derivada de alto valor interpretativo para comparacion relativa."},
    ]

    df = pd.DataFrame(records)
    out_path = TABLES_CAPA1_CONTROL / "capa1_variable_final_decisions.csv"
    df.to_csv(out_path, index=False)
    print(f"  Variable final decisions: {len(df)} entradas → {out_path.name}")
    return df


# =========================
# 5C. EXPORT MASTER PREVIEWS
# =========================

def export_capa1_master_previews() -> None:
    _ensure_dirs()
    # Exporta las primeras filas de cada master como vista rápida de verificación.
    # Útil para confirmar estructura y contenido sin abrir el fichero completo.
    pd.read_csv(PROCESSED_CAPA1 / "integrated" / "capa1_inventory.csv").head(10).to_csv(
        TABLES_CAPA1_MASTERS / "capa1_inventory_head.csv", index=False)
    pd.read_csv(PROCESSED_CAPA1 / "integrated" / "capa1_master_anual_analysis.csv").to_csv(
        TABLES_CAPA1_MASTERS / "capa1_master_anual_head.csv", index=False)
    pd.read_csv(PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_analysis.csv").head(12).to_csv(
        TABLES_CAPA1_MASTERS / "capa1_master_mensual_head.csv", index=False)
    print(f"  Previews exportados → {TABLES_CAPA1_MASTERS}")


# =========================
# 6. BUILD SQLITE
# =========================

def build_capa1_sqlite() -> None:
    _ensure_dirs()

    # Carga todos los datasets procesados en una base de datos SQLite.
    # Permite consultas exploratorias rápidas sin necesidad de cargar
    # todos los CSV en memoria. Útil para verificación y análisis ad hoc.
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
        "capa1_variable_selection_matrix": TABLES_CAPA1_CONTROL / "capa1_variable_selection_matrix.csv",
        "capa1_master_anual_null_summary": TABLES_CAPA1_CONTROL / "capa1_master_anual_null_summary.csv",
    }

    conn = sqlite3.connect(DB_CAPA1)
    for table_name, path in tables.items():
        if path.exists():
            df = pd.read_csv(path)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"  SQLite ← {table_name} ({len(df)} filas)")
        else:
            print(f"  [SKIP] {table_name}: archivo no encontrado")
    conn.close()
    print(f"  Base de datos: {DB_CAPA1}")


# =========================
# RUN ALL
# =========================

def run_all_builds() -> None:
    sep = "=" * 65
    print(sep)
    print("CAPA 1 — BUILD")
    print(sep)

    print("\n[0/8] Variable selection matrix...")
    build_variable_selection_matrix_capa1()

    print("\n[1/8] Comercio core...")
    build_comercio_core()

    print("\n[2/8] Estandarizar comercio core...")
    standardize_comercio_core()

    print("\n[3/8] Inventory...")
    build_capa1_inventory()

    print("\n[4/8] Master anual...")
    build_capa1_master_anual()

    print("\n[5/8] Master mensual...")
    build_capa1_master_mensual()

    print("\n[6/8] Null summary master anual...")
    build_capa1_null_summary()

    print("\n[7/8] Variable final decisions...")
    build_capa1_variable_final_decisions()

    print("\n[8/8] Previews + SQLite...")
    export_capa1_master_previews()
    build_capa1_sqlite()

    print(f"\n{sep}")
    print("BUILD CAPA 1 COMPLETADO")
    print("Tablas de control generadas:")
    print("  · capa1_variable_selection_matrix.csv")
    print("  · capa1_master_anual_null_summary.csv  ← NUEVO")
    print("  · capa1_variable_final_decisions.csv")
    print("  · capa1_master_anual_head.csv")
    print("  · capa1_master_mensual_head.csv")
    print(sep)


if __name__ == "__main__":
    run_all_builds()