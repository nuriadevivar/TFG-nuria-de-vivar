import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from src.common.config import (
    FIGURES_CAPA1,
    FIGURES_CAPA1_CONTROL,
    FIGURES_CAPA1_EDA,
    PROCESSED_CAPA1,
    TABLES_CAPA1,
    TABLES_CAPA1_CONTROL,
    TABLES_CAPA1_EDA,
)


# =========================
# HELPERS
# =========================

def _ensure_dirs() -> None:
    # Crea la estructura de carpetas de salida para tablas y figuras del EDA.
    # Se separan en subcarpetas /control (calidad y trazabilidad) y /eda
    # (resultados analíticos) para facilitar la navegación del repositorio.
    TABLES_CAPA1.mkdir(parents=True, exist_ok=True)
    TABLES_CAPA1_CONTROL.mkdir(parents=True, exist_ok=True)
    TABLES_CAPA1_EDA.mkdir(parents=True, exist_ok=True)

    FIGURES_CAPA1.mkdir(parents=True, exist_ok=True)
    FIGURES_CAPA1_CONTROL.mkdir(parents=True, exist_ok=True)
    FIGURES_CAPA1_EDA.mkdir(parents=True, exist_ok=True)


def _save_plot(filename: str, folder) -> None:
    # Guarda la figura activa en la carpeta indicada con resolución 300 dpi,
    # adecuada para incluir en documentos académicos sin pérdida de calidad.
    # tight_layout() ajusta márgenes automáticamente para evitar recortes.
    plt.tight_layout()
    plt.savefig(folder / filename, dpi=300, bbox_inches="tight")
    plt.close()


def _numeric_descriptives(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # Calcula estadísticos descriptivos personalizados para una lista de columnas.
    # A diferencia de df.describe(), incluye solo las métricas relevantes para
    # el análisis del TFG y garantiza tipos numéricos correctos en la salida.
    records = []

    for col in cols:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()

        records.append(
            {
                "variable": col,
                "count": int(s.shape[0]),
                "mean": round(float(s.mean()), 4) if not s.empty else np.nan,
                "median": round(float(s.median()), 4) if not s.empty else np.nan,
                "std": round(float(s.std()), 4) if not s.empty else np.nan,
                "min": round(float(s.min()), 4) if not s.empty else np.nan,
                "p25": round(float(s.quantile(0.25)), 4) if not s.empty else np.nan,
                "p75": round(float(s.quantile(0.75)), 4) if not s.empty else np.nan,
                "max": round(float(s.max()), 4) if not s.empty else np.nan,
            }
        )

    return pd.DataFrame(records)


def _get_period_bounds(df: pd.DataFrame) -> tuple[str | int | None, str | int | None]:
    # Detecta automáticamente el periodo cubierto por un dataset buscando
    # columna 'fecha' (series mensuales) o 'anio' (series anuales).
    # Devuelve None si no se encuentra ninguna columna temporal reconocida.
    start_period = None
    end_period = None

    if "fecha" in df.columns:
        fechas = pd.to_datetime(df["fecha"], errors="coerce")
        if fechas.notna().any():
            start_period = str(fechas.min().date())
            end_period = str(fechas.max().date())
    elif "anio" in df.columns:
        anios = pd.to_numeric(df["anio"], errors="coerce")
        if anios.notna().any():
            start_period = int(anios.min())
            end_period = int(anios.max())

    return start_period, end_period


# =========================
# 1. PROFILE GENERAL
# =========================

def profile_capa1() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()

    # Diccionario de todos los datasets de la Capa 1 que se van a perfilar.
    # El perfil incluye dimensiones, cobertura temporal, nulos y estadísticos
    # numéricos para cada dataset, generando tres CSVs de trazabilidad.
    datasets = {
        "contexto_digitalizacion_clean": PROCESSED_CAPA1 / "contexto_digitalizacion" / "contexto_digitalizacion_clean.csv",
        "contexto_digitalizacion_extended": PROCESSED_CAPA1 / "contexto_digitalizacion" / "contexto_digitalizacion_extended.csv",
        "contexto_digitalizacion_documentado": PROCESSED_CAPA1 / "contexto_digitalizacion" / "contexto_digitalizacion_documentado.csv",
        "eurostat_moda_mensual_clean": PROCESSED_CAPA1 / "eurostat" / "eurostat_moda_mensual_clean.csv",
        "eurostat_retail_total_mensual_clean": PROCESSED_CAPA1 / "eurostat" / "eurostat_retail_total_mensual_clean.csv",
        "eurostat_online_empresas_clean": PROCESSED_CAPA1 / "eurostat" / "eurostat_online_empresas_clean.csv",
        "comercio_electronico_core_std": PROCESSED_CAPA1 / "comercio_electronico" / "comercio_electronico_core_std.csv",
        "capa1_master_anual_analysis": PROCESSED_CAPA1 / "integrated" / "capa1_master_anual_analysis.csv",
        "capa1_master_mensual_analysis": PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_analysis.csv",
    }

    summary_records = []
    nulls_records = []
    numeric_records = []

    for name, path in datasets.items():
        df = pd.read_csv(path)
        start_period, end_period = _get_period_bounds(df)

        # Resumen de dimensiones y cobertura temporal por dataset
        summary_records.append(
            {
                "dataset": name,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "start_period": start_period,
                "end_period": end_period,
                "columns": ", ".join(df.columns.tolist()),
            }
        )

        # Nulos por variable: base para el análisis de decisiones de limpieza
        for col in df.columns:
            nulls_records.append(
                {
                    "dataset": name,
                    "variable": col,
                    "n_nulls": int(df[col].isna().sum()),
                    "pct_nulls": round(float(df[col].isna().mean() * 100), 2),
                }
            )

        # Estadísticos numéricos estándar para todas las columnas numéricas
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            desc = numeric_df.describe().T.reset_index().rename(columns={"index": "variable"})
            desc.insert(0, "dataset", name)
            numeric_records.append(desc)

    summary_df = pd.DataFrame(summary_records)
    nulls_df = pd.DataFrame(nulls_records)
    numeric_df = pd.concat(numeric_records, ignore_index=True) if numeric_records else pd.DataFrame()

    summary_df.to_csv(TABLES_CAPA1_CONTROL / "capa1_profile_summary.csv", index=False)
    nulls_df.to_csv(TABLES_CAPA1_CONTROL / "capa1_profile_nulls.csv", index=False)
    numeric_df.to_csv(TABLES_CAPA1_CONTROL / "capa1_profile_numeric.csv", index=False)

    print("Profiling completado.")
    print("Archivos generados:")
    print(TABLES_CAPA1_CONTROL / "capa1_profile_summary.csv")
    print(TABLES_CAPA1_CONTROL / "capa1_profile_nulls.csv")
    print(TABLES_CAPA1_CONTROL / "capa1_profile_numeric.csv")
    print("")
    print(summary_df)

    return summary_df, nulls_df, numeric_df


# =========================
# 2. ANALISIS DE NULOS
# =========================

def analyze_nulls_capa1() -> tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()

    input_path = TABLES_CAPA1_CONTROL / "capa1_profile_nulls.csv"
    df = pd.read_csv(input_path)

    # Filtra solo las variables con al menos un nulo y ordena por dataset y % de nulos
    detailed = df[df["pct_nulls"] > 0].copy().sort_values(
        by=["dataset", "pct_nulls"], ascending=[True, False]
    )
    detailed.to_csv(TABLES_CAPA1_CONTROL / "capa1_nulls_detailed.csv", index=False)

    def assign_null_decision(row: pd.Series) -> str:
        # Asigna una decisión analítica a cada variable con nulos según
        # su naturaleza y porcentaje de ausencia. Las variables de trazabilidad
        # y las de cobertura insuficiente reciben tratamiento específico
        # independientemente de su porcentaje de nulos.
        variable = str(row["variable"])
        pct = float(row["pct_nulls"])

        if variable in {
            "comentarios_hitos",
            "fuente_usuarios_redes",
            "fuente_compra_online",
            "fuente_compra_ropa_online",
        }:
            return "mantener_solo_trazabilidad"

        if variable == "pct_personas_compra_ropa_online":
            return "mantener_solo_contexto"

        if pct >= 80:
            return "descartar_analiticamente"

        if pct >= 20:
            return "mantener_con_cautela"

        return "mantener"

    def assign_null_comment(row: pd.Series) -> str:
        # Genera un comentario metodológico para cada variable con nulos,
        # explicando el motivo de la ausencia y la decisión adoptada.
        variable = str(row["variable"])
        pct = float(row["pct_nulls"])

        if variable == "pct_personas_compra_ropa_online":
            return "Variable temática útil, pero con cobertura insuficiente para incorporarla al master analítico principal."

        if variable == "comentarios_hitos":
            return "Campo cualitativo útil para interpretación narrativa, no para análisis cuantitativo."

        if variable.startswith("fuente_"):
            return "Campo de trazabilidad documental; no forma parte del análisis numérico."

        if pct >= 80:
            return "Nivel de nulos demasiado alto para uso analítico robusto."

        if pct >= 20:
            return "Variable utilizable solo con cautela y sin sobreponderarla en conclusiones."

        return "Nivel de nulos asumible."

    decisions = detailed.copy()
    decisions["decision_analitica"] = decisions.apply(assign_null_decision, axis=1)
    decisions["comentario"] = decisions.apply(assign_null_comment, axis=1)
    decisions.to_csv(TABLES_CAPA1_CONTROL / "capa1_nulls_decision_matrix.csv", index=False)

    # Genera un gráfico de barras de nulos por variable para los datasets
    # con nulos relevantes: contexto extended, documentado y master anual
    for dataset_name in [
        "contexto_digitalizacion_extended",
        "contexto_digitalizacion_documentado",
        "capa1_master_anual_analysis",
    ]:
        subset = detailed[detailed["dataset"] == dataset_name].copy()
        if subset.empty:
            continue

        plt.figure(figsize=(8, 4))
        plt.bar(subset["variable"], subset["pct_nulls"])
        plt.title(f"Nulos por variable - {dataset_name}")
        plt.ylabel("% nulos")
        plt.xticks(rotation=45, ha="right")
        _save_plot(f"{dataset_name}_nulls.png", FIGURES_CAPA1_CONTROL)

    print("Análisis de nulos completado.")
    print("Tablas guardadas en:")
    print(TABLES_CAPA1_CONTROL / "capa1_nulls_detailed.csv")
    print(TABLES_CAPA1_CONTROL / "capa1_nulls_decision_matrix.csv")

    return detailed, decisions


# =========================
# 3. JUSTIFICACION TIPO ANALISIS DESCRIPTIVO
# =========================

def eda_capa1_justificacion_analitica() -> pd.DataFrame:
    """
    Exporta la justificación del tipo de análisis descriptivo por dataset.
    Refuerza la trazabilidad metodológica del EDA y ayuda a documentar la rúbrica.
    """
    _ensure_dirs()

     # Cada registro justifica por qué se eligió un tipo de análisis concreto
    # para cada dataset y qué alternativas se descartaron y por qué.
    # Esta tabla responde directamente al criterio de marco teórico de la rúbrica.
    records = [
        {
            "dataset": "capa1_master_mensual_analysis",
            "tipo_dato": "serie_temporal_regular",
            "frecuencia": "mensual",
            "periodo": "2015-2023 (108 obs)",
            "tipo_analisis_elegido": "analisis_series_temporales",
            "justificacion": (
                "Serie mensual regular con suficiente longitud temporal para analizar "
                "tendencia, estacionalidad, evolución interanual y cambios estructurales. "
                "Se aplica descomposición temporal aditiva con periodo 12 por tratarse de "
                "una frecuencia mensual y ser adecuada para separar tendencia, componente "
                "estacional y residuo en una primera aproximación descriptiva. "
                "Se complementa con detección de outliers por IQR para identificar "
                "meses atípicos asociados a shocks estructurales."
            ),
            "alternativa_descartada": (
                "Análisis puramente transversal descartado porque ignoraría la dimensión "
                "temporal, que es la fuente principal de información del dataset."
            ),
        },
        {
            "dataset": "capa1_master_anual_analysis",
            "tipo_dato": "panel_longitudinal_corto",
            "frecuencia": "anual",
            "periodo": "2020-2023 (4 obs)",
            "tipo_analisis_elegido": "analisis_descriptivo_comparativo",
            "justificacion": (
                "Con solo cuatro observaciones anuales, el enfoque adecuado es un análisis "
                "descriptivo comparativo basado en evolución temporal, rangos, medias y "
                "comparación entre indicadores. El valor analítico se encuentra en la lectura "
                "conjunta de las variables de digitalización, compra online y ecommerce "
                "empresarial, más que en contrastes estadísticos formales."
            ),
            "alternativa_descartada": (
                "No se aplica descomposición temporal ni análisis estacional porque la serie "
                "es anual y demasiado corta para ello."
            ),
        },
        {
            "dataset": "eurostat_moda_mensual_clean / eurostat_retail_total_mensual_clean",
            "tipo_dato": "series_temporales_con_bases_distintas",
            "frecuencia": "mensual",
            "periodo": "2010-2023 / 2010-2025",
            "tipo_analisis_elegido": "comparacion_relativa_con_variables_derivadas",
            "justificacion": (
                "Los índices proceden de series con bases distintas, por lo que no es "
                "metodológicamente correcto comparar directamente sus niveles absolutos. "
                "Se construyen variables derivadas como ratio y diferencia para realizar una "
                "comparación relativa consistente entre la evolución del retail de moda y la "
                "del retail total."
            ),
            "alternativa_descartada": (
                "Comparación directa de valores absolutos descartada por no ser homogénea "
                "entre series con distinta base de referencia."
            ),
        },
        {
            "dataset": "comercio_electronico_core_std",
            "tipo_dato": "panel_anual_por_tamano_empresa",
            "frecuencia": "anual",
            "periodo": "2015-2023 (9 años x segmentos)",
            "tipo_analisis_elegido": "analisis_panel_con_segmentacion",
            "justificacion": (
                "El dataset combina dimensión temporal, indicador y segmentación por tamaño "
                "de empresa, lo que permite estudiar heterogeneidad empresarial en la adopción "
                "del canal ecommerce. Para el master anual se utiliza el agregado total con el "
                "fin de mantener comparabilidad con el resto de fuentes integradas."
            ),
            "alternativa_descartada": (
                "Agregación completa sin conservar la segmentación descartada como enfoque "
                "único porque haría perder información sobre brechas empresariales."
            ),
        },
        {
            "dataset": "eurostat_online_empresas_clean",
            "tipo_dato": "serie_temporal_anual",
            "frecuencia": "anual",
            "periodo": "2015-2024 (10 obs)",
            "tipo_analisis_elegido": "analisis_tendencia_anual",
            "justificacion": (
                "Serie anual adecuada para describir la evolución de la adopción del ecommerce "
                "empresarial a lo largo del tiempo mediante tasas de cambio, tendencia visual "
                "y comparación interanual. Al no tratarse de una serie mensual, no procede "
                "analizar estacionalidad."
            ),
            "alternativa_descartada": (
                "Descomposición estacional descartada por no ser aplicable a una serie anual."
            ),
        },
    ]

    df = pd.DataFrame(records)
    out_path = TABLES_CAPA1_CONTROL / "capa1_justificacion_tipo_analisis.csv"
    df.to_csv(out_path, index=False)

    print(f"Justificación tipo análisis: {len(df)} datasets → {out_path.name}")
    return df


# =========================
# 4. EDA ANUAL
# =========================

def eda_capa1_anual() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA1 / "integrated" / "capa1_master_anual_analysis.csv"
    df = pd.read_csv(input_path)

    # Estadísticos descriptivos completos con df.describe() para trazabilidad
    descriptivos = df.describe(include="all")
    descriptivos.to_csv(TABLES_CAPA1_EDA / "capa1_anual_descriptivos.csv")

    # Estadísticos personalizados para las variables analíticas principales
    annual_numeric_cols = [
        "pct_usuarios_rrss",
        "pct_personas_compra_online",
        "pct_facturacion_empresas_online",
        "pct_empresas_venden_ecommerce",
        "pct_empresas_venden_web_apps",
        "pct_ventas_ecommerce_sobre_total",
        "pct_ventas_ecommerce_sobre_total_empresas_que_venden",
    ]
    annual_numeric_desc = _numeric_descriptives(df, annual_numeric_cols)
    annual_numeric_desc.to_csv(TABLES_CAPA1_EDA / "capa1_anual_numeric_descriptives.csv", index=False)

    # Evolución del uso de RRSS y la compra online: dos indicadores clave del
    # contexto de digitalización del consumidor en España (2020-2023)
    plt.figure(figsize=(8, 5))
    plt.plot(df["anio"], df["pct_usuarios_rrss"], marker="o", label="Usuarios RRSS (%)")
    plt.plot(df["anio"], df["pct_personas_compra_online"], marker="o", label="Personas que compran online (%)")
    plt.title("Evolución del uso de RRSS y la compra online en España")
    plt.xlabel("Año")
    plt.ylabel("Porcentaje")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("capa1_anual_rrss_vs_compra_online.png", FIGURES_CAPA1_EDA)

    # Adopción empresarial del ecommerce: empresas que venden por canal digital
    plt.figure(figsize=(8, 5))
    plt.plot(df["anio"], df["pct_empresas_venden_ecommerce"], marker="o", label="Empresas que venden ecommerce (%)")
    plt.plot(df["anio"], df["pct_empresas_venden_web_apps"], marker="o", label="Empresas que venden por web/apps (%)")
    plt.title("Adopción empresarial del ecommerce")
    plt.xlabel("Año")
    plt.ylabel("Porcentaje")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("capa1_anual_empresas_venden_online.png", FIGURES_CAPA1_EDA)

    # Facturación online como % de ventas totales empresariales
    plt.figure(figsize=(8, 5))
    plt.plot(df["anio"], df["pct_facturacion_empresas_online"], marker="o")
    plt.title("Porcentaje de facturación empresarial procedente de ventas online")
    plt.xlabel("Año")
    plt.ylabel("Porcentaje")
    plt.grid(True, alpha=0.3)
    _save_plot("capa1_anual_facturacion_online_empresas.png", FIGURES_CAPA1_EDA)

    # Peso del ecommerce en las ventas: indicador de madurez del canal digital
    plt.figure(figsize=(8, 5))
    plt.plot(df["anio"], df["pct_ventas_ecommerce_sobre_total"], marker="o", label="Ecommerce sobre ventas totales (%)")
    plt.plot(
        df["anio"],
        df["pct_ventas_ecommerce_sobre_total_empresas_que_venden"],
        marker="o",
        label="Ecommerce sobre ventas totales (empresas que venden) (%)",
    )
    plt.title("Peso del ecommerce en las ventas empresariales")
    plt.xlabel("Año")
    plt.ylabel("Porcentaje")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("capa1_anual_peso_ventas_ecommerce.png", FIGURES_CAPA1_EDA)

    print("EDA anual completado.")
    print("Descriptivos guardados en:", TABLES_CAPA1_EDA / "capa1_anual_descriptivos.csv")
    print("Descriptivos numéricos guardados en:", TABLES_CAPA1_EDA / "capa1_anual_numeric_descriptives.csv")
    print("Gráficos guardados en:", FIGURES_CAPA1_EDA)

    return df


# =========================
# 5. EDA MENSUAL
# =========================

def eda_capa1_mensual() -> tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()

    input_path = PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_analysis.csv"
    df = pd.read_csv(input_path)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    # Estadísticos descriptivos completos y personalizados para las series mensuales
    descriptivos = df.describe(include="all")
    descriptivos.to_csv(TABLES_CAPA1_EDA / "capa1_mensual_descriptivos.csv")

    monthly_numeric_cols = [
        "indice_retail_moda",
        "indice_retail_total",
        "ratio_moda_vs_total",
        "dif_moda_vs_total",
    ]
    monthly_numeric_desc = _numeric_descriptives(df, monthly_numeric_cols)
    monthly_numeric_desc.to_csv(TABLES_CAPA1_EDA / "capa1_mensual_numeric_descriptives.csv", index=False)

    # Media anual de los índices: permite ver la tendencia interanual
    # suavizando la estacionalidad mensual
    media_anual = (
        df.groupby("anio")[["indice_retail_moda", "indice_retail_total"]]
        .mean()
        .reset_index()
    )
    media_anual.to_csv(TABLES_CAPA1_EDA / "capa1_mensual_media_anual.csv", index=False)

    # Evolución mensual de ambos índices: visualiza el shock de 2020
    # y la recuperación diferencial del sector moda frente al retail total
    plt.figure(figsize=(12, 6))
    plt.plot(df["fecha"], df["indice_retail_moda"], label="Retail moda")
    plt.plot(df["fecha"], df["indice_retail_total"], label="Retail total")
    plt.title("Evolución mensual del retail de moda frente al retail total")
    plt.xlabel("Fecha")
    plt.ylabel("Índice")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("capa1_mensual_moda_vs_total.png", FIGURES_CAPA1_EDA)

    # Ratio moda/total: evidencia el desacoplamiento estructural entre ambas series.
    # Valores >1 indican que la moda supera al retail general; valores <1, lo contrario.
    plt.figure(figsize=(12, 6))
    plt.plot(df["fecha"], df["ratio_moda_vs_total"])
    plt.title("Ratio entre retail de moda y retail total")
    plt.xlabel("Fecha")
    plt.ylabel("Ratio")
    plt.grid(True, alpha=0.3)
    _save_plot("capa1_mensual_ratio_moda_vs_total.png", FIGURES_CAPA1_EDA)

    # Diferencia absoluta entre índices: la línea horizontal en 0 es la referencia.
    # Valores negativos indican que la moda está por debajo del retail general.
    plt.figure(figsize=(12, 6))
    plt.plot(df["fecha"], df["dif_moda_vs_total"])
    plt.axhline(0, linestyle="--")
    plt.title("Diferencia entre retail de moda y retail total")
    plt.xlabel("Fecha")
    plt.ylabel("Diferencia")
    plt.grid(True, alpha=0.3)
    _save_plot("capa1_mensual_dif_moda_vs_total.png", FIGURES_CAPA1_EDA)

    # Media anual para visualizar tendencia sin ruido estacional
    plt.figure(figsize=(10, 6))
    plt.plot(media_anual["anio"], media_anual["indice_retail_moda"], marker="o", label="Retail moda")
    plt.plot(media_anual["anio"], media_anual["indice_retail_total"], marker="o", label="Retail total")
    plt.title("Media anual del retail de moda frente al retail total")
    plt.xlabel("Año")
    plt.ylabel("Índice medio anual")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("capa1_mensual_media_anual_moda_vs_total.png", FIGURES_CAPA1_EDA)

    # Heatmap año × mes para visualizar el patrón estacional del retail de moda.
    # Permite ver de un vistazo en qué meses y años se concentran los picos de actividad.
    pivot_moda = df.pivot(index="anio", columns="mes", values="indice_retail_moda")
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot_moda, aspect="auto")
    plt.colorbar(label="Índice retail moda")
    plt.xticks(range(12), range(1, 13))
    plt.yticks(range(len(pivot_moda.index)), pivot_moda.index)
    plt.title("Heatmap mensual del retail de moda")
    plt.xlabel("Mes")
    plt.ylabel("Año")
    _save_plot("capa1_mensual_heatmap_moda.png", FIGURES_CAPA1_EDA)

    # Heatmap del ratio: muestra la evolución relativa de la moda frente al
    # retail total mes a mes, evidenciando los meses de mayor/menor desacoplamiento
    pivot_ratio = df.pivot(index="anio", columns="mes", values="ratio_moda_vs_total")
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot_ratio, aspect="auto")
    plt.colorbar(label="Ratio moda / total")
    plt.xticks(range(12), range(1, 13))
    plt.yticks(range(len(pivot_ratio.index)), pivot_ratio.index)
    plt.title("Heatmap del ratio retail moda frente a retail total")
    plt.xlabel("Mes")
    plt.ylabel("Año")
    _save_plot("capa1_mensual_heatmap_ratio.png", FIGURES_CAPA1_EDA)

    print("EDA mensual completado.")
    print("Descriptivos guardados en:", TABLES_CAPA1_EDA / "capa1_mensual_descriptivos.csv")
    print("Descriptivos numéricos guardados en:", TABLES_CAPA1_EDA / "capa1_mensual_numeric_descriptives.csv")
    print("Resumen anual guardado en:", TABLES_CAPA1_EDA / "capa1_mensual_media_anual.csv")
    print("Gráficos guardados en:", FIGURES_CAPA1_EDA)

    return df, media_anual


# =========================
# 6. DESCOMPOSICION TEMPORAL
# =========================

def decomposition_capa1_mensual() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_analysis.csv"
    df = pd.read_csv(input_path)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    # Se descomponen las tres series principales del master mensual.
    # El modelo aditivo Y(t) = T(t) + S(t) + R(t) es apropiado porque
    # la amplitud estacional es constante independientemente del nivel de la serie,
    # condición verificada en el análisis descriptivo previo.
    series_cols = [
        "indice_retail_moda",
        "indice_retail_total",
        "ratio_moda_vs_total",
    ]

    records = []

    for col in series_cols:
        ts = df[["fecha", col]].copy()
        ts[col] = pd.to_numeric(ts[col], errors="coerce")
        ts = ts.dropna()
        # Convierte a serie temporal con frecuencia mensual estricta (Month Start)
        ts = ts.set_index("fecha")[col].asfreq("MS")

        # Mínimo de 24 observaciones requerido para una descomposición robusta
        # con periodo 12 (dos ciclos estacionales completos)
        if ts.dropna().shape[0] < 24:
            records.append(
                {
                    "serie": col,
                    "n_obs_validas": int(ts.dropna().shape[0]),
                    "periodo_estacional": 12,
                    "descomposicion_realizada": "no",
                    "comentario": "Serie demasiado corta para una descomposición robusta.",
                }
            )
            continue

        # Descomposición aditiva con periodo 12 (estacionalidad anual).
        # extrapolate_trend="freq" rellena los NaN de los extremos de la tendencia
        # usando la frecuencia de la serie, evitando valores ausentes en los bordes.
        result = seasonal_decompose(ts, model="additive", period=12, extrapolate_trend="freq")

        # Exporta los componentes descompuestos para análisis adicional
        comp_df = pd.DataFrame(
            {
                "fecha": ts.index,
                "observed": result.observed.values,
                "trend": result.trend.values,
                "seasonal": result.seasonal.values,
                "resid": result.resid.values,
            }
        )
        comp_df.to_csv(TABLES_CAPA1_EDA / f"capa1_decomposition_{col}.csv", index=False)

        fig = result.plot()
        fig.set_size_inches(12, 8)
        plt.suptitle(f"Descomposición temporal - {col}", y=1.02)
        _save_plot(f"capa1_decomposition_{col}.png", FIGURES_CAPA1_EDA)

        records.append(
            {
                "serie": col,
                "n_obs_validas": int(ts.dropna().shape[0]),
                "periodo_estacional": 12,
                "descomposicion_realizada": "si",
                "comentario": "Se separan los componentes de tendencia, estacionalidad y residuo.",
            }
        )

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(TABLES_CAPA1_EDA / "capa1_decomposition_summary.csv", index=False)

    print("Descomposición temporal completada.")
    print(summary_df)

    return summary_df


# =========================
# 7. OUTLIERS MENSUALES
# =========================

def outliers_capa1_mensual() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_analysis.csv"
    df = pd.read_csv(input_path)
    df["fecha"] = pd.to_datetime(df["fecha"])

    numeric_cols = [
        "indice_retail_moda",
        "indice_retail_total",
        "ratio_moda_vs_total",
        "dif_moda_vs_total",
    ]

    outlier_flags = pd.DataFrame(index=df.index)

    for col in numeric_cols:
        # Detección de outliers por método IQR (Tukey):
        # se consideran atípicos los valores fuera de [Q1 - 1.5·IQR, Q3 + 1.5·IQR].
        # En series temporales con shocks estructurales (2020), los outliers son
        # esperados y se documentan sin eliminarlos del análisis.
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_flags[col + "_outlier"] = (df[col] < lower) | (df[col] > upper)

        # Boxplot individual por variable para visualizar la distribución
        # y los valores atípicos identificados por el método IQR
        plt.figure(figsize=(8, 5))
        plt.boxplot(df[col].dropna())
        plt.title(f"Boxplot - {col}")
        plt.ylabel(col)
        _save_plot(f"boxplot_{col}.png", FIGURES_CAPA1_EDA)

    # Filtra solo las filas con al menos un outlier en cualquier variable
    mask_any = outlier_flags.any(axis=1)
    outliers_df = pd.concat([df, outlier_flags], axis=1)
    outliers_df = outliers_df[mask_any].copy().reset_index(drop=True)

    outliers_df.to_csv(TABLES_CAPA1_EDA / "capa1_mensual_outliers_iqr.csv", index=False)

    print("Outliers detectados.")
    print("Archivo guardado en:")
    print(TABLES_CAPA1_EDA / "capa1_mensual_outliers_iqr.csv")
    print("")
    print(outliers_df.head(20))

    return outliers_df


# =========================
# 7B. RESUMEN DE OUTLIERS
# =========================

def outlier_summary_capa1_mensual() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_analysis.csv"
    df = pd.read_csv(input_path)

    numeric_cols = [
        "indice_retail_moda",
        "indice_retail_total",
        "ratio_moda_vs_total",
        "dif_moda_vs_total",
    ]

    records = []

    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()

         # Cálculo de los límites IQR para documentar los umbrales exactos
        # que definen la frontera entre valores típicos y atípicos en cada serie
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (s < lower) | (s > upper)
        n_outliers = int(mask.sum())
        pct_outliers = round((n_outliers / len(s)) * 100, 2) if len(s) > 0 else 0.0

        # La decisión metodológica es mantener todos los outliers como shocks
        # estructurales (pandemia 2020) y no como errores de datos.
        # Esta decisión se documenta explícitamente para el corrector.
        records.append(
            {
                "variable": col,
                "n_obs": int(len(s)),
                "q1": round(float(q1), 4),
                "q3": round(float(q3), 4),
                "iqr": round(float(iqr), 4),
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
                "n_outliers": n_outliers,
                "pct_outliers": pct_outliers,
                "tipo_atipicidad": "extremo_estadistico_no_error",
                "decision_metodologica": "mantener_por_shock_estructural",
            }
        )

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(TABLES_CAPA1_EDA / "capa1_mensual_outliers_summary.csv", index=False)

    print("Resumen de outliers completado.")
    print(summary_df)

    return summary_df


# =========================
# 8. CORRELACION MENSUAL
# =========================

def correlation_capa1_mensual() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_analysis.csv"
    df = pd.read_csv(input_path)

    # Variables del master mensual para el análisis de correlación.
    # Se incluyen las cuatro variables principales para detectar relaciones
    # lineales entre índices directos y variables derivadas.
    cols = [
        "indice_retail_moda",
        "indice_retail_total",
        "ratio_moda_vs_total",
        "dif_moda_vs_total",
    ]
    corr = df[cols].corr()

    corr.to_csv(TABLES_CAPA1_EDA / "capa1_mensual_correlation.csv")

    # Heatmap de correlación: visualiza la fuerza y dirección de las relaciones
    # entre variables del master mensual
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlación")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Matriz de correlación - capa 1 mensual")
    _save_plot("capa1_mensual_correlation_heatmap.png", FIGURES_CAPA1_EDA)

    print("Correlación calculada.")
    print(corr)
    print("")
    print("Archivos guardados en:")
    print(TABLES_CAPA1_EDA / "capa1_mensual_correlation.csv")
    print(FIGURES_CAPA1_EDA / "capa1_mensual_correlation_heatmap.png")

    return corr


# =========================
# 9. SOURCE COVERAGE CON JUSTIFICACION
# =========================

def source_coverage_capa1() -> pd.DataFrame:
    """
    Cobertura temporal y justificación analítica de inclusión de cada fuente.
    Refuerza la parte de trazabilidad y control de ingeniería del dato.
    """
    _ensure_dirs()

    source_map = {
        "contexto_digitalizacion_clean": PROCESSED_CAPA1 / "contexto_digitalizacion" / "contexto_digitalizacion_clean.csv",
        "contexto_digitalizacion_documentado": PROCESSED_CAPA1 / "contexto_digitalizacion" / "contexto_digitalizacion_documentado.csv",
        "eurostat_moda_mensual_clean": PROCESSED_CAPA1 / "eurostat" / "eurostat_moda_mensual_clean.csv",
        "eurostat_retail_total_mensual_clean": PROCESSED_CAPA1 / "eurostat" / "eurostat_retail_total_mensual_clean.csv",
        "eurostat_online_empresas_clean": PROCESSED_CAPA1 / "eurostat" / "eurostat_online_empresas_clean.csv",
        "comercio_electronico_core_std": PROCESSED_CAPA1 / "comercio_electronico" / "comercio_electronico_core_std.csv",
        "capa1_master_anual_analysis": PROCESSED_CAPA1 / "integrated" / "capa1_master_anual_analysis.csv",
        "capa1_master_mensual_analysis": PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_analysis.csv",
    }

    uso_map = {
        "contexto_digitalizacion_clean": "contexto_macro_anual",
        "contexto_digitalizacion_documentado": "trazabilidad_contextual",
        "eurostat_moda_mensual_clean": "serie_temporal_sectorial_mensual",
        "eurostat_retail_total_mensual_clean": "serie_temporal_referencia_mensual",
        "eurostat_online_empresas_clean": "adopcion_online_empresarial_anual",
        "comercio_electronico_core_std": "ecommerce_empresarial_anual_segmentado",
        "capa1_master_anual_analysis": "master_analitico_anual_integrado",
        "capa1_master_mensual_analysis": "master_analitico_mensual_integrado",
    }

    # Justificaciones metodológicas de inclusión de cada fuente.
    # Documentan por qué cada dataset fue seleccionado y qué limitaciones
    # debe tener en cuenta el analista al interpretarlo.
    justificaciones = {
        "contexto_digitalizacion_clean": (
            "Incluye la serie anual principal de contexto digital. El valor imputado de 2025, "
            "si existe, debe usarse solo como apoyo contextual y con trazabilidad explícita."
        ),
        "contexto_digitalizacion_documentado": (
            "Dataset orientado a trazabilidad documental y soporte metodológico, no a análisis cuantitativo principal."
        ),
        "eurostat_moda_mensual_clean": (
            "Serie principal mensual del sector moda utilizada para analizar patrón temporal del retail especializado."
        ),
        "eurostat_retail_total_mensual_clean": (
            "Serie de referencia general del retail. Su función es contextualizar el desempeño relativo del retail de moda."
        ),
        "eurostat_online_empresas_clean": (
            "Serie anual de adopción empresarial del canal online, útil para contextualizar digitalización de la oferta."
        ),
        "comercio_electronico_core_std": (
            "Fuente anual segmentada por tamaño de empresa. Se utiliza para integrar indicadores de ecommerce "
            "empresarial, priorizando el agregado total para asegurar comparabilidad."
        ),
        "capa1_master_anual_analysis": (
            "Integración anual final de variables comparables entre fuentes. Se limita al tramo común con cobertura suficiente."
        ),
        "capa1_master_mensual_analysis": (
            "Integración mensual final del retail de moda frente al retail total y variables derivadas para comparación relativa."
        ),
    }

    records = []

    for source_name, path in source_map.items():
        df = pd.read_csv(path)
        start_period, end_period = _get_period_bounds(df)

        records.append(
            {
                "fuente": source_name,
                "fecha_inicio": start_period,
                "fecha_fin": end_period,
                "n_filas": int(df.shape[0]),
                "n_columnas": int(df.shape[1]),
                "uso_analitico": uso_map.get(source_name, "otro"),
                "nota_metodologica": justificaciones.get(source_name, ""),
            }
        )

    coverage_df = pd.DataFrame(records)
    coverage_df.to_csv(TABLES_CAPA1_CONTROL / "capa1_source_coverage.csv", index=False)

    print(f"Source coverage: {len(coverage_df)} fuentes documentadas")
    return coverage_df


# =========================
# RUN ALL
# =========================

def run_all_eda() -> None:
    sep = "=" * 65
    print(sep)
    print("CAPA 1 — EDA")
    print(sep)

    print("\n[1/9] Profiling general...")
    profile_capa1()

    print("\n[2/9] Análisis de nulos...")
    analyze_nulls_capa1()

    print("\n[3/9] Justificación tipo análisis descriptivo...")
    eda_capa1_justificacion_analitica()

    print("\n[4/9] EDA anual...")
    eda_capa1_anual()

    print("\n[5/9] EDA mensual...")
    eda_capa1_mensual()

    print("\n[6/9] Descomposición temporal...")
    decomposition_capa1_mensual()

    print("\n[7/9] Outliers mensuales...")
    outliers_capa1_mensual()
    outlier_summary_capa1_mensual()

    print("\n[8/9] Correlación mensual...")
    correlation_capa1_mensual()

    print("\n[9/9] Source coverage...")
    source_coverage_capa1()

    print(f"\n{sep}")
    print("EDA CAPA 1 COMPLETADO")
    print("Tablas clave generadas:")
    print("  · capa1_profile_summary.csv")
    print("  · capa1_nulls_decision_matrix.csv")
    print("  · capa1_justificacion_tipo_analisis.csv")
    print("  · capa1_anual_descriptivos.csv")
    print("  · capa1_mensual_descriptivos.csv")
    print("  · capa1_decomposition_summary.csv")
    print("  · capa1_mensual_outliers_iqr.csv")
    print("  · capa1_mensual_outliers_summary.csv")
    print("  · capa1_mensual_correlation.csv")
    print("  · capa1_source_coverage.csv")
    print(sep)


if __name__ == "__main__":
    run_all_eda()