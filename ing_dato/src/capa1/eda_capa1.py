import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    TABLES_CAPA1.mkdir(parents=True, exist_ok=True)
    TABLES_CAPA1_CONTROL.mkdir(parents=True, exist_ok=True)
    TABLES_CAPA1_EDA.mkdir(parents=True, exist_ok=True)

    FIGURES_CAPA1.mkdir(parents=True, exist_ok=True)
    FIGURES_CAPA1_CONTROL.mkdir(parents=True, exist_ok=True)
    FIGURES_CAPA1_EDA.mkdir(parents=True, exist_ok=True)


def _save_plot(filename: str, folder) -> None:
    plt.tight_layout()
    plt.savefig(folder / filename, dpi=300, bbox_inches="tight")
    plt.close()


def _numeric_descriptives(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
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


# =========================
# 1. PROFILE GENERAL
# =========================

def profile_capa1() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()

    datasets = {
        "contexto_digitalizacion_clean": PROCESSED_CAPA1 / "contexto_digitalizacion" / "contexto_digitalizacion_clean.csv",
        "contexto_digitalizacion_extended": PROCESSED_CAPA1 / "contexto_digitalizacion" / "contexto_digitalizacion_extended.csv",
        "eurostat_moda_mensual_clean": PROCESSED_CAPA1 / "eurostat" / "eurostat_moda_mensual_clean.csv",
        "eurostat_retail_total_mensual_clean": PROCESSED_CAPA1 / "eurostat" / "eurostat_retail_total_mensual_clean.csv",
        "eurostat_online_empresas_clean": PROCESSED_CAPA1 / "eurostat" / "eurostat_online_empresas_clean.csv",
        "comercio_electronico_core_std": PROCESSED_CAPA1 / "comercio_electronico" / "comercio_electronico_core_std.csv",
        "capa1_master_anual_analysis": PROCESSED_CAPA1 / "integrated" / "capa1_master_anual_analysis.csv",
        "capa1_master_mensual_analysis": PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_analysis.csv",
        "contexto_digitalizacion_documentado": PROCESSED_CAPA1 / "contexto_digitalizacion" / "contexto_digitalizacion_documentado.csv",
    }

    summary_records = []
    nulls_records = []
    numeric_records = []

    for name, path in datasets.items():
        df = pd.read_csv(path)

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

        summary_records.append(
            {
                "dataset": name,
                "rows": df.shape[0],
                "cols": df.shape[1],
                "start_period": start_period,
                "end_period": end_period,
                "columns": ", ".join(df.columns.tolist()),
            }
        )

        for col in df.columns:
            nulls_records.append(
                {
                    "dataset": name,
                    "variable": col,
                    "n_nulls": int(df[col].isna().sum()),
                    "pct_nulls": round(float(df[col].isna().mean() * 100), 2),
                }
            )

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

    detailed = df[df["pct_nulls"] > 0].copy().sort_values(
        by=["dataset", "pct_nulls"], ascending=[True, False]
    )
    detailed.to_csv(TABLES_CAPA1_CONTROL / "capa1_nulls_detailed.csv", index=False)

    def assign_null_decision(row: pd.Series) -> str:
        dataset = str(row["dataset"])
        variable = str(row["variable"])
        pct = float(row["pct_nulls"])

        if variable in {"comentarios_hitos", "fuente_usuarios_redes", "fuente_compra_online", "fuente_compra_ropa_online"}:
            return "mantener_solo_trazabilidad"
        if variable == "pct_personas_compra_ropa_online":
            return "mantener_solo_contexto"
        if pct >= 80:
            return "descartar_analiticamente"
        if pct >= 20:
            return "mantener_con_cautela"
        return "mantener"

    def assign_null_comment(row: pd.Series) -> str:
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
# 3. EDA ANUAL
# =========================

def eda_capa1_anual() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA1 / "integrated" / "capa1_master_anual_analysis.csv"
    df = pd.read_csv(input_path)

    descriptivos = df.describe(include="all")
    descriptivos.to_csv(TABLES_CAPA1_EDA / "capa1_anual_descriptivos.csv")

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

    # RRSS vs compra online
    plt.figure(figsize=(8, 5))
    plt.plot(df["anio"], df["pct_usuarios_rrss"], marker="o", label="Usuarios RRSS (%)")
    plt.plot(df["anio"], df["pct_personas_compra_online"], marker="o", label="Personas que compran online (%)")
    plt.title("Evolución del uso de RRSS y la compra online en España")
    plt.xlabel("Año")
    plt.ylabel("Porcentaje")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("capa1_anual_rrss_vs_compra_online.png", FIGURES_CAPA1_EDA)

    # Empresas que venden
    plt.figure(figsize=(8, 5))
    plt.plot(df["anio"], df["pct_empresas_venden_ecommerce"], marker="o", label="Empresas que venden ecommerce (%)")
    plt.plot(df["anio"], df["pct_empresas_venden_web_apps"], marker="o", label="Empresas que venden por web/apps (%)")
    plt.title("Adopción empresarial del ecommerce")
    plt.xlabel("Año")
    plt.ylabel("Porcentaje")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("capa1_anual_empresas_venden_online.png", FIGURES_CAPA1_EDA)

    # Facturación online
    plt.figure(figsize=(8, 5))
    plt.plot(df["anio"], df["pct_facturacion_empresas_online"], marker="o")
    plt.title("Porcentaje de facturación empresarial procedente de ventas online")
    plt.xlabel("Año")
    plt.ylabel("Porcentaje")
    plt.grid(True, alpha=0.3)
    _save_plot("capa1_anual_facturacion_online_empresas.png", FIGURES_CAPA1_EDA)

    # Peso ecommerce
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
    print("Gráficos guardados en:", FIGURES_CAPA1_EDA)

    return df


# =========================
# 4. EDA MENSUAL
# =========================

def eda_capa1_mensual() -> tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()

    input_path = PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_analysis.csv"
    df = pd.read_csv(input_path)
    df["fecha"] = pd.to_datetime(df["fecha"])

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

    media_anual = (
        df.groupby("anio")[["indice_retail_moda", "indice_retail_total"]]
        .mean()
        .reset_index()
    )
    media_anual.to_csv(TABLES_CAPA1_EDA / "capa1_mensual_media_anual.csv", index=False)

    # Evolución mensual
    plt.figure(figsize=(12, 6))
    plt.plot(df["fecha"], df["indice_retail_moda"], label="Retail moda")
    plt.plot(df["fecha"], df["indice_retail_total"], label="Retail total")
    plt.title("Evolución mensual del retail de moda frente al retail total")
    plt.xlabel("Fecha")
    plt.ylabel("Índice base 2015=100")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("capa1_mensual_moda_vs_total.png", FIGURES_CAPA1_EDA)

    # Ratio
    plt.figure(figsize=(12, 6))
    plt.plot(df["fecha"], df["ratio_moda_vs_total"])
    plt.title("Ratio entre retail de moda y retail total")
    plt.xlabel("Fecha")
    plt.ylabel("Ratio")
    plt.grid(True, alpha=0.3)
    _save_plot("capa1_mensual_ratio_moda_vs_total.png", FIGURES_CAPA1_EDA)

    # Diferencia
    plt.figure(figsize=(12, 6))
    plt.plot(df["fecha"], df["dif_moda_vs_total"])
    plt.axhline(0, linestyle="--")
    plt.title("Diferencia entre retail de moda y retail total")
    plt.xlabel("Fecha")
    plt.ylabel("Diferencia de índices")
    plt.grid(True, alpha=0.3)
    _save_plot("capa1_mensual_dif_moda_vs_total.png", FIGURES_CAPA1_EDA)

    # Media anual
    plt.figure(figsize=(10, 6))
    plt.plot(media_anual["anio"], media_anual["indice_retail_moda"], marker="o", label="Retail moda")
    plt.plot(media_anual["anio"], media_anual["indice_retail_total"], marker="o", label="Retail total")
    plt.title("Media anual del retail de moda frente al retail total")
    plt.xlabel("Año")
    plt.ylabel("Índice medio anual")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("capa1_mensual_media_anual_moda_vs_total.png", FIGURES_CAPA1_EDA)

    # Heatmap moda
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

    # Heatmap ratio
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
    print("Resumen anual guardado en:", TABLES_CAPA1_EDA / "capa1_mensual_media_anual.csv")
    print("Gráficos guardados en:", FIGURES_CAPA1_EDA)

    return df, media_anual


# =========================
# 5. OUTLIERS MENSUALES
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
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_flags[col + "_outlier"] = (df[col] < lower) | (df[col] > upper)

        plt.figure(figsize=(8, 5))
        plt.boxplot(df[col].dropna())
        plt.title(f"Boxplot - {col}")
        plt.ylabel(col)
        _save_plot(f"boxplot_{col}.png", FIGURES_CAPA1_EDA)

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
# 6. CORRELACION MENSUAL
# =========================

def correlation_capa1_mensual() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA1 / "integrated" / "capa1_master_mensual_analysis.csv"
    df = pd.read_csv(input_path)

    cols = [
        "indice_retail_moda",
        "indice_retail_total",
        "ratio_moda_vs_total",
        "dif_moda_vs_total",
    ]
    corr = df[cols].corr()

    corr.to_csv(TABLES_CAPA1_EDA / "capa1_mensual_correlation.csv")

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
# 7. SOURCE COVERAGE
# =========================

def source_coverage_capa1() -> pd.DataFrame:
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

    records = []

    for source_name, path in source_map.items():
        df = pd.read_csv(path)

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
                start_period = str(int(anios.min()))
                end_period = str(int(anios.max()))

        records.append(
            {
                "fuente": source_name,
                "fecha_inicio": start_period,
                "fecha_fin": end_period,
                "n_filas": int(df.shape[0]),
                "n_columnas": int(df.shape[1]),
            }
        )

    coverage_df = pd.DataFrame(records)

    use_map = {
        "contexto_digitalizacion_clean": "contexto_macro_anual",
        "contexto_digitalizacion_documentado": "trazabilidad_contextual",
        "eurostat_moda_mensual_clean": "historico_sectorial_mensual",
        "eurostat_retail_total_mensual_clean": "historico_general_mensual",
        "eurostat_online_empresas_clean": "adopcion_online_empresarial",
        "comercio_electronico_core_std": "ecommerce_empresarial_anual",
        "capa1_master_anual_analysis": "master_anual_final",
        "capa1_master_mensual_analysis": "master_mensual_final",
    }

    coverage_df["uso_analitico"] = coverage_df["fuente"].map(use_map)
    coverage_df.to_csv(TABLES_CAPA1_CONTROL / "capa1_source_coverage.csv", index=False)

    print("Cobertura temporal de fuentes calculada.")
    print(coverage_df)

    return coverage_df


# =========================
# RUN ALL
# =========================

def run_all_eda() -> None:
    profile_capa1()
    analyze_nulls_capa1()
    eda_capa1_anual()
    eda_capa1_mensual()
    outliers_capa1_mensual()
    correlation_capa1_mensual()
    source_coverage_capa1()
    print("Todo el EDA de capa 1 completado.")


if __name__ == "__main__":
    run_all_eda()