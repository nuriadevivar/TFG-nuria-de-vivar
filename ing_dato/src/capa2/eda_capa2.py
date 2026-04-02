import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import (
    FIGURES_CAPA2_EDA,
    PROCESSED_CAPA2,
    TABLES_CAPA2_CONTROL,
    TABLES_CAPA2_EDA,
)


# =========================
# HELPERS
# =========================

def _ensure_dirs() -> None:
    TABLES_CAPA2_EDA.mkdir(parents=True, exist_ok=True)
    TABLES_CAPA2_CONTROL.mkdir(parents=True, exist_ok=True)
    FIGURES_CAPA2_EDA.mkdir(parents=True, exist_ok=True)


def _save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_CAPA2_EDA / filename, dpi=300, bbox_inches="tight")
    plt.close()


def _safe_read_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def _normalize_series_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    min_val = s.min()
    max_val = s.max()
    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series(np.nan, index=series.index)
    if max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (s - min_val) / (max_val - min_val)


def _plot_correlation_matrix(corr_df: pd.DataFrame, title: str, filename: str) -> None:
    labels = corr_df.columns.tolist()

    plt.figure(figsize=(9, 7))
    plt.imshow(corr_df, aspect="auto")
    plt.colorbar(label="Correlación")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    _save_plot(filename)


# =========================
# HELPER OUTLIERS
# =========================

def _iqr_outlier_summary(df: pd.DataFrame, cols: list[str], bloque: str) -> pd.DataFrame:
    records = []

    for col in cols:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (s < lower) | (s > upper)
        n_outliers = int(mask.sum())
        pct_outliers = round((n_outliers / len(s)) * 100, 2) if len(s) > 0 else 0.0

        records.append(
            {
                "bloque": bloque,
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
                "decision_metodologica": "mantener_y_documentar",
            }
        )

    return pd.DataFrame(records)


# =========================
# 1. PROFILE GENERAL
# =========================

def profile_capa2() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()

    datasets = {
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
        "capa2_master_terminos_mensual": PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_mensual.csv",
        "capa2_term_coverage": PROCESSED_CAPA2 / "integrated" / "capa2_term_coverage.csv",
        "capa2_master_productos_mensual": PROCESSED_CAPA2 / "integrated" / "capa2_master_productos_mensual.csv",
        "capa2_master_eventos": PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos.csv",
        "capa2_master_eventos_mensual": PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos_mensual.csv",
        "capa2_master_integrated": PROCESSED_CAPA2 / "integrated" / "capa2_master_integrated.csv",
        "capa2_master_social": PROCESSED_CAPA2 / "integrated" / "capa2_master_social.csv",
        "capa2_master_brand_digital": PROCESSED_CAPA2 / "integrated" / "capa2_master_brand_digital.csv",
        "capa2_term_analysis_priority": PROCESSED_CAPA2 / "integrated" / "capa2_term_analysis_priority.csv",
        "capa2_term_quality_decisions": PROCESSED_CAPA2 / "integrated" / "capa2_term_quality_decisions.csv",
        "capa2_master_terminos_main": PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_main.csv",
    }

    summary_records = []
    nulls_records = []
    numeric_records = []

    for name, path in datasets.items():
        df = _safe_read_csv(path)
        if df is None:
            continue

        start_period = None
        end_period = None

        if "fecha" in df.columns:
            fechas = pd.to_datetime(df["fecha"], errors="coerce")
            if fechas.notna().any():
                start_period = str(fechas.min().date())
                end_period = str(fechas.max().date())
        elif "fecha_post" in df.columns:
            fechas = pd.to_datetime(df["fecha_post"], errors="coerce")
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

    summary_df.to_csv(TABLES_CAPA2_CONTROL / "capa2_profile_summary.csv", index=False)
    nulls_df.to_csv(TABLES_CAPA2_CONTROL / "capa2_profile_nulls.csv", index=False)
    numeric_df.to_csv(TABLES_CAPA2_CONTROL / "capa2_profile_numeric.csv", index=False)

    print("Profiling capa 2 completado.")
    print(summary_df)

    return summary_df, nulls_df, numeric_df


# =========================
# 2. NULOS
# =========================

def analyze_nulls_capa2() -> tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()

    input_path = TABLES_CAPA2_CONTROL / "capa2_profile_nulls.csv"
    df = pd.read_csv(input_path)

    detailed = df[df["pct_nulls"] > 0].copy().sort_values(
        by=["dataset", "pct_nulls"], ascending=[True, False]
    )
    detailed.to_csv(TABLES_CAPA2_CONTROL / "capa2_nulls_detailed.csv", index=False)

    def assign_null_decision(row: pd.Series) -> str:
        dataset = str(row["dataset"])
        variable = str(row["variable"])
        pct = row["pct_nulls"]

        if variable == "hashtags":
            return "descartar_analiticamente"
        if variable in {"caption", "perfil_nombre"} and dataset == "instagram_posts_clean":
            return "mantener_con_cautela"
        if variable == "valor_trends" and dataset in {
            "trends_sofisticado_clean",
            "trends_urbano_clean",
            "trends_moda_total_clean",
            "capa2_master_terminos_mensual",
            "capa2_master_terminos_main",
            "capa2_master_integrated",
        }:
            return "mantener_con_priorizacion"
        if pct >= 80:
            return "descartar_analiticamente"
        if pct >= 20:
            return "mantener_con_cautela"
        return "mantener"

    def assign_null_comment(row: pd.Series) -> str:
        dataset = str(row["dataset"])
        variable = str(row["variable"])
        pct = row["pct_nulls"]

        if variable == "hashtags":
            return "Campo no recuperado de forma útil por Apify; no se usará en análisis."
        if variable == "caption":
            return "Variable textual utilizable, aunque con algunos faltantes; no afecta al núcleo cuantitativo."
        if variable == "valor_trends" and dataset == "trends_sofisticado_clean":
            return "Los nulos responden a baja continuidad/interés de algunos términos sofisticados."
        if variable == "valor_trends" and dataset == "trends_urbano_clean":
            return "Los nulos responden a menor continuidad de algunos términos urbanos."
        if variable == "valor_trends" and dataset == "trends_moda_total_clean":
            return "Existen términos con señal parcial; se controlan mediante quality_flag y prioridad analítica."
        if pct >= 80:
            return "Nivel de nulos demasiado alto para uso analítico robusto."
        if pct >= 20:
            return "Variable válida solo con cautela e interpretación prudente."
        return "Nivel de nulos bajo y asumible."

    decisions = detailed.copy()
    decisions["decision_analitica"] = decisions.apply(assign_null_decision, axis=1)
    decisions["comentario"] = decisions.apply(assign_null_comment, axis=1)
    decisions.to_csv(TABLES_CAPA2_CONTROL / "capa2_nulls_decision_matrix.csv", index=False)

    datasets_to_plot = [
        "capa2_master_terminos_mensual",
        "instagram_posts_clean",
        "capa2_master_brand_digital",
    ]

    for dataset_name in datasets_to_plot:
        subset = detailed[detailed["dataset"] == dataset_name].copy()
        if subset.empty:
            continue

        plt.figure(figsize=(10, 4))
        plt.bar(subset["variable"], subset["pct_nulls"])
        plt.title(f"Nulos por variable - {dataset_name}")
        plt.ylabel("% nulos")
        plt.xticks(rotation=45, ha="right")
        _save_plot(f"{dataset_name}_nulls.png")

    print("Análisis de nulos capa 2 completado.")
    return detailed, decisions


# =========================
# 3. EDA TÉRMINOS
# =========================

def eda_capa2_terminos() -> tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_mensual.csv"
    coverage_path = PROCESSED_CAPA2 / "integrated" / "capa2_term_coverage.csv"

    df = pd.read_csv(input_path)
    coverage = pd.read_csv(coverage_path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    descriptivos = df.describe(include="all")
    descriptivos.to_csv(TABLES_CAPA2_EDA / "capa2_terminos_descriptivos.csv")

    numeric_desc = df[["valor_trends"]].describe().T
    numeric_desc.to_csv(TABLES_CAPA2_EDA / "capa2_terminos_numeric_descriptives.csv")

    ranking_terminos = (
        df.groupby(["grupo", "termino", "subgrupo", "tipo_termino", "familia_analitica"], dropna=False)
        .agg(
            valor_trends_medio=("valor_trends", "mean"),
            valor_trends_mediana=("valor_trends", "median"),
            n_obs=("valor_trends", "size"),
            n_non_null=("valor_trends", lambda s: s.notna().sum()),
        )
        .reset_index()
        .sort_values("valor_trends_medio", ascending=False)
    )
    ranking_terminos.to_csv(TABLES_CAPA2_EDA / "capa2_ranking_terminos.csv", index=False)

    ranking_grupo = (
        df.groupby("grupo", dropna=False)["valor_trends"]
        .mean()
        .reset_index()
        .rename(columns={"valor_trends": "valor_trends_medio"})
        .sort_values("valor_trends_medio", ascending=False)
    )
    ranking_grupo.to_csv(TABLES_CAPA2_EDA / "capa2_ranking_grupo.csv", index=False)

    ranking_familia = (
        df.groupby("familia_analitica", dropna=False)["valor_trends"]
        .mean()
        .reset_index()
        .rename(columns={"valor_trends": "valor_trends_medio"})
        .sort_values("valor_trends_medio", ascending=False)
    )
    ranking_familia.to_csv(TABLES_CAPA2_EDA / "capa2_ranking_familia_analitica.csv", index=False)

    coverage.to_csv(TABLES_CAPA2_CONTROL / "capa2_term_coverage_eda.csv", index=False)

    trends_by_group = (
        df.groupby(["fecha", "grupo"], dropna=False)["valor_trends"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    for grp in sorted(trends_by_group["grupo"].dropna().unique()):
        subset = trends_by_group[trends_by_group["grupo"] == grp]
        plt.plot(subset["fecha"], subset["valor_trends"], label=grp)
    plt.title("Evolución media mensual por grupo")
    plt.xlabel("Fecha")
    plt.ylabel("Valor medio Trends")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("capa2_terminos_evolucion_por_grupo.png")

    trends_by_family = (
        df.groupby(["fecha", "familia_analitica"], dropna=False)["valor_trends"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    for fam in sorted(trends_by_family["familia_analitica"].dropna().unique()):
        subset = trends_by_family[trends_by_family["familia_analitica"] == fam]
        plt.plot(subset["fecha"], subset["valor_trends"], label=fam)
    plt.title("Evolución media mensual por familia analítica")
    plt.xlabel("Fecha")
    plt.ylabel("Valor medio Trends")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("capa2_terminos_evolucion_por_familia.png")

    top_marcas = (
        df[df["tipo_termino"] == "marca"]
        .groupby("termino", dropna=False)["valor_trends"]
        .mean()
        .reset_index()
        .rename(columns={"valor_trends": "valor_trends_medio"})
        .sort_values("valor_trends_medio", ascending=False)
    )
    top_marcas.to_csv(TABLES_CAPA2_EDA / "capa2_top_marcas.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(top_marcas["termino"], top_marcas["valor_trends_medio"])
    plt.title("Ranking de marcas por interés medio")
    plt.ylabel("Valor medio Trends")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa2_top_marcas.png")

    top_esteticas = (
        df[df["tipo_termino"] == "estetica"]
        .groupby("termino", dropna=False)["valor_trends"]
        .mean()
        .reset_index()
        .rename(columns={"valor_trends": "valor_trends_medio"})
        .sort_values("valor_trends_medio", ascending=False)
    )
    top_esteticas.to_csv(TABLES_CAPA2_EDA / "capa2_top_esteticas.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(top_esteticas["termino"], top_esteticas["valor_trends_medio"])
    plt.title("Ranking de estéticas por interés medio")
    plt.ylabel("Valor medio Trends")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa2_top_esteticas.png")

    plt.figure(figsize=(8, 5))
    plt.boxplot(df["valor_trends"].dropna())
    plt.title("Boxplot de valor_trends - términos")
    plt.ylabel("valor_trends")
    _save_plot("capa2_boxplot_valor_trends_terminos.png")

    top_terms = (
        df.groupby("termino", dropna=False)["valor_trends"]
        .mean()
        .reset_index()
        .sort_values("valor_trends", ascending=False)
        .head(8)["termino"]
        .tolist()
    )

    heat_df = df[df["termino"].isin(top_terms)].copy()
    heat_pivot = heat_df.pivot_table(index="termino", columns="fecha", values="valor_trends", aggfunc="mean")

    plt.figure(figsize=(14, 5))
    plt.imshow(heat_pivot, aspect="auto")
    plt.colorbar(label="valor_trends")
    plt.yticks(range(len(heat_pivot.index)), heat_pivot.index)
    plt.xticks([])
    plt.title("Heatmap de top términos")
    _save_plot("capa2_heatmap_top_terminos.png")

    low_quality = coverage[coverage["quality_flag"] == "baja"].copy().sort_values(
        ["grupo", "pct_non_null"], ascending=[True, False]
    )
    low_quality.to_csv(TABLES_CAPA2_CONTROL / "capa2_terminos_baja_cobertura.csv", index=False)

    print("EDA términos completado.")
    return df, coverage


# =========================
# 3B. EDA TÉRMINOS MAIN
# =========================

def eda_capa2_terminos_main() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_main.csv"
    df = pd.read_csv(input_path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    descriptivos = df.describe(include="all")
    descriptivos.to_csv(TABLES_CAPA2_EDA / "capa2_terminos_main_descriptivos.csv")

    numeric_desc = df[["valor_trends"]].describe().T
    numeric_desc.to_csv(TABLES_CAPA2_EDA / "capa2_terminos_main_numeric_descriptives.csv")

    ranking_terminos_main = (
        df.groupby(
            ["grupo", "termino", "subgrupo", "tipo_termino", "familia_analitica", "prioridad_analitica"],
            dropna=False,
        )
        .agg(
            valor_trends_medio=("valor_trends", "mean"),
            valor_trends_mediana=("valor_trends", "median"),
            n_obs=("valor_trends", "size"),
            n_non_null=("valor_trends", lambda s: s.notna().sum()),
        )
        .reset_index()
        .sort_values("valor_trends_medio", ascending=False)
    )
    ranking_terminos_main.to_csv(TABLES_CAPA2_EDA / "capa2_ranking_terminos_main.csv", index=False)

    ranking_grupo_main = (
        df.groupby("grupo", dropna=False)["valor_trends"]
        .mean()
        .reset_index()
        .rename(columns={"valor_trends": "valor_trends_medio"})
        .sort_values("valor_trends_medio", ascending=False)
    )
    ranking_grupo_main.to_csv(TABLES_CAPA2_EDA / "capa2_ranking_grupo_main.csv", index=False)

    ranking_familia_main = (
        df.groupby("familia_analitica", dropna=False)["valor_trends"]
        .mean()
        .reset_index()
        .rename(columns={"valor_trends": "valor_trends_medio"})
        .sort_values("valor_trends_medio", ascending=False)
    )
    ranking_familia_main.to_csv(TABLES_CAPA2_EDA / "capa2_ranking_familia_analitica_main.csv", index=False)

    evol_grupo_main = (
        df.groupby(["fecha", "grupo"], dropna=False)["valor_trends"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    for grp in sorted(evol_grupo_main["grupo"].dropna().unique()):
        subset = evol_grupo_main[evol_grupo_main["grupo"] == grp]
        plt.plot(subset["fecha"], subset["valor_trends"], label=grp)
    plt.title("Evolución media mensual por grupo (subset principal)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor medio Trends")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("capa2_terminos_evolucion_por_grupo_main.png")

    evol_familia_main = (
        df.groupby(["fecha", "familia_analitica"], dropna=False)["valor_trends"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    for fam in sorted(evol_familia_main["familia_analitica"].dropna().unique()):
        subset = evol_familia_main[evol_familia_main["familia_analitica"] == fam]
        plt.plot(subset["fecha"], subset["valor_trends"], label=fam)
    plt.title("Evolución media mensual por familia analítica (subset principal)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor medio Trends")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("capa2_terminos_evolucion_por_familia_main.png")

    top_main = (
        df.groupby("termino", dropna=False)["valor_trends"]
        .mean()
        .reset_index()
        .rename(columns={"valor_trends": "valor_trends_medio"})
        .sort_values("valor_trends_medio", ascending=False)
        .head(10)
    )
    top_main.to_csv(TABLES_CAPA2_EDA / "capa2_top_terminos_main.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(top_main["termino"], top_main["valor_trends_medio"])
    plt.title("Top términos del subset principal")
    plt.ylabel("Valor medio Trends")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa2_top_terminos_main.png")

    heat_df = df[df["termino"].isin(top_main["termino"].tolist())].copy()
    heat_pivot = heat_df.pivot_table(index="termino", columns="fecha", values="valor_trends", aggfunc="mean")

    plt.figure(figsize=(14, 5))
    plt.imshow(heat_pivot, aspect="auto")
    plt.colorbar(label="valor_trends")
    plt.yticks(range(len(heat_pivot.index)), heat_pivot.index)
    plt.xticks([])
    plt.title("Heatmap de términos principales")
    _save_plot("capa2_heatmap_top_terminos_main.png")

    print("EDA términos main completado.")
    return df


# =========================
# 3C. OUTLIERS TÉRMINOS
# =========================

def eda_capa2_outliers_terminos() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_mensual.csv"
    df = pd.read_csv(input_path)

    outlier_df = _iqr_outlier_summary(
        df=df,
        cols=["valor_trends"],
        bloque="terminos",
    )

    outlier_df.to_csv(TABLES_CAPA2_EDA / "capa2_outliers_terminos_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.boxplot(pd.to_numeric(df["valor_trends"], errors="coerce").dropna())
    plt.title("Boxplot - valor_trends (términos)")
    plt.ylabel("valor_trends")
    _save_plot("capa2_outliers_boxplot_terminos.png")

    print("EDA outliers términos completado.")
    print(outlier_df)

    return outlier_df


# =========================
# 4. EDA PRODUCTOS
# =========================

def eda_capa2_productos() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_productos_mensual.csv"
    df = pd.read_csv(input_path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    descriptivos = df.describe(include="all")
    descriptivos.to_csv(TABLES_CAPA2_EDA / "capa2_productos_descriptivos.csv")

    numeric_desc = df[["valor_trends"]].describe().T
    numeric_desc.to_csv(TABLES_CAPA2_EDA / "capa2_productos_numeric_descriptives.csv")

    ranking_marcas = (
        df.groupby("marca", dropna=False)["valor_trends"]
        .mean()
        .reset_index()
        .rename(columns={"valor_trends": "valor_trends_medio"})
        .sort_values("valor_trends_medio", ascending=False)
    )
    ranking_marcas.to_csv(TABLES_CAPA2_EDA / "capa2_productos_ranking_marcas.csv", index=False)

    ranking_categorias = (
        df.groupby("categoria_producto", dropna=False)["valor_trends"]
        .mean()
        .reset_index()
        .rename(columns={"valor_trends": "valor_trends_medio"})
        .sort_values("valor_trends_medio", ascending=False)
    )
    ranking_categorias.to_csv(TABLES_CAPA2_EDA / "capa2_productos_ranking_categorias.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(ranking_marcas["marca"], ranking_marcas["valor_trends_medio"])
    plt.title("Ranking de marcas (productos) por interés medio")
    plt.ylabel("Valor medio Trends")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa2_productos_ranking_marcas.png")

    plt.figure(figsize=(8, 5))
    plt.bar(ranking_categorias["categoria_producto"], ranking_categorias["valor_trends_medio"])
    plt.title("Ranking de categorías de producto por interés medio")
    plt.ylabel("Valor medio Trends")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa2_productos_ranking_categorias.png")

    evol_marcas = (
        df.groupby(["fecha", "marca"], dropna=False)["valor_trends"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    for marca in sorted(evol_marcas["marca"].dropna().unique()):
        subset = evol_marcas[evol_marcas["marca"] == marca]
        plt.plot(subset["fecha"], subset["valor_trends"], label=marca)
    plt.title("Evolución mensual del interés por productos según marca")
    plt.xlabel("Fecha")
    plt.ylabel("Valor medio Trends")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("capa2_productos_evolucion_por_marca.png")

    print("EDA productos completado.")
    return df


# =========================
# 5. EDA EVENTOS
# =========================

def eda_capa2_eventos() -> tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()

    eventos_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos.csv"
    eventos_mensual_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos_mensual.csv"

    df = pd.read_csv(eventos_path)
    monthly = pd.read_csv(eventos_mensual_path)

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    monthly["fecha"] = pd.to_datetime(monthly["fecha"], errors="coerce")

    resumen_anual = (
        df.groupby("anio", dropna=False)
        .size()
        .reset_index(name="n_eventos")
        .sort_values("anio")
    )
    resumen_anual.to_csv(TABLES_CAPA2_EDA / "capa2_eventos_resumen_anual.csv", index=False)

    resumen_plataforma = (
        df.groupby("plataforma_std", dropna=False)
        .size()
        .reset_index(name="n_eventos")
        .sort_values("n_eventos", ascending=False)
    )
    resumen_plataforma.to_csv(TABLES_CAPA2_EDA / "capa2_eventos_resumen_plataforma.csv", index=False)

    resumen_categoria = (
        df.groupby("categoria_evento", dropna=False)
        .size()
        .reset_index(name="n_eventos")
        .sort_values("n_eventos", ascending=False)
    )
    resumen_categoria.to_csv(TABLES_CAPA2_EDA / "capa2_eventos_resumen_categoria.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(resumen_anual["anio"].astype(str), resumen_anual["n_eventos"])
    plt.title("Número de hitos por año")
    plt.ylabel("Nº eventos")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa2_eventos_por_anio.png")

    plt.figure(figsize=(8, 5))
    plt.bar(resumen_plataforma["plataforma_std"], resumen_plataforma["n_eventos"])
    plt.title("Distribución de hitos por plataforma")
    plt.ylabel("Nº eventos")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa2_eventos_por_plataforma.png")

    plt.figure(figsize=(8, 5))
    plt.bar(resumen_categoria["categoria_evento"], resumen_categoria["n_eventos"])
    plt.title("Distribución de hitos por categoría")
    plt.ylabel("Nº eventos")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa2_eventos_por_categoria.png")

    plt.figure(figsize=(12, 5))
    plt.plot(monthly["fecha"], monthly["n_eventos_total"], marker="o")
    plt.title("Densidad mensual de hitos")
    plt.xlabel("Fecha")
    plt.ylabel("Nº eventos")
    plt.grid(True, alpha=0.3)
    _save_plot("capa2_eventos_densidad_mensual.png")

    print("EDA eventos completado.")
    return df, monthly


# =========================
# 6. EDA INTEGRATED
# =========================

def eda_capa2_integrated() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_integrated.csv"
    df = pd.read_csv(input_path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    descriptivos = df.describe(include="all")
    descriptivos.to_csv(TABLES_CAPA2_EDA / "capa2_integrated_descriptivos.csv")

    top_event_months = (
        df.groupby("fecha", dropna=False)[["n_eventos_total", "n_plataformas", "n_tipos_evento"]]
        .max()
        .reset_index()
        .sort_values("n_eventos_total", ascending=False)
    )
    top_event_months.to_csv(TABLES_CAPA2_EDA / "capa2_integrated_top_event_months.csv", index=False)

    monthly_trends = (
        df.groupby("fecha", dropna=False)
        .agg(
            valor_trends_medio=("valor_trends", "mean"),
            n_eventos_total=("n_eventos_total", "max"),
            n_plataformas=("n_plataformas", "max"),
            n_tipos_evento=("n_tipos_evento", "max"),
        )
        .reset_index()
    )
    monthly_trends.to_csv(TABLES_CAPA2_EDA / "capa2_integrated_monthly_summary.csv", index=False)

    monthly_trends["valor_trends_medio_norm"] = _normalize_series_minmax(monthly_trends["valor_trends_medio"])
    monthly_trends["n_eventos_total_norm"] = _normalize_series_minmax(monthly_trends["n_eventos_total"])

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_trends["fecha"], monthly_trends["valor_trends_medio_norm"], label="Valor medio trends (norm.)")
    plt.plot(monthly_trends["fecha"], monthly_trends["n_eventos_total_norm"], label="Nº eventos mensuales (norm.)")
    plt.title("Comparación normalizada entre interés medio mensual y densidad de hitos")
    plt.xlabel("Fecha")
    plt.ylabel("Escala normalizada 0-1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("capa2_integrated_trends_vs_eventos.png")

    print("EDA integrated completado.")
    return df


# =========================
# 7. EDA DECISIONES DE CALIDAD
# =========================

def eda_capa2_term_quality_decisions() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "integrated" / "capa2_term_quality_decisions.csv"
    df = pd.read_csv(input_path)

    resumen_prioridad = (
        df.groupby("prioridad_analitica", dropna=False)
        .size()
        .reset_index(name="n_terminos")
        .sort_values("n_terminos", ascending=False)
    )
    resumen_prioridad.to_csv(TABLES_CAPA2_CONTROL / "capa2_resumen_prioridad_analitica.csv", index=False)

    resumen_quality = (
        df.groupby("quality_flag", dropna=False)
        .size()
        .reset_index(name="n_terminos")
        .sort_values("n_terminos", ascending=False)
    )
    resumen_quality.to_csv(TABLES_CAPA2_CONTROL / "capa2_resumen_quality_flag.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(resumen_prioridad["prioridad_analitica"], resumen_prioridad["n_terminos"])
    plt.title("Número de términos por prioridad analítica")
    plt.ylabel("Nº términos")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa2_resumen_prioridad_analitica.png")

    plt.figure(figsize=(7, 4))
    plt.bar(resumen_quality["quality_flag"], resumen_quality["n_terminos"])
    plt.title("Número de términos por quality_flag")
    plt.ylabel("Nº términos")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa2_resumen_quality_flag.png")

    print("EDA decisiones de calidad completado.")
    return df


# =========================
# 8. COBERTURA TEMPORAL FUENTES
# =========================

def eda_capa2_source_coverage() -> pd.DataFrame:
    _ensure_dirs()

    source_map = {
        "googletrends_marcas": PROCESSED_CAPA2 / "googletrends" / "trends_marcas_clean.csv",
        "googletrends_productos": PROCESSED_CAPA2 / "googletrends" / "trends_productos_clean.csv",
        "googletrends_terminos": PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_mensual.csv",
        "eventos": PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos_mensual.csv",
        "instagram_posts": PROCESSED_CAPA2 / "apify" / "instagram_posts_clean.csv",
        "instagram_brand_monthly": PROCESSED_CAPA2 / "apify" / "instagram_brand_monthly.csv",
        "brand_digital": PROCESSED_CAPA2 / "integrated" / "capa2_master_brand_digital.csv",
    }

    records = []

    for source_name, path in source_map.items():
        df = _safe_read_csv(path)
        if df is None:
            continue

        fecha_col = "fecha_post" if "fecha_post" in df.columns else "fecha"
        fechas = pd.to_datetime(df[fecha_col], errors="coerce")

        records.append(
            {
                "fuente": source_name,
                "fecha_inicio": str(fechas.min().date()) if fechas.notna().any() else None,
                "fecha_fin": str(fechas.max().date()) if fechas.notna().any() else None,
                "n_filas": int(df.shape[0]),
                "n_columnas": int(df.shape[1]),
            }
        )

    coverage_df = pd.DataFrame(records).sort_values("fuente")
    coverage_df.to_csv(TABLES_CAPA2_CONTROL / "capa2_source_coverage.csv", index=False)

    use_map = {
        "googletrends_marcas": "historico_principal",
        "googletrends_productos": "historico_complementario",
        "googletrends_terminos": "historico_principal",
        "eventos": "contexto",
        "instagram_posts": "social_reciente",
        "instagram_brand_monthly": "social_reciente_agregado",
        "brand_digital": "integracion_marca_ventana_comun",
    }

    coverage_df["uso_analitico"] = coverage_df["fuente"].map(use_map)
    coverage_df.to_csv(TABLES_CAPA2_CONTROL / "capa2_source_coverage_interpreted.csv", index=False)

    print("Cobertura temporal de fuentes calculada.")
    print(coverage_df)

    return coverage_df


# =========================
# 9. EDA SOCIAL
# =========================

def eda_capa2_social() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_social.csv"
    df = pd.read_csv(input_path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    descriptivos = df.describe(include="all")
    descriptivos.to_csv(TABLES_CAPA2_EDA / "capa2_social_descriptivos.csv")

    social_numeric_cols = [
        "n_posts",
        "likes_totales",
        "comentarios_totales",
        "engagement_total",
        "likes_medios_post",
        "comentarios_medios_post",
        "engagement_medio_post",
    ]
    numeric_desc = df[social_numeric_cols].describe().T
    numeric_desc.to_csv(TABLES_CAPA2_EDA / "capa2_social_numeric_descriptives.csv")

    ranking_posts = (
        df.groupby("marca", dropna=False)["n_posts"]
        .sum()
        .reset_index()
        .sort_values("n_posts", ascending=False)
    )
    ranking_posts.to_csv(TABLES_CAPA2_EDA / "capa2_social_ranking_posts.csv", index=False)

    ranking_engagement = (
        df.groupby("marca", dropna=False)["engagement_total"]
        .sum()
        .reset_index()
        .sort_values("engagement_total", ascending=False)
    )
    ranking_engagement.to_csv(TABLES_CAPA2_EDA / "capa2_social_ranking_engagement.csv", index=False)

    ranking_engagement_medio = (
        df.groupby("marca", dropna=False)["engagement_medio_post"]
        .mean()
        .reset_index()
        .sort_values("engagement_medio_post", ascending=False)
    )
    ranking_engagement_medio.to_csv(TABLES_CAPA2_EDA / "capa2_social_ranking_engagement_medio.csv", index=False)

    ranking_likes_medios = (
        df.groupby("marca", dropna=False)["likes_medios_post"]
        .mean()
        .reset_index()
        .sort_values("likes_medios_post", ascending=False)
    )
    ranking_likes_medios.to_csv(TABLES_CAPA2_EDA / "capa2_social_ranking_likes_medios.csv", index=False)

    ranking_comments_medios = (
        df.groupby("marca", dropna=False)["comentarios_medios_post"]
        .mean()
        .reset_index()
        .sort_values("comentarios_medios_post", ascending=False)
    )
    ranking_comments_medios.to_csv(TABLES_CAPA2_EDA / "capa2_social_ranking_comentarios_medios.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(ranking_posts["marca"], ranking_posts["n_posts"])
    plt.title("Volumen total de posts por marca")
    plt.ylabel("Nº posts")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa2_social_ranking_posts.png")

    plt.figure(figsize=(8, 5))
    plt.bar(ranking_engagement["marca"], ranking_engagement["engagement_total"])
    plt.title("Engagement total por marca")
    plt.ylabel("Engagement total")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa2_social_ranking_engagement.png")

    plt.figure(figsize=(8, 5))
    plt.bar(ranking_engagement_medio["marca"], ranking_engagement_medio["engagement_medio_post"])
    plt.title("Engagement medio por post y marca")
    plt.ylabel("Engagement medio por post")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa2_social_ranking_engagement_medio.png")

    plt.figure(figsize=(12, 6))
    for marca in sorted(df["marca"].dropna().unique()):
        subset = df[df["marca"] == marca]
        plt.plot(subset["fecha"], subset["engagement_total"], marker="o", label=marca)
    plt.title("Evolución mensual del engagement por marca")
    plt.xlabel("Fecha")
    plt.ylabel("Engagement total")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("capa2_social_evolucion_engagement.png")

    print("EDA social completado.")
    return df


# =========================
# 9B. OUTLIERS SOCIAL
# =========================

def eda_capa2_outliers_social() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_social.csv"
    df = pd.read_csv(input_path)

    social_numeric_cols = [
        "n_posts",
        "likes_totales",
        "comentarios_totales",
        "engagement_total",
        "likes_medios_post",
        "comentarios_medios_post",
        "engagement_medio_post",
    ]

    outlier_df = _iqr_outlier_summary(
        df=df,
        cols=social_numeric_cols,
        bloque="social",
    )

    outlier_df.to_csv(TABLES_CAPA2_EDA / "capa2_outliers_social_summary.csv", index=False)

    for col in social_numeric_cols:
        if col not in df.columns:
            continue
        plt.figure(figsize=(8, 5))
        plt.boxplot(pd.to_numeric(df[col], errors="coerce").dropna())
        plt.title(f"Boxplot - {col} (social)")
        plt.ylabel(col)
        _save_plot(f"capa2_outliers_boxplot_social_{col}.png")

    print("EDA outliers social completado.")
    print(outlier_df)

    return outlier_df


# =========================
# 10. VENTANA COMÚN POR MARCA
# =========================

def eda_capa2_brand_common_window() -> pd.DataFrame:
    _ensure_dirs()

    trends_path = PROCESSED_CAPA2 / "googletrends" / "trends_marcas_clean.csv"
    social_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_social.csv"

    trends = pd.read_csv(trends_path)
    social = pd.read_csv(social_path)

    trends["fecha"] = pd.to_datetime(trends["fecha"], errors="coerce")
    social["fecha"] = pd.to_datetime(social["fecha"], errors="coerce")

    trends = trends.rename(columns={"termino": "marca"})
    trends["marca"] = trends["marca"].astype(str).str.strip().str.lower()
    social["marca"] = social["marca"].astype(str).str.strip().str.lower()

    brands = sorted(set(trends["marca"].dropna().unique()).union(set(social["marca"].dropna().unique())))

    records = []
    for marca in brands:
        t = trends[trends["marca"] == marca]
        s = social[social["marca"] == marca]

        t_min = t["fecha"].min() if not t.empty else pd.NaT
        t_max = t["fecha"].max() if not t.empty else pd.NaT
        s_min = s["fecha"].min() if not s.empty else pd.NaT
        s_max = s["fecha"].max() if not s.empty else pd.NaT

        common_start = max(t_min, s_min) if pd.notna(t_min) and pd.notna(s_min) else pd.NaT
        common_end = min(t_max, s_max) if pd.notna(t_max) and pd.notna(s_max) else pd.NaT

        n_common_months = None
        if pd.notna(common_start) and pd.notna(common_end) and common_start <= common_end:
            n_common_months = len(pd.period_range(common_start, common_end, freq="M"))
        else:
            common_start = pd.NaT
            common_end = pd.NaT

        records.append(
            {
                "marca": marca,
                "fecha_inicio_trends": str(t_min.date()) if pd.notna(t_min) else None,
                "fecha_fin_trends": str(t_max.date()) if pd.notna(t_max) else None,
                "fecha_inicio_social": str(s_min.date()) if pd.notna(s_min) else None,
                "fecha_fin_social": str(s_max.date()) if pd.notna(s_max) else None,
                "fecha_inicio_comun": str(common_start.date()) if pd.notna(common_start) else None,
                "fecha_fin_comun": str(common_end.date()) if pd.notna(common_end) else None,
                "n_meses_comunes": n_common_months,
            }
        )

    window_df = pd.DataFrame(records).sort_values("marca")
    window_df.to_csv(TABLES_CAPA2_EDA / "capa2_brand_common_window.csv", index=False)

    print("Ventana común por marca calculada.")
    print(window_df)

    return window_df


# =========================
# 10B. BRAND DIGITAL EN VENTANA COMÚN
# =========================

def eda_capa2_brand_digital_common_window() -> pd.DataFrame:
    _ensure_dirs()

    brand_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_brand_digital.csv"
    window_path = TABLES_CAPA2_EDA / "capa2_brand_common_window.csv"

    df = pd.read_csv(brand_path)
    window_df = pd.read_csv(window_path)

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["marca"] = df["marca"].astype(str).str.strip().str.lower()

    window_df["fecha_inicio_comun"] = pd.to_datetime(window_df["fecha_inicio_comun"], errors="coerce")
    window_df["fecha_fin_comun"] = pd.to_datetime(window_df["fecha_fin_comun"], errors="coerce")

    def _brand_window_comment(row: pd.Series) -> str:
        if pd.isna(row["n_meses_comunes"]):
            return "Sin solapamiento entre Google Trends e Instagram."
        if row["marca"] == "shein" and row["n_meses_comunes"] <= 1:
            return "Comparabilidad muy limitada; excluir del análisis comparativo fuerte en ventana común."
        if row["n_meses_comunes"] < 6:
            return "Ventana común reducida; interpretar con cautela."
        return "Ventana común suficiente para comparación descriptiva."

    window_df["comentario_metodologico"] = window_df.apply(_brand_window_comment, axis=1)
    window_df.to_csv(TABLES_CAPA2_EDA / "capa2_brand_common_window.csv", index=False)

    valid_windows = window_df[
        window_df["fecha_inicio_comun"].notna() & window_df["fecha_fin_comun"].notna()
    ].copy()

    valid_windows = valid_windows[valid_windows["marca"] != "shein"].copy()

    summary_records = []
    records = []

    for _, row in valid_windows.iterrows():
        marca = str(row["marca"]).strip().lower()
        start = row["fecha_inicio_comun"]
        end = row["fecha_fin_comun"]

        subset = df[
            (df["marca"] == marca) &
            (df["fecha"] >= start) &
            (df["fecha"] <= end)
        ].copy()

        if subset.empty:
            continue

        summary_records.append(
            {
                "marca": marca,
                "fecha_inicio_comun": start.date(),
                "fecha_fin_comun": end.date(),
                "n_meses": len(subset),
                "avg_valor_trends": subset["valor_trends"].mean(),
                "avg_n_posts": subset["n_posts"].mean(),
                "avg_engagement_total": subset["engagement_total"].mean(),
            }
        )

        subset["valor_trends_norm"] = _normalize_series_minmax(subset["valor_trends"])
        subset["n_posts_norm"] = _normalize_series_minmax(subset["n_posts"])
        subset["engagement_total_norm"] = _normalize_series_minmax(subset["engagement_total"])

        export_cols = [
            "fecha",
            "marca",
            "valor_trends",
            "n_posts",
            "engagement_total",
            "valor_trends_norm",
            "n_posts_norm",
            "engagement_total_norm",
        ]
        records.append(subset[export_cols].copy())

        plt.figure(figsize=(10, 5))
        plt.plot(subset["fecha"], subset["valor_trends_norm"], marker="o", label="Valor Trends (norm.)")
        plt.plot(subset["fecha"], subset["n_posts_norm"], marker="o", label="Nº posts (norm.)")
        plt.plot(subset["fecha"], subset["engagement_total_norm"], marker="o", label="Engagement total (norm.)")
        plt.title(f"Marca digital integrada - {marca} (ventana común)")
        plt.xlabel("Fecha")
        plt.ylabel("Escala normalizada 0-1")
        plt.legend()
        plt.grid(True, alpha=0.3)
        _save_plot(f"capa2_brand_digital_{marca}_common_window_normalized.png")

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(TABLES_CAPA2_EDA / "capa2_brand_digital_common_window_summary.csv", index=False)

    common_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    common_df.to_csv(TABLES_CAPA2_EDA / "capa2_brand_digital_common_window.csv", index=False)

    print("EDA brand digital en ventana común completado.")
    return common_df


# =========================
# 11. EDA BRAND DIGITAL
# =========================

def eda_capa2_brand_digital() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_brand_digital.csv"
    df = pd.read_csv(input_path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["marca"] = df["marca"].astype(str).str.strip().str.lower()

    descriptivos = df.describe(include="all")
    descriptivos.to_csv(TABLES_CAPA2_EDA / "capa2_brand_digital_descriptivos.csv")

    brand_numeric_cols = [
        "valor_trends",
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
    brand_numeric_cols = [c for c in brand_numeric_cols if c in df.columns]
    numeric_desc = df[brand_numeric_cols].describe().T
    numeric_desc.to_csv(TABLES_CAPA2_EDA / "capa2_brand_digital_numeric_descriptives.csv")

    recent_df = df[df["n_posts"] > 0].copy()
    recent_df.to_csv(TABLES_CAPA2_EDA / "capa2_brand_digital_recent_window.csv", index=False)

    corr_cols = [
        "valor_trends",
        "n_posts",
        "likes_totales",
        "comentarios_totales",
        "engagement_total",
        "likes_medios_post",
        "comentarios_medios_post",
        "engagement_medio_post",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]

    corr_global = df[corr_cols].corr()
    corr_global.to_csv(TABLES_CAPA2_EDA / "capa2_brand_digital_correlation.csv")
    _plot_correlation_matrix(
        corr_global,
        "Matriz de correlación - brand digital (global)",
        "capa2_brand_digital_correlation_heatmap.png",
    )

    if not recent_df.empty:
        corr_recent = recent_df[corr_cols].corr()
        corr_recent.to_csv(TABLES_CAPA2_EDA / "capa2_brand_digital_correlation_social_window.csv")
        _plot_correlation_matrix(
            corr_recent,
            "Matriz de correlación - brand digital (ventana social)",
            "capa2_brand_digital_correlation_social_window_heatmap.png",
        )

    monthly_brand = (
        df.groupby(["fecha", "marca"], dropna=False)
        .agg(
            valor_trends=("valor_trends", "mean"),
            n_posts=("n_posts", "sum"),
            engagement_total=("engagement_total", "sum"),
        )
        .reset_index()
    )
    monthly_brand.to_csv(TABLES_CAPA2_EDA / "capa2_brand_digital_monthly_summary.csv", index=False)

    brands = sorted(monthly_brand["marca"].dropna().unique())

    for marca in brands:
        subset = monthly_brand[monthly_brand["marca"] == marca].copy()

        plt.figure(figsize=(12, 6))
        plt.plot(subset["fecha"], subset["valor_trends"], label="Valor Trends")
        plt.plot(subset["fecha"], subset["n_posts"], label="Nº posts")
        plt.plot(subset["fecha"], subset["engagement_total"], label="Engagement total")
        plt.title(f"Marca digital integrada - {marca}")
        plt.xlabel("Fecha")
        plt.ylabel("Magnitud")
        plt.legend()
        plt.grid(True, alpha=0.3)
        _save_plot(f"capa2_brand_digital_{marca}.png")

        subset["valor_trends_norm"] = _normalize_series_minmax(subset["valor_trends"])
        subset["n_posts_norm"] = _normalize_series_minmax(subset["n_posts"])
        subset["engagement_total_norm"] = _normalize_series_minmax(subset["engagement_total"])

        plt.figure(figsize=(12, 6))
        plt.plot(subset["fecha"], subset["valor_trends_norm"], label="Valor Trends (norm.)")
        plt.plot(subset["fecha"], subset["n_posts_norm"], label="Nº posts (norm.)")
        plt.plot(subset["fecha"], subset["engagement_total_norm"], label="Engagement total (norm.)")
        plt.title(f"Marca digital integrada - {marca} (series normalizadas)")
        plt.xlabel("Fecha")
        plt.ylabel("Escala normalizada 0-1")
        plt.legend()
        plt.grid(True, alpha=0.3)
        _save_plot(f"capa2_brand_digital_{marca}_normalized.png")

    print("EDA brand digital completado.")
    return df


# =========================
# 11B. OUTLIERS BRAND DIGITAL
# =========================

def eda_capa2_outliers_brand_digital() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_brand_digital.csv"
    df = pd.read_csv(input_path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    cols = [
        "valor_trends",
        "n_posts",
        "likes_totales",
        "comentarios_totales",
        "engagement_total",
        "engagement_medio_post",
    ]

    records = []

    for col in cols:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        n_outliers = int(((s < lower) | (s > upper)).sum())
        pct_outliers = round((n_outliers / len(s)) * 100, 2) if len(s) > 0 else 0.0

        records.append(
            {
                "bloque": "brand_digital_common_window",
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
                "decision_metodologica": "mantener_y_documentar_en_ventana_comun",
            }
        )

    summary_df = pd.DataFrame(records)
    out_path = TABLES_CAPA2_EDA / "capa2_outliers_brand_digital_summary.csv"
    summary_df.to_csv(out_path, index=False)

    print("EDA outliers brand digital completado.")
    print(summary_df)

    return summary_df


# =========================
# JUSTIFICACION ANALITICA
# =========================

def eda_capa2_justificacion_analitica() -> pd.DataFrame:
    """
    Documenta y exporta la justificación del tipo de análisis descriptivo
    aplicado a cada dataset de capa 2.
    """
    _ensure_dirs()

    records = [
        {
            "dataset": "capa2_master_terminos_mensual",
            "tipo_dato": "series_temporales_multivariadas",
            "frecuencia": "mensual",
            "periodo": "2015-2025 (serie mensual, con longitud variable según término y cobertura)",
            "tipo_analisis_elegido": "analisis_series_temporales_con_segmentacion",
            "justificacion": (
                "Google Trends devuelve series mensuales de interes relativo (0-100) para cada termino. "
                "El analisis temporal permite detectar emergencia, auge y decaimiento de terminos. "
                "Se aplican descriptivos por termino y grupo, rankings por volumen medio, evolucion temporal, "
                "heatmaps y comparaciones entre familias analiticas. "
                "El valor_trends=0 se mantiene documentado porque en Google Trends significa volumen insuficiente "
                "para ser representado en la escala, no ausencia real de interes."
            ),
            "alternativa_descartada": (
                "Analisis transversal: perderia la dimension temporal, que es el nucleo del fenomeno estudiado."
            ),
        },
        {
            "dataset": "capa2_master_terminos_main",
            "tipo_dato": "series_temporales_filtradas_por_calidad",
            "frecuencia": "mensual",
            "periodo": "2015-2025 (subset principal de términos)",
            "tipo_analisis_elegido": "analisis_tendencia_con_control_calidad",
            "justificacion": (
                "Subconjunto del master de terminos restringido a terminos con suficiente robustez analitica. "
                "Permite realizar analisis de tendencias con menor ruido y mayor continuidad temporal."
            ),
            "alternativa_descartada": (
                "Usar todos los terminos sin filtro introduciria ruido por señales demasiado débiles o discontinuas."
            ),
        },
        {
            "dataset": "capa2_master_eventos",
            "tipo_dato": "datos_categoricos_con_fecha",
            "frecuencia": "irregular",
            "periodo": "2015-2025",
            "tipo_analisis_elegido": "analisis_frecuencias_y_distribucion_temporal",
            "justificacion": (
                "Los eventos son hitos discretos no regulares. Tiene sentido analizarlos por frecuencia, "
                "distribucion temporal, plataforma y categoria, y utilizarlos como contexto interpretativo "
                "para los picos de interes digital."
            ),
            "alternativa_descartada": (
                "Serie temporal clasica: no procede porque no existe una frecuencia regular intrinseca del dato."
            ),
        },
        {
            "dataset": "capa2_master_social",
            "tipo_dato": "panel_mensual_por_marca",
            "frecuencia": "mensual",
            "periodo": "ventana reciente disponible por marca",
            "tipo_analisis_elegido": "analisis_panel_engagement_por_marca",
            "justificacion": (
                "Los datos agregados de Instagram permiten estudiar volumen de actividad y engagement por marca "
                "a nivel mensual. Se aplican descriptivos, rankings y evolucion temporal del engagement. "
                "Los posts con metricas_imputadas=True pueden subestimar el engagement real y por ello se documentan."
            ),
            "alternativa_descartada": (
                "Analisis exclusivo a nivel de post: excesivamente granular para identificar patrones temporales por marca."
            ),
        },
        {
            "dataset": "capa2_master_brand_digital",
            "tipo_dato": "master_integrado_marca_mes",
            "frecuencia": "mensual",
            "periodo": "ventana comun entre Google Trends e Instagram",
            "tipo_analisis_elegido": "analisis_conjunto_trends_social_eventos",
            "justificacion": (
                "Este master integra interes digital, actividad social y contexto de eventos a nivel marca-mes. "
                "Permite observar si los picos de Google Trends coinciden con picos de engagement y con hitos "
                "del sector, siempre dentro de una ventana temporal comparable."
            ),
            "alternativa_descartada": (
                "Analisis aislado por fuente: impediria estudiar la relacion entre señales digitales complementarias."
            ),
        },
    ]

    df = pd.DataFrame(records)
    out_path = TABLES_CAPA2_CONTROL / "capa2_justificacion_tipo_analisis.csv"
    df.to_csv(out_path, index=False)
    print(f"  Justificación tipo análisis capa2: {len(df)} datasets → {out_path.name}")
    return df


def eda_capa2_source_coverage_v2() -> pd.DataFrame:
    """
    Cobertura temporal y nota metodológica de cada fuente de capa 2.
    """
    _ensure_dirs()

    source_map = {
        "googletrends_marcas": PROCESSED_CAPA2 / "googletrends" / "trends_marcas_clean.csv",
        "googletrends_productos": PROCESSED_CAPA2 / "googletrends" / "trends_productos_clean.csv",
        "googletrends_terminos": PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_mensual.csv",
        "eventos": PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos_mensual.csv",
        "instagram_posts": PROCESSED_CAPA2 / "apify" / "instagram_posts_clean.csv",
        "instagram_brand_monthly": PROCESSED_CAPA2 / "apify" / "instagram_brand_monthly.csv",
        "brand_digital": PROCESSED_CAPA2 / "integrated" / "capa2_master_brand_digital.csv",
    }

    notas = {
        "googletrends_marcas": (
            "Series mensuales de interes relativo para marcas. Google Trends usa escala 0-100 relativa al maximo del termino. "
            "El valor 0 indica volumen insuficiente para la escala, no interes nulo."
        ),
        "googletrends_productos": (
            "Series mensuales por categoria de producto y marca. Permiten comparar interes relativo por tipo de producto."
        ),
        "googletrends_terminos": (
            "Master unificado de terminos de marcas, esteticas y comportamientos. Base del analisis de tendencias digitales."
        ),
        "eventos": (
            "Dataset contextual de hitos del sector moda, utilizado para interpretar picos temporales y cambios de señal."
        ),
        "instagram_posts": (
            "Posts scrapeados de perfiles oficiales. No recoge menciones externas ni UGC. Algunas metricas pueden haber sido imputadas a 0."
        ),
        "instagram_brand_monthly": (
            "Agregacion mensual por marca de actividad y engagement en Instagram."
        ),
        "brand_digital": (
            "Master integrado marca-mes con Google Trends, señal social y contexto de eventos en ventana temporal comparable."
        ),
    }

    records = []

    for source_name, path in source_map.items():
        df = _safe_read_csv(path)
        if df is None:
            continue

        fecha_col = "fecha_post" if "fecha_post" in df.columns else "fecha"
        fechas = pd.to_datetime(df.get(fecha_col, pd.Series(dtype="object")), errors="coerce")

        start = str(fechas.min().date()) if fechas.notna().any() else "n/a"
        end = str(fechas.max().date()) if fechas.notna().any() else "n/a"

        records.append({
            "fuente": source_name,
            "fecha_inicio": start,
            "fecha_fin": end,
            "n_filas": len(df),
            "n_columnas": df.shape[1],
            "nota_metodologica": notas.get(source_name, ""),
        })

    coverage_df = pd.DataFrame(records)
    out_path = TABLES_CAPA2_CONTROL / "capa2_source_coverage.csv"
    coverage_df.to_csv(out_path, index=False)
    print(f"  Source coverage capa2: {len(coverage_df)} fuentes documentadas → {out_path.name}")
    return coverage_df


# =========================
# RUN ALL
# =========================

def run_all_eda() -> None:
    profile_capa2()
    analyze_nulls_capa2()
    eda_capa2_justificacion_analitica()
    eda_capa2_terminos()
    eda_capa2_terminos_main()
    eda_capa2_outliers_terminos()
    eda_capa2_productos()
    eda_capa2_eventos()
    eda_capa2_integrated()
    eda_capa2_term_quality_decisions()
    eda_capa2_source_coverage_v2()
    eda_capa2_social()
    eda_capa2_outliers_social()
    eda_capa2_brand_common_window()
    eda_capa2_brand_digital_common_window()
    eda_capa2_brand_digital()
    eda_capa2_outliers_brand_digital()
    print("Todo el EDA de capa 2 completado.")


if __name__ == "__main__":
    run_all_eda()