import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import (
    FIGURES_CAPA3_EDA,
    PROCESSED_CAPA3,
    TABLES_CAPA3_CONTROL,
    TABLES_CAPA3_EDA,
)


# =========================
# HELPERS
# =========================

def _ensure_dirs() -> None:
    TABLES_CAPA3_CONTROL.mkdir(parents=True, exist_ok=True)
    TABLES_CAPA3_EDA.mkdir(parents=True, exist_ok=True)
    FIGURES_CAPA3_EDA.mkdir(parents=True, exist_ok=True)


def _save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_CAPA3_EDA / filename, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_corr(corr_df: pd.DataFrame, title: str, filename: str) -> None:
    plt.figure(figsize=(9, 7))
    plt.imshow(corr_df, aspect="auto")
    plt.colorbar(label="Correlación")
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr_df.index)), corr_df.index)
    plt.title(title)
    _save_plot(filename)


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# =========================
# 1. PROFILE
# =========================

def profile_capa3() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()

    datasets = {
        "encuesta_fastfashion_clean": PROCESSED_CAPA3 / "survey" / "encuesta_fastfashion_clean.csv",
        "encuesta_rrss_long": PROCESSED_CAPA3 / "survey" / "encuesta_rrss_long.csv",
        "encuesta_marcas_vistas_long": PROCESSED_CAPA3 / "survey" / "encuesta_marcas_vistas_long.csv",
        "encuesta_marcas_influyen_long": PROCESSED_CAPA3 / "survey" / "encuesta_marcas_influyen_long.csv",
        "capa3_master_encuesta": PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv",
        "capa3_clustering_ready": PROCESSED_CAPA3 / "integrated" / "capa3_clustering_ready.csv",
        "capa3_supervised_ready": PROCESSED_CAPA3 / "integrated" / "capa3_supervised_ready.csv",
        "capa3_generacion_summary": PROCESSED_CAPA3 / "integrated" / "capa3_generacion_summary.csv",
        "capa3_sample_structure": PROCESSED_CAPA3 / "integrated" / "capa3_sample_structure.csv",
        "capa3_target_summary": PROCESSED_CAPA3 / "integrated" / "capa3_target_summary.csv",
        "capa3_dataset_quality_summary": TABLES_CAPA3_CONTROL / "capa3_dataset_quality_summary.csv",
    }

    summary_records = []
    nulls_records = []
    numeric_records = []

    for name, path in datasets.items():
        df = pd.read_csv(path)

        summary_records.append(
            {
                "dataset": name,
                "rows": df.shape[0],
                "cols": df.shape[1],
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

    summary_df.to_csv(TABLES_CAPA3_CONTROL / "capa3_profile_summary.csv", index=False)
    nulls_df.to_csv(TABLES_CAPA3_CONTROL / "capa3_profile_nulls.csv", index=False)
    numeric_df.to_csv(TABLES_CAPA3_CONTROL / "capa3_profile_numeric.csv", index=False)

    print("Profiling capa 3 completado.")
    print(summary_df)

    return summary_df, nulls_df, numeric_df

# =========================
# 2. NULOS
# =========================

def analyze_nulls_capa3() -> pd.DataFrame:
    _ensure_dirs()

    df = pd.read_csv(TABLES_CAPA3_CONTROL / "capa3_profile_nulls.csv")
    detailed = df[df["pct_nulls"] > 0].copy().sort_values(
        ["dataset", "pct_nulls"], ascending=[True, False]
    )

    detailed.to_csv(TABLES_CAPA3_CONTROL / "capa3_nulls_detailed.csv", index=False)

    def assign_null_decision(row: pd.Series) -> str:
        variable = str(row["variable"])
        pct = row["pct_nulls"]

        if variable == "timestamp":
            return "mantener_solo_control"
        if variable in {
            "rrss_habituales",
            "marcas_ve_frecuencia_rrss",
            "marcas_que_mas_influyen_compra_rrss",
        }:
            return "mantener_por_trazabilidad"
        if pct >= 50:
            return "revisar"
        if pct >= 20:
            return "mantener_con_cautela"
        return "mantener"

    def assign_null_comment(row: pd.Series) -> str:
        variable = str(row["variable"])
        pct = row["pct_nulls"]

        if variable == "timestamp":
            return "Variable útil para control de captura, no esencial para modelización."
        if variable in {
            "rrss_habituales",
            "marcas_ve_frecuencia_rrss",
            "marcas_que_mas_influyen_compra_rrss",
        }:
            return "Campo multirrespuesta conservado en tablas long; el master puede tener nulos sin comprometer análisis principal."
        if pct >= 50:
            return "Nivel de nulos elevado; conviene revisar su uso analítico."
        if pct >= 20:
            return "Variable utilizable con cautela."
        return "Nivel de nulos asumible."

    decision_df = detailed.copy()
    decision_df["decision_analitica"] = decision_df.apply(assign_null_decision, axis=1)
    decision_df["comentario"] = decision_df.apply(assign_null_comment, axis=1)

    decision_df.to_csv(
        TABLES_CAPA3_CONTROL / "capa3_nulls_decision_matrix.csv",
        index=False
    )

    print("Análisis de nulos capa 3 completado.")
    return detailed


# =========================
# 3. EDA MASTER ENCUESTA
# =========================

def eda_capa3_master_encuesta() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv"
    df = pd.read_csv(input_path)

    df.describe(include="all").to_csv(TABLES_CAPA3_EDA / "capa3_master_descriptivos.csv")

    edad_dist = df["grupo_edad"].value_counts(dropna=False).reset_index()
    edad_dist.columns = ["grupo_edad", "n"]
    edad_dist.to_csv(TABLES_CAPA3_EDA / "capa3_dist_grupo_edad.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(edad_dist["grupo_edad"], edad_dist["n"])
    plt.title("Distribución de respuestas por grupo de edad")
    plt.ylabel("Nº respuestas")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa3_dist_grupo_edad.png")

    sexo_dist = df["sexo"].value_counts(dropna=False).reset_index()
    sexo_dist.columns = ["sexo", "n"]
    sexo_dist.to_csv(TABLES_CAPA3_EDA / "capa3_dist_sexo.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(sexo_dist["sexo"], sexo_dist["n"])
    plt.title("Distribución de respuestas por sexo")
    plt.ylabel("Nº respuestas")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa3_dist_sexo.png")

    print("EDA master encuesta completado.")
    return df


# =========================
# 4. DESCRIPTIVOS DE ÍNDICES
# =========================

def eda_capa3_indices_descriptivos() -> pd.DataFrame:
    _ensure_dirs()

    df = pd.read_csv(PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv")

    idx_cols = [
        "indice_influencia_rrss",
        "indice_impulso_tendencia",
        "indice_confianza_influencers",
        "indice_escepticismo_influencers",
        "indice_difusion_fastfashion",
        "indice_postcompra",
        "indice_riesgo_arrepentimiento",
    ]

    df = _safe_numeric(df, idx_cols)

    records = []
    for col in idx_cols:
        s = df[col].dropna()
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

    desc_df = pd.DataFrame(records)
    desc_df.to_csv(TABLES_CAPA3_EDA / "capa3_indices_descriptivos.csv", index=False)

    print("EDA descriptivos de índices completado.")
    return desc_df

def eda_capa3_sample_structure() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA3 / "integrated" / "capa3_sample_structure.csv"
    df = pd.read_csv(input_path)

    df.to_csv(TABLES_CAPA3_EDA / "capa3_sample_structure.csv", index=False)

    plot_df = (
        df.groupby("grupo_edad", dropna=False)["n_respuestas"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["grupo_edad"], plot_df["n_respuestas"])
    plt.title("Estructura muestral por grupo de edad")
    plt.ylabel("Nº respuestas")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa3_sample_structure_grupo_edad.png")

    print("EDA estructura muestral completado.")
    return df

# =========================
# 5. EDA GENERACIONES
# =========================

def eda_capa3_generaciones() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA3 / "integrated" / "capa3_generacion_summary.csv"
    df = pd.read_csv(input_path)

    df.to_csv(TABLES_CAPA3_EDA / "capa3_generaciones_resumen.csv", index=False)

    metric_cols = [
        "influencia_rrss_media",
        "impulso_tendencia_medio",
        "confianza_influencers_media",
        "postcompra_media",
        "arrepentimiento_medio",
    ]

    for col in metric_cols:
        plt.figure(figsize=(8, 5))
        plt.bar(df["grupo_edad"], df[col])
        plt.title(f"{col} por grupo de edad")
        plt.ylabel(col)
        plt.xticks(rotation=45, ha="right")
        _save_plot(f"capa3_{col}_por_generacion.png")

    print("EDA generaciones completado.")
    return df


# =========================
# 6. TARGETS Y VARIABLES BINARIAS
# =========================

def eda_capa3_targets() -> None:
    _ensure_dirs()

    df = pd.read_csv(PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv")

    target_cols = [
        "compra_ult_6m_por_rrss_bin",
        "target_recomendaria_bin",
        "target_seguira_comprando_bin",
    ]

    for col in target_cols:
        tmp = df[col].value_counts(dropna=False).reset_index()
        tmp.columns = [col, "n"]
        tmp["pct"] = round(tmp["n"] / tmp["n"].sum() * 100, 2)
        tmp.to_csv(TABLES_CAPA3_EDA / f"capa3_{col}_distribution.csv", index=False)

        plot_df = tmp[tmp[col].notna()].copy()
        plt.figure(figsize=(6, 4))
        plt.bar(plot_df[col].astype(str), plot_df["n"])
        plt.title(f"Distribución de {col}")
        plt.ylabel("Nº respuestas")
        _save_plot(f"capa3_{col}_distribution.png")

    target_cols_gen = [
        "compra_ult_6m_por_rrss_bin",
        "target_recomendaria_bin",
        "target_seguira_comprando_bin",
    ]

    for col in target_cols_gen:
        cross_pct = pd.crosstab(
            df["grupo_edad"],
            df[col],
            normalize="index"
        ) * 100
        cross_pct = cross_pct.round(2)
        cross_pct.to_csv(TABLES_CAPA3_EDA / f"capa3_{col}_por_generacion_pct.csv")

        cross_abs = pd.crosstab(
            df["grupo_edad"],
            df[col]
        )
        cross_abs.to_csv(TABLES_CAPA3_EDA / f"capa3_{col}_por_generacion_abs.csv")

    print("EDA targets completado.")

def eda_capa3_target_summary() -> pd.DataFrame:
    _ensure_dirs()

    df = pd.read_csv(PROCESSED_CAPA3 / "integrated" / "capa3_target_summary.csv")
    df.to_csv(TABLES_CAPA3_EDA / "capa3_target_summary_eda.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(df["target"], df["pct_clase_1"])
    plt.title("Porcentaje de clase positiva por target")
    plt.ylabel("% clase 1")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa3_target_pct_clase_1.png")

    print("EDA target summary completado.")
    return df


# =========================
# 7. CRUCES POR GENERACIÓN
# =========================

def eda_capa3_cruces_generacion() -> None:
    _ensure_dirs()

    df = pd.read_csv(PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv")

    # grupo_edad x compra_ult_6m_por_rrss_bin
    cruz_compra = pd.crosstab(
        df["grupo_edad"],
        df["compra_ult_6m_por_rrss_bin"],
        margins=False,
        normalize="index"
    ) * 100
    cruz_compra = cruz_compra.round(2)
    cruz_compra.to_csv(TABLES_CAPA3_EDA / "capa3_cruce_generacion_compra_ult_6m_rrss_pct.csv")

    cruz_compra_abs = pd.crosstab(
        df["grupo_edad"],
        df["compra_ult_6m_por_rrss_bin"],
        margins=False
    )
    cruz_compra_abs.to_csv(TABLES_CAPA3_EDA / "capa3_cruce_generacion_compra_ult_6m_rrss_abs.csv")

    # grupo_edad x canal_compra_moda
    cruz_canal = pd.crosstab(
        df["grupo_edad"],
        df["canal_compra_moda"],
        margins=False,
        normalize="index"
    ) * 100
    cruz_canal = cruz_canal.round(2)
    cruz_canal.to_csv(TABLES_CAPA3_EDA / "capa3_cruce_generacion_canal_compra_pct.csv")

    cruz_canal_abs = pd.crosstab(
        df["grupo_edad"],
        df["canal_compra_moda"],
        margins=False
    )
    cruz_canal_abs.to_csv(TABLES_CAPA3_EDA / "capa3_cruce_generacion_canal_compra_abs.csv")

    # grupo_edad x sigue_influencers_moda
    cruz_influencers = pd.crosstab(
        df["grupo_edad"],
        df["sigue_influencers_moda"],
        margins=False,
        normalize="index"
    ) * 100
    cruz_influencers = cruz_influencers.round(2)
    cruz_influencers.to_csv(TABLES_CAPA3_EDA / "capa3_cruce_generacion_sigue_influencers_pct.csv")

    print("EDA cruces por generación completado.")


# =========================
# 8. BOXPLOTS POR GENERACIÓN
# =========================

def eda_capa3_boxplots_generacion() -> None:
    _ensure_dirs()

    df = pd.read_csv(PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv")

    box_cols = [
        "indice_influencia_rrss",
        "indice_impulso_tendencia",
        "indice_postcompra",
        "indice_riesgo_arrepentimiento",
    ]
    df = _safe_numeric(df, box_cols)

    order = ["18 - 28 años", "29 - 44 años", "45 - 55 años", "56 años o más"]

    for col in box_cols:
        grouped = []
        labels = []
        for grp in order:
            s = df.loc[df["grupo_edad"] == grp, col].dropna()
            if not s.empty:
                grouped.append(s.values)
                labels.append(grp)

        if not grouped:
            continue

        plt.figure(figsize=(9, 5))
        plt.boxplot(grouped, tick_labels=labels)
        plt.title(f"{col} por grupo de edad")
        plt.ylabel(col)
        plt.xticks(rotation=45, ha="right")
        _save_plot(f"capa3_boxplot_{col}_por_generacion.png")

    print("EDA boxplots por generación completado.")


# =========================
# 9. EDA RRSS Y MARCAS
# =========================

def eda_capa3_multirrespuesta() -> None:
    _ensure_dirs()

    rrss = pd.read_csv(PROCESSED_CAPA3 / "survey" / "encuesta_rrss_long.csv")
    marcas_vistas = pd.read_csv(PROCESSED_CAPA3 / "survey" / "encuesta_marcas_vistas_long.csv")
    marcas_influyen = pd.read_csv(PROCESSED_CAPA3 / "survey" / "encuesta_marcas_influyen_long.csv")
    master = pd.read_csv(PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv")

    rrss_rank = rrss["rrss"].value_counts(dropna=False).reset_index()
    rrss_rank.columns = ["rrss", "n"]
    rrss_rank.to_csv(TABLES_CAPA3_EDA / "capa3_rrss_rank.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(rrss_rank["rrss"], rrss_rank["n"])
    plt.title("Redes sociales más utilizadas")
    plt.ylabel("Nº menciones")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa3_rrss_rank.png")

    marcas_vistas_rank = marcas_vistas["marca_vista_rrss"].value_counts(dropna=False).reset_index()
    marcas_vistas_rank.columns = ["marca_vista_rrss", "n"]
    marcas_vistas_rank.to_csv(TABLES_CAPA3_EDA / "capa3_marcas_vistas_rank.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(marcas_vistas_rank["marca_vista_rrss"], marcas_vistas_rank["n"])
    plt.title("Marcas vistas con más frecuencia en redes")
    plt.ylabel("Nº menciones")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa3_marcas_vistas_rank.png")

    marcas_influyen_rank = marcas_influyen["marca_influye_compra_rrss"].value_counts(dropna=False).reset_index()
    marcas_influyen_rank.columns = ["marca_influye_compra_rrss", "n"]
    marcas_influyen_rank.to_csv(TABLES_CAPA3_EDA / "capa3_marcas_influyen_rank.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(marcas_influyen_rank["marca_influye_compra_rrss"], marcas_influyen_rank["n"])
    plt.title("Marcas que más influyen en compra")
    plt.ylabel("Nº menciones")
    plt.xticks(rotation=45, ha="right")
    _save_plot("capa3_marcas_influyen_rank.png")

    # ranking de marcas influyentes por generación
    marcas_influyen_gen = marcas_influyen.merge(
        master[["id_respuesta", "grupo_edad"]],
        on="id_respuesta",
        how="left"
    )

    rank_gen = (
        marcas_influyen_gen.groupby(["grupo_edad", "marca_influye_compra_rrss"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["grupo_edad", "n"], ascending=[True, False])
    )
    rank_gen.to_csv(TABLES_CAPA3_EDA / "capa3_marcas_influyen_rank_por_generacion.csv", index=False)

    print("EDA multirrespuesta completado.")


# =========================
# 10. EDA CORRELACIONES
# =========================

def eda_capa3_correlations() -> pd.DataFrame:
    _ensure_dirs()

    df = pd.read_csv(PROCESSED_CAPA3 / "integrated" / "capa3_master_encuesta.csv")

    corr_cols = [
        "indice_influencia_rrss",
        "indice_impulso_tendencia",
        "indice_confianza_influencers",
        "indice_escepticismo_influencers",
        "indice_difusion_fastfashion",
        "indice_postcompra",
        "indice_riesgo_arrepentimiento",
        "compra_ult_6m_por_rrss_bin",
        "target_recomendaria_bin",
        "target_seguira_comprando_bin",
    ]

    df = _safe_numeric(df, corr_cols)
    corr_df = df[corr_cols].corr()
    corr_df.to_csv(TABLES_CAPA3_EDA / "capa3_correlation_indices.csv")

    _plot_corr(
        corr_df,
        "Correlación entre índices de encuesta",
        "capa3_correlation_indices.png",
    )

    print("EDA correlaciones completado.")
    return corr_df

# =========================
# 11. EDA CLUSTERING READY
# =========================

def eda_capa3_clustering_ready() -> pd.DataFrame:
    _ensure_dirs()

    df = pd.read_csv(PROCESSED_CAPA3 / "integrated" / "capa3_clustering_ready.csv")

    numeric_cols = [
        "freq_compra_anual",
        "canal_compra_moda",
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
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    numeric_desc = df[numeric_cols].describe().T
    numeric_desc.to_csv(TABLES_CAPA3_EDA / "capa3_clustering_ready_numeric_descriptives.csv")

    print("EDA clustering ready completado.")
    return df

# =========================
# 12. EDA SUPERVISED READY
# =========================

def eda_capa3_supervised_ready() -> pd.DataFrame:
    _ensure_dirs()

    df = pd.read_csv(PROCESSED_CAPA3 / "integrated" / "capa3_supervised_ready.csv")

    numeric_cols = [
        "freq_compra_anual",
        "canal_compra_moda",
        "tiempo_rrss_dia",
        "freq_contenido_moda_rrss",
        "sigue_influencers_moda",
        "compra_ult_6m_por_rrss_bin",
        "indice_influencia_rrss",
        "indice_impulso_tendencia",
        "indice_confianza_influencers",
        "indice_escepticismo_influencers",
        "indice_difusion_fastfashion",
        "indice_postcompra",
        "indice_riesgo_arrepentimiento",
        "target_recomendaria_bin",
        "target_seguira_comprando_bin",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    numeric_desc = df[numeric_cols].describe().T
    numeric_desc.to_csv(TABLES_CAPA3_EDA / "capa3_supervised_ready_numeric_descriptives.csv")

    print("EDA supervised ready completado.")
    return df

# =========================
# RUN ALL
# =========================

def run_all_eda() -> None:
    profile_capa3()
    analyze_nulls_capa3()
    eda_capa3_master_encuesta()
    eda_capa3_indices_descriptivos()
    eda_capa3_generaciones()
    eda_capa3_targets()
    eda_capa3_cruces_generacion()
    eda_capa3_boxplots_generacion()
    eda_capa3_multirrespuesta()
    eda_capa3_clustering_ready()
    eda_capa3_supervised_ready()
    eda_capa3_correlations()
    eda_capa3_sample_structure()
    eda_capa3_target_summary()
    print("Todo el EDA de capa 3 completado.")

if __name__ == "__main__":
    run_all_eda()