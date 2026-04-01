import pandas as pd
import numpy as np

from src.common.config import RAW_CAPA3, PROCESSED_CAPA3


# =========================
# HELPERS
# =========================

def _ensure_dirs() -> None:
    (PROCESSED_CAPA3 / "survey").mkdir(parents=True, exist_ok=True)
    (PROCESSED_CAPA3 / "integrated").mkdir(parents=True, exist_ok=True)


def _split_multiselect(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(";") if str(x).strip()]


def _to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# =========================
# RENAME MAP
# =========================

RENAME_MAP = {
    "Marca temporal": "timestamp",
    "¿A qué grupo de edad perteneces?": "grupo_edad",
    "¿Cuál es tu sexo?": "sexo",
    "¿Con qué frecuencia compras ropa, calzado o accesorios a lo largo del año? ": "freq_compra_anual",
    "¿Cómo realizas normalmente tus compras de moda?": "canal_compra_moda",
    "¿Cuál es tu gasto mensual aproximado en moda?": "gasto_mensual_moda",
    "¿Qué redes sociales utilizas habitualmente?": "rrss_habituales",
    "¿Qué marcas de moda ves con más frecuencia en redes sociales?  Selecciona un máximo de 3 opciones   ": "marcas_ve_frecuencia_rrss",
    "¿Cuánto tiempo pasas al día en las redes sociales?": "tiempo_rrss_dia",
    "¿Con qué frecuencia consumes contenido relacionado con moda en redes sociales?  ": "freq_contenido_moda_rrss",
    "¿Sigues a influencers o creadores de contenido de moda?  ": "sigue_influencers_moda",
    "Las redes sociales influyen en mis decisiones de compra de moda.  ": "rs_influyen_compra",
    "He comprado ropa o accesorios después de verlos en redes sociales.  ": "ha_comprado_por_ver_en_rrss",
    "Descubro nuevas marcas de moda gracias a las redes sociales.  ": "descubre_marcas_por_rrss",
    "Ver contenido de moda en redes aumenta mis ganas de comprar.  ": "contenido_moda_aumenta_ganas",
    "A veces compro prendas después de verlas repetidamente en redes sociales, aunque realmente no las necesite. ": "compra_por_repeticion_rrss",
    "Las redes sociales influyen más en mis compras de moda que la publicidad tradicional.": "rrss_mas_que_publicidad",
    "Me atrae comprar prendas que están de moda en ese momento.  ": "atraen_prendas_de_moda",
    "A veces compro ropa por impulso.": "compra_impulso",
    "Me influye ver que muchas personas llevan una prenda o una marca concreta.  ": "influye_prueba_social",
    "Suelo interesarme por productos o prendas virales en redes sociales.  ": "interes_productos_virales",
    "Creo que las redes sociales aceleran la difusión de tendencias de moda.  ": "rrss_aceleran_tendencias",
    "Las redes sociales favorecen un consumo rápido y constante de moda.  ": "rrss_favorecen_consumo_rapido",
    "He comprado prendas por tendencia o por influencia de redes que después apenas he usado.  ": "ha_comprado_y_apenas_usado",
    "Confío en las recomendaciones de influencers de moda.": "confia_influencers",
    "Si un creador de contenido me parece auténtico, es más probable que compre lo que recomienda.  ": "autenticidad_influencer_compra",
    "Creo que muchos influencers recomiendan productos solo por interés comercial.  ": "influencers_interes_comercial",
    "En los últimos 6 meses, ¿has comprado alguna prenda o accesorio por influencia de redes sociales?  ": "compra_ult_6m_por_rrss",
    "Si has respondido sí, ¿con qué frecuencia te ocurre?  ": "freq_compra_por_rrss_6m",
    "¿Qué marcas te han influido más en alguna compra después de verlas en redes sociales? Selecciona un máximo de 3 opciones": "marcas_que_mas_influyen_compra_rrss",
    "En general, quedo satisfecho/a con las compras de moda que realizo tras ver contenido en redes sociales.  ": "satisfaccion_compra_rrss",
    "Me he arrepentido alguna vez de comprar ropa que había visto previamente en las redes sociales.  ": "arrepentimiento_compra_rrss",
    "Seguiré comprando moda influido/a por contenido que vea en redes sociales.  ": "seguira_comprando_influido_rrss",
    "Recomendaría comprar productos de moda descubiertos en redes sociales.  ": "recomendaria_productos_descubiertos_rrss",
}


LIKERT_COLS = [
    "freq_compra_anual",
    "canal_compra_moda",
    "tiempo_rrss_dia",
    "freq_contenido_moda_rrss",
    "sigue_influencers_moda",
    "rs_influyen_compra",
    "ha_comprado_por_ver_en_rrss",
    "descubre_marcas_por_rrss",
    "contenido_moda_aumenta_ganas",
    "compra_por_repeticion_rrss",
    "rrss_mas_que_publicidad",
    "atraen_prendas_de_moda",
    "compra_impulso",
    "influye_prueba_social",
    "interes_productos_virales",
    "rrss_aceleran_tendencias",
    "rrss_favorecen_consumo_rapido",
    "ha_comprado_y_apenas_usado",
    "confia_influencers",
    "autenticidad_influencer_compra",
    "influencers_interes_comercial",
    "freq_compra_por_rrss_6m",
    "satisfaccion_compra_rrss",
    "arrepentimiento_compra_rrss",
    "seguira_comprando_influido_rrss",
    "recomendaria_productos_descubiertos_rrss",
]


# =========================
# MAIN TRANSFORM
# =========================

def transform_encuesta_fastfashion() -> pd.DataFrame:
    _ensure_dirs()

    input_path = RAW_CAPA3 / "encuesta_fastfashion.csv"
    df = pd.read_csv(input_path).copy()

    df = df.rename(columns=RENAME_MAP)

    df["id_respuesta"] = range(1, len(df) + 1)

    df["timestamp"] = (
        df["timestamp"]
        .astype(str)
        .str.replace("\u202f", " ", regex=False)
        .str.replace("a. m.", "AM", regex=False)
        .str.replace("p. m.", "PM", regex=False)
        .str.replace(" EET", "", regex=False)
        .str.strip()
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y/%m/%d %I:%M:%S %p",
        errors="coerce"
    )

    text_cols = [
        "grupo_edad",
        "sexo",
        "canal_compra_moda",
        "gasto_mensual_moda",
        "rrss_habituales",
        "marcas_ve_frecuencia_rrss",
        "compra_ult_6m_por_rrss",
        "marcas_que_mas_influyen_compra_rrss",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"nan": np.nan, "": np.nan})
            )

    for col in LIKERT_COLS:
        if col in df.columns:
            df[col] = _to_numeric_safe(df[col])

    df["compra_ult_6m_por_rrss_bin"] = (
        df["compra_ult_6m_por_rrss"]
        .replace({"Sí": 1, "No": 0})
    )
    df["compra_ult_6m_por_rrss_bin"] = pd.to_numeric(df["compra_ult_6m_por_rrss_bin"], errors="coerce")

    # =========================
    # VARIABLES DERIVADAS / ÍNDICES
    # =========================
    df["indice_influencia_rrss"] = df[
        ["rs_influyen_compra", "ha_comprado_por_ver_en_rrss", "descubre_marcas_por_rrss", "contenido_moda_aumenta_ganas", "rrss_mas_que_publicidad"]
    ].mean(axis=1)

    df["indice_impulso_tendencia"] = df[
        ["compra_por_repeticion_rrss", "atraen_prendas_de_moda", "compra_impulso", "influye_prueba_social", "interes_productos_virales"]
    ].mean(axis=1)

    df["indice_confianza_influencers"] = df[
        ["confia_influencers", "autenticidad_influencer_compra"]
    ].mean(axis=1)

    df["indice_escepticismo_influencers"] = df[
        ["influencers_interes_comercial"]
    ].mean(axis=1)

    df["indice_difusion_fastfashion"] = df[
        ["rrss_aceleran_tendencias", "rrss_favorecen_consumo_rapido"]
    ].mean(axis=1)

    df["indice_postcompra"] = df[
        ["satisfaccion_compra_rrss", "seguira_comprando_influido_rrss", "recomendaria_productos_descubiertos_rrss"]
    ].mean(axis=1)

    df["indice_riesgo_arrepentimiento"] = df[
        ["ha_comprado_y_apenas_usado", "arrepentimiento_compra_rrss"]
    ].mean(axis=1)

    # target sugerido para modelos supervisados
    df["target_recomendaria_bin"] = np.where(
        df["recomendaria_productos_descubiertos_rrss"] >= 4, 1,
        np.where(df["recomendaria_productos_descubiertos_rrss"].notna(), 0, np.nan)
    )

    df["target_seguira_comprando_bin"] = np.where(
        df["seguira_comprando_influido_rrss"] >= 4, 1,
        np.where(df["seguira_comprando_influido_rrss"].notna(), 0, np.nan)
    )

    # =========================
    # MULTIRRESPUESTA: RRSS
    # =========================
    rrss_records = []
    for _, row in df[["id_respuesta", "rrss_habituales"]].iterrows():
        opciones = _split_multiselect(row["rrss_habituales"])
        for opcion in opciones:
            rrss_records.append(
                {"id_respuesta": row["id_respuesta"], "rrss": opcion}
            )
    rrss_df = pd.DataFrame(rrss_records)

    # =========================
    # MULTIRRESPUESTA: MARCAS VISIBLES EN RRSS
    # =========================
    marcas_vistas_records = []
    for _, row in df[["id_respuesta", "marcas_ve_frecuencia_rrss"]].iterrows():
        opciones = _split_multiselect(row["marcas_ve_frecuencia_rrss"])
        for opcion in opciones:
            marcas_vistas_records.append(
                {"id_respuesta": row["id_respuesta"], "marca_vista_rrss": opcion}
            )
    marcas_vistas_df = pd.DataFrame(marcas_vistas_records)

    # =========================
    # MULTIRRESPUESTA: MARCAS QUE MÁS INFLUYEN
    # =========================
    marcas_influyen_records = []
    for _, row in df[["id_respuesta", "marcas_que_mas_influyen_compra_rrss"]].iterrows():
        opciones = _split_multiselect(row["marcas_que_mas_influyen_compra_rrss"])
        for opcion in opciones:
            marcas_influyen_records.append(
                {"id_respuesta": row["id_respuesta"], "marca_influye_compra_rrss": opcion}
            )
    marcas_influyen_df = pd.DataFrame(marcas_influyen_records)

    # =========================
    # EXPORTS
    # =========================
    output_main = PROCESSED_CAPA3 / "survey" / "encuesta_fastfashion_clean.csv"
    output_rrss = PROCESSED_CAPA3 / "survey" / "encuesta_rrss_long.csv"
    output_marcas_vistas = PROCESSED_CAPA3 / "survey" / "encuesta_marcas_vistas_long.csv"
    output_marcas_influyen = PROCESSED_CAPA3 / "survey" / "encuesta_marcas_influyen_long.csv"

    df.to_csv(output_main, index=False)
    rrss_df.to_csv(output_rrss, index=False)
    marcas_vistas_df.to_csv(output_marcas_vistas, index=False)
    marcas_influyen_df.to_csv(output_marcas_influyen, index=False)

    print("encuesta_fastfashion_clean.csv guardado en:")
    print(output_main)
    print("")
    print(df.head())

    return df


def run_all_transforms() -> None:
    transform_encuesta_fastfashion()
    print("Todas las transformaciones de capa 3 completadas.")


if __name__ == "__main__":
    run_all_transforms()