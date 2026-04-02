"""
transform_capa3.py — Transformación ETL de Capa 3: Encuesta fast fashion

Fuente: encuesta_fastfashion.csv (elaboración propia, Google Forms, n=~200)

Decisiones metodológicas clave:
  - Escala Likert 1-5: todas las variables de actitud y comportamiento usan
    escala Likert de 5 puntos. Se conservan como valores numéricos continuos
    para permitir cálculo de medias e índices compuestos.
  - Índices compuestos: 7 índices calculados como media aritmética de ítems
    conceptualmente relacionados.# La consistencia interna de los índices se evalúa de forma exploratoria
    mediante alpha de Cronbach en build_capa3.py.
  - Justificación de los índices: cada índice agrupa ítems con coherencia
    conceptual documentada. Ver función _INDICES_JUSTIFICACION abajo.
  - NaN en variables Likert: valores no respondidos. Se conservan como NaN
    (no se imputan) para no distorsionar los índices compuestos.
  - Timestamps NaT: el formato original del CSV de Google Forms incluye
    marcadores de zona horaria no estándar. Se registra el número de NaT.
  - Tabla antes/después: snapshot pre/post transformación exportado para
    auditoría metodológica.
"""

import warnings

import numpy as np
import pandas as pd

from src.common.config import PROCESSED_CAPA3, RAW_CAPA3

warnings.filterwarnings("ignore")


# =========================
# JUSTIFICACIÓN DE ÍNDICES COMPUESTOS
# =========================
# Esta constante documenta la coherencia conceptual de cada índice.
# La validación estadística (alpha de Cronbach) se realiza en build_capa3.py.

INDICES_JUSTIFICACION = {
    "indice_influencia_rrss": {
        "items": ["rs_influyen_compra", "ha_comprado_por_ver_en_rrss",
                  "descubre_marcas_por_rrss", "contenido_moda_aumenta_ganas",
                  "rrss_mas_que_publicidad"],
        "justificacion": (
            "Mide el grado en que las redes sociales influyen en las decisiones "
            "de compra de moda. Los 5 items capturan distintas dimensiones del "
            "mismo constructo: influencia declarada, comportamiento de compra "
            "post-exposicion, descubrimiento de marcas, estimulacion del deseo "
            "y comparacion con publicidad tradicional."
        ),
    },
    "indice_impulso_tendencia": {
        "items": ["compra_por_repeticion_rrss", "atraen_prendas_de_moda",
                  "compra_impulso", "influye_prueba_social", "interes_productos_virales"],
        "justificacion": (
            "Mide la propension a comprar por impulso o seguimiento de tendencias. "
            "Los items capturan la compra compulsiva por repeticion de estimulos, "
            "atraccion por lo que esta de moda, compra impulsiva general, "
            "influencia de la prueba social y seguimiento de productos virales."
        ),
    },
    "indice_confianza_influencers": {
        "items": ["confia_influencers", "autenticidad_influencer_compra"],
        "justificacion": (
            "Mide la confianza en las recomendaciones de influencers. "
            "Solo 2 items (minimo para un indice); la validez se evalua "
            "principalmente por coherencia conceptual y correlacion entre items."
        ),
    },
    "indice_escepticismo_influencers": {
        "items": ["influencers_interes_comercial"],
        "justificacion": (
            "Item unico que mide la percepcion de que los influencers actuan "
            "por interes comercial. Se mantiene como indice de 1 item para "
            "permitir comparacion directa con indice_confianza_influencers. "
            "NOTA: alpha de Cronbach no aplicable a indices de 1 item."
        ),
    },
    "indice_difusion_fastfashion": {
        "items": ["rrss_aceleran_tendencias", "rrss_favorecen_consumo_rapido"],
        "justificacion": (
            "Mide la percepcion del rol de las RRSS en la difusion del fast fashion. "
            "Los 2 items capturan la aceleracion de tendencias y el consumo rapido "
            "como consecuencias del uso de redes en el sector moda."
        ),
    },
    "indice_postcompra": {
        "items": ["satisfaccion_compra_rrss", "seguira_comprando_influido_rrss",
                  "recomendaria_productos_descubiertos_rrss"],
        "justificacion": (
            "Mide la experiencia y actitud post-compra tras adquirir productos "
            "descubiertos en RRSS. Los 3 items capturan satisfaccion, intencion "
            "de repeticion y disposicion a recomendar: las 3 dimensiones clasicas "
            "del modelo de satisfaccion del consumidor (Oliver, 1980)."
        ),
    },
    "indice_riesgo_arrepentimiento": {
        "items": ["ha_comprado_y_apenas_usado", "arrepentimiento_compra_rrss"],
        "justificacion": (
            "Mide el riesgo de arrepentimiento post-compra. Los 2 items capturan "
            "el comportamiento (comprar y no usar) y la actitud (arrepentimiento) "
            "como dos facetas del mismo constructo de riesgo de compra impulsiva."
        ),
    },
}


# =========================
# HELPERS
# =========================

def _ensure_dirs() -> None:
    for sub in ["survey", "integrated", "calidad"]:
        (PROCESSED_CAPA3 / sub).mkdir(parents=True, exist_ok=True)


def _split_multiselect(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(";") if str(x).strip()]


def _to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _save_antes_despues(before_summary: dict, after_summary: dict,
                         label: str, out_dir) -> None:
    rows = [
        {"_fase": "ANTES", **before_summary},
        {"_fase": "DESPUÉS", **after_summary},
    ]
    pd.DataFrame(rows).to_csv(out_dir / f"antes_despues_{label}.csv", index=False)
    print(f"  ✓ antes_despues_{label}.csv")


def _build_null_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """Genera tabla de decisiones de nulos para la encuesta."""
    records = []
    for col in df.columns:
        n_null = int(df[col].isnull().sum())
        pct_null = round(float(df[col].isnull().mean() * 100), 2)
        if n_null == 0:
            continue

        if col == "timestamp":
            decision = "mantener_solo_control"
            comentario = "NaT por formato zona horaria Google Forms. No afecta al analisis sustantivo."
        elif col in {"rrss_habituales", "marcas_ve_frecuencia_rrss", "marcas_que_mas_influyen_compra_rrss"}:
            decision = "mantener_por_trazabilidad"
            comentario = "Campos multirrespuesta — el analisis opera sobre tablas long normalizadas."
        elif col == "freq_compra_por_rrss_6m":
            decision = "esperado — solo aplica a quien respondio Si en compra_ult_6m_por_rrss"
            comentario = "Pregunta condicional: NaN valido para quienes respondieron No."
        elif col in [c for idx in INDICES_JUSTIFICACION.values() for c in idx["items"]]:
            decision = "mantener_NaN — no imputable sin distorsion del indice"
            comentario = "Item Likert no respondido. Imputer distorsionaria la media del indice compuesto."
        elif pct_null >= 50:
            decision = "revisar"
            comentario = f"Nivel de nulos muy alto ({pct_null}%). Revisar antes de usar en modelos."
        elif pct_null >= 20:
            decision = "mantener_con_cautela"
            comentario = f"Nivel de nulos elevado ({pct_null}%). Usar con precaucion en modelos."
        else:
            decision = "mantener"
            comentario = "Nivel de nulos asumible."

        records.append({
            "variable": col,
            "n_nulos": n_null,
            "pct_nulos": pct_null,
            "decision": decision,
            "comentario": comentario,
        })

    return pd.DataFrame(records)


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
    "freq_compra_anual", "canal_compra_moda", "tiempo_rrss_dia",
    "freq_contenido_moda_rrss", "sigue_influencers_moda",
    "rs_influyen_compra", "ha_comprado_por_ver_en_rrss", "descubre_marcas_por_rrss",
    "contenido_moda_aumenta_ganas", "compra_por_repeticion_rrss",
    "rrss_mas_que_publicidad", "atraen_prendas_de_moda", "compra_impulso",
    "influye_prueba_social", "interes_productos_virales", "rrss_aceleran_tendencias",
    "rrss_favorecen_consumo_rapido", "ha_comprado_y_apenas_usado",
    "confia_influencers", "autenticidad_influencer_compra",
    "influencers_interes_comercial", "freq_compra_por_rrss_6m",
    "satisfaccion_compra_rrss", "arrepentimiento_compra_rrss",
    "seguira_comprando_influido_rrss", "recomendaria_productos_descubiertos_rrss",
]


# =========================
# MAIN TRANSFORM
# =========================

def transform_encuesta_fastfashion() -> pd.DataFrame:
    """
    Transforma la encuesta de fast fashion aplicando:
    1. Renombrado de columnas a snake_case
    2. Parsing de timestamps (con log de NaT)
    3. Limpieza de variables de texto y Likert
    4. Cálculo de índices compuestos (ver INDICES_JUSTIFICACION)
    5. Variables binarias target para modelos supervisados
    6. Expansión de preguntas multirrespuesta a tablas long
    7. Tabla antes/después y decisiones de nulos exportadas
    """
    _ensure_dirs()
    cal_dir = PROCESSED_CAPA3 / "calidad"
    survey_dir = PROCESSED_CAPA3 / "survey"

    input_path = RAW_CAPA3 / "encuesta_fastfashion.csv"
    df_raw = pd.read_csv(input_path).copy()

    # --- Snapshot ANTES ---
    before_summary = {
        "n_filas": len(df_raw),
        "n_columnas": df_raw.shape[1],
        "n_nulos_total": int(df_raw.isnull().sum().sum()),
        "columnas_originales": df_raw.shape[1],
        "indices_calculados": 0,
        "variables_binarias": 0,
    }

    df = df_raw.rename(columns=RENAME_MAP)
    df["id_respuesta"] = range(1, len(df) + 1)

    # --- Timestamps ---
    df["timestamp"] = (
        df["timestamp"].astype(str)
        .str.replace("\u202f", " ", regex=False)
        .str.replace("a. m.", "AM", regex=False)
        .str.replace("p. m.", "PM", regex=False)
        .str.replace(" EET", "", regex=False)
        .str.strip()
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %I:%M:%S %p", errors="coerce")
    n_nat = int(df["timestamp"].isna().sum())
    if n_nat > 0:
        print(f"  [TIMESTAMP] {n_nat} NaT tras parsing — formato no estándar Google Forms. "
              f"No afecta al análisis sustantivo.")

    # --- Variables de texto ---
    text_cols = ["grupo_edad", "sexo", "canal_compra_moda", "gasto_mensual_moda",
                 "rrss_habituales", "marcas_ve_frecuencia_rrss",
                 "compra_ult_6m_por_rrss", "marcas_que_mas_influyen_compra_rrss"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})

    # --- Likert ---
    for col in LIKERT_COLS:
        if col in df.columns:
            df[col] = _to_numeric_safe(df[col])

    # --- Variable binaria compra por RRSS ---
    df["compra_ult_6m_por_rrss_bin"] = pd.to_numeric(
        df["compra_ult_6m_por_rrss"].replace({"Sí": 1, "No": 0}), errors="coerce"
    )

    # --- Índices compuestos ---
    for nombre_indice, meta in INDICES_JUSTIFICACION.items():
        items = [c for c in meta["items"] if c in df.columns]
        if len(items) >= 1:
            df[nombre_indice] = df[items].mean(axis=1)

    # --- Targets supervisados ---
    df["target_recomendaria_bin"] = np.where(
        df["recomendaria_productos_descubiertos_rrss"] >= 4, 1,
        np.where(df["recomendaria_productos_descubiertos_rrss"].notna(), 0, np.nan)
    )
    df["target_seguira_comprando_bin"] = np.where(
        df["seguira_comprando_influido_rrss"] >= 4, 1,
        np.where(df["seguira_comprando_influido_rrss"].notna(), 0, np.nan)
    )

    # --- Snapshot DESPUÉS ---
    after_summary = {
        "n_filas": len(df),
        "n_columnas": df.shape[1],
        "n_nulos_total": int(df.isnull().sum().sum()),
        "columnas_originales": df_raw.shape[1],
        "indices_calculados": len(INDICES_JUSTIFICACION),
        "variables_binarias": 3,  # compra_bin + 2 targets
    }
    _save_antes_despues(before_summary, after_summary, "encuesta_fastfashion", survey_dir)

    # --- Decisiones de nulos ---
    null_decisions = _build_null_decisions(df)
    null_decisions.to_csv(cal_dir / "encuesta_null_decisions.csv", index=False)
    print(f"  Decisiones de nulos: {len(null_decisions)} variables con nulos documentadas")

    # --- Justificación de índices ---
    idx_records = []
    for nombre, meta in INDICES_JUSTIFICACION.items():
        idx_records.append({
            "indice": nombre,
            "n_items": len(meta["items"]),
            "items": ", ".join(meta["items"]),
            "justificacion_conceptual": meta["justificacion"],
            "validacion_estadistica": "alpha_de_Cronbach — ver build_capa3.py",
            "umbral_alpha_minimo": "0.6 (aceptable) / 0.7 (bueno) / 0.8 (muy bueno)",
        })
    indices_df = pd.DataFrame(idx_records)
    indices_df.to_csv(cal_dir / "encuesta_indices_justificacion.csv", index=False)
    print(f"  Justificación índices: {len(indices_df)} índices documentados")

    # --- Multirrespuesta ---
    def _expand_multiselect(df, id_col, value_col, output_col):
        records = []
        for _, row in df[[id_col, value_col]].iterrows():
            for opcion in _split_multiselect(row[value_col]):
                records.append({id_col: row[id_col], output_col: opcion})
        return pd.DataFrame(records)

    rrss_df = _expand_multiselect(df, "id_respuesta", "rrss_habituales", "rrss")
    marcas_vistas_df = _expand_multiselect(df, "id_respuesta", "marcas_ve_frecuencia_rrss", "marca_vista_rrss")
    marcas_influyen_df = _expand_multiselect(df, "id_respuesta", "marcas_que_mas_influyen_compra_rrss", "marca_influye_compra_rrss")

    # --- Exports ---
    df.to_csv(survey_dir / "encuesta_fastfashion_clean.csv", index=False)
    rrss_df.to_csv(survey_dir / "encuesta_rrss_long.csv", index=False)
    marcas_vistas_df.to_csv(survey_dir / "encuesta_marcas_vistas_long.csv", index=False)
    marcas_influyen_df.to_csv(survey_dir / "encuesta_marcas_influyen_long.csv", index=False)

    n_indices = len(INDICES_JUSTIFICACION)
    print(f"  Encuesta transformada: {len(df)} respuestas | "
          f"{n_indices} índices | "
          f"{len(LIKERT_COLS)} variables Likert | "
          f"NaT timestamps: {n_nat}")
    return df


def run_all_transforms() -> None:
    sep = "=" * 65
    print(sep)
    print("CAPA 3 — TRANSFORMACIONES ETL")
    print(sep)
    print("\n[1/1] Encuesta fast fashion...")
    transform_encuesta_fastfashion()
    print(f"\n{sep}")
    print("TRANSFORM CAPA 3 COMPLETADO")
    print("Outputs calidad generados:")
    print("  · calidad/encuesta_null_decisions.csv")
    print("  · calidad/encuesta_indices_justificacion.csv")
    print("  · survey/antes_despues_encuesta_fastfashion.csv")
    print("Nota: validacion estadistica de indices → build_capa3.py (Cronbach alpha)")
    print(sep)


if __name__ == "__main__":
    run_all_transforms()