import pandas as pd
from config import RAW_CONTEXTO, DATA_PROCESSED
from utils import clean_percentage, ensure_year_int

def transform_contexto_digitalizacion():
    print("Ruta esperada del archivo:")
    print(RAW_CONTEXTO)
    print("")

    if not RAW_CONTEXTO.exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {RAW_CONTEXTO}")

    df = pd.read_excel(RAW_CONTEXTO)

    df = df.rename(columns={
        "pct_usuarios_RRSS": "pct_usuarios_rrss"
    })

    for col in [
        "pct_usuarios_rrss",
        "pct_personas_compra_online",
        "pct_personas_compra_ropa_online"
    ]:
        if col in df.columns:
            df[col] = df[col].apply(clean_percentage)

    df = ensure_year_int(df, "anio")

    # Dataset principal
    analytic_cols = [
        "anio",
        "pct_usuarios_rrss",
        "pct_personas_compra_online"
    ]
    df_analytic = df[analytic_cols].copy()

    # Dataset extendido
    extended_cols = [
        "anio",
        "pct_usuarios_rrss",
        "pct_personas_compra_online",
        "pct_personas_compra_ropa_online"
    ]
    df_extended = df[extended_cols].copy()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    clean_path = DATA_PROCESSED / "contexto_digitalizacion_clean.csv"
    extended_path = DATA_PROCESSED / "contexto_digitalizacion_extended.csv"
    doc_path = DATA_PROCESSED / "contexto_digitalizacion_documentado.csv"

    df_analytic.to_csv(clean_path, index=False)
    df_extended.to_csv(extended_path, index=False)
    df.to_csv(doc_path, index=False)

    print("Archivos guardados en:")
    print(clean_path)
    print(extended_path)
    print(doc_path)
    print("")
    print("Preview del dataset limpio:")
    print(df_analytic.head())

    return df_analytic, df_extended, df

if __name__ == "__main__":
    analytic, extended, documented = transform_contexto_digitalizacion()