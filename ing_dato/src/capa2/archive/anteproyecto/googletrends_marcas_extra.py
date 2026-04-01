from pathlib import Path
import time
import pandas as pd
from pytrends.request import TrendReq

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
RAW_GT_AGG = BASE_DIR / "data" / "raw" / "capa2" / "googletrends" / "aggregated"
RAW_GT_AGG.mkdir(parents=True, exist_ok=True)

# Marcas que te faltan para alinear con encuesta + Instagram
BRANDS = [
    "H&M",
    "Massimo Dutti",
]

# Google Trends config
GEO = "ES"                 # España
TIMEFRAME = "2015-01-01 2025-12-31"
SLEEP_SECONDS = 8          # para reducir riesgo de 429/rate limit

# =========================================================
# HELPERS
# =========================================================
def normalize_brand_filename(brand: str) -> str:
    return (
        brand.lower()
        .replace("&", "and")
        .replace(" ", "_")
        .replace("/", "_")
    )

def fetch_brand_trends(pytrends: TrendReq, brand: str) -> pd.DataFrame:
    pytrends.build_payload(
        kw_list=[brand],
        cat=0,
        timeframe=TIMEFRAME,
        geo=GEO,
        gprop=""
    )

    df = pytrends.interest_over_time()

    if df.empty:
        print(f"[WARN] Sin datos para: {brand}")
        return pd.DataFrame(columns=["fecha", "termino", "valor_trends", "anio", "mes_num", "grupo"])

    # pytrends devuelve una columna con el nombre exacto del término
    df = df.reset_index()

    # a veces aparece 'isPartial'
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    df = df.rename(columns={
        "date": "fecha",
        brand: "valor_trends"
    })

    df["termino"] = brand.lower()
    df["anio"] = pd.to_datetime(df["fecha"]).dt.year
    df["mes_num"] = pd.to_datetime(df["fecha"]).dt.month
    df["grupo"] = "marcas"

    df = df[["fecha", "termino", "valor_trends", "anio", "mes_num", "grupo"]].copy()
    return df


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    pytrends = TrendReq(
        hl="es-ES",
        tz=360,
        retries=3,
        backoff_factor=0.5,
    )

    all_dfs = []

    for brand in BRANDS:
        print(f"Descargando Google Trends para: {brand}")
        try:
            df = fetch_brand_trends(pytrends, brand)

            filename = f"googletrends_{normalize_brand_filename(brand)}_2015_2025.csv"
            output_path = RAW_GT_AGG / filename
            df.to_csv(output_path, index=False)
            print(f"Guardado en: {output_path}")

            all_dfs.append(df)

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"[ERROR] {brand}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)

        combined_output = RAW_GT_AGG / "googletrends_marcas_extra_2015_2025.csv"
        combined.to_csv(combined_output, index=False)
        print(f"\nArchivo combinado guardado en: {combined_output}")

        print("\nPreview:")
        print(combined.head())


if __name__ == "__main__":
    main()