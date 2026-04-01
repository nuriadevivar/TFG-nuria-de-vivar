from pathlib import Path
import time

import pandas as pd
from pytrends.request import TrendReq


# --- CONFIGURACIÓN BÁSICA ---

pytrends = TrendReq(
    hl="es-ES",
    tz=0
)

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
TIMEFRAME = f"{START_DATE} {END_DATE}"
GEO = "ES"

OUTPUT_FILE = Path(
    r"C:\Users\nuria\OneDrive\Escritorio\ing_dato_nuriaDeVivar\data\raw\capa2\googletrends\aggregated\trends_productos_2015_2025.csv"
)

# Definimos marcas y categorías de forma sistemática
BRANDS = [
    {"brand_search": "zara", "marca": "Zara"},
    {"brand_search": "shein", "marca": "Shein"},
    {"brand_search": "mango", "marca": "Mango"},
    {"brand_search": "h&m", "marca": "H&M"},
    {"brand_search": "massimo dutti", "marca": "Massimo Dutti"},
]

CATEGORIES = ["vestidos", "tops", "zapatos", "pantalones"]


def build_product_terms() -> list[dict]:
    terms = []
    for brand in BRANDS:
        for category in CATEGORIES:
            terms.append(
                {
                    "termino": f"{brand['brand_search']} {category}",
                    "marca": brand["marca"],
                    "categoria_producto": category,
                }
            )
    return terms


TERMINOS_PRODUCTO = build_product_terms()


def descargar_trend(termino: str) -> pd.DataFrame:
    """
    Descarga la serie de Google Trends para un término concreto
    y devuelve un DataFrame con columnas: fecha, interes_trends.
    """
    pytrends.build_payload(
        kw_list=[termino],
        timeframe=TIMEFRAME,
        geo=GEO
    )

    df = pytrends.interest_over_time()

    if df.empty:
        print(f"[AVISO] Serie vacía para término: {termino}")
        return pd.DataFrame(columns=["fecha", "interes_trends"])

    df = df.reset_index()
    df = df.rename(
        columns={
            "date": "fecha",
            termino: "interes_trends",
        }
    )

    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    df = df[["fecha", "interes_trends"]]
    return df


def crear_trends_productos() -> None:
    todas_las_series = []

    for item in TERMINOS_PRODUCTO:
        termino = item["termino"]
        marca = item["marca"]
        categoria = item["categoria_producto"]

        print(f"Descargando: {termino} ({marca} - {categoria})")

        try:
            df_term = descargar_trend(termino)
        except Exception as e:
            print(f"[ERROR] {termino}: {e}")
            continue

        if df_term.empty:
            continue

        df_term["marca"] = marca
        df_term["categoria_producto"] = categoria
        df_term["termino_busqueda"] = termino

        todas_las_series.append(df_term)

        # Pausa para evitar rate limits
        time.sleep(2)

    if not todas_las_series:
        print("No se ha descargado ninguna serie.")
        return

    df_final = pd.concat(todas_las_series, ignore_index=True)
    df_final = df_final.sort_values(["fecha", "marca", "categoria_producto"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"✅ Dataset guardado en: {OUTPUT_FILE}")
    print(df_final.head())


if __name__ == "__main__":
    crear_trends_productos()