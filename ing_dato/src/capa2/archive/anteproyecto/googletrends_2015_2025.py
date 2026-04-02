# -*- coding: utf-8 -*-
"""
Google Trends (ES) robusto con caché y barras de progreso.
- Descarga por término y año (2015–2025) para evitar 429.
- Caché por término-año en /data/trends_cache para poder reanudar.
- Barras de progreso: total, por grupo y por término (años).
"""

import os, time, random
from typing import List, Dict
import pandas as pd
from pytrends.request import TrendReq
from tqdm import tqdm

# ------------------ Config ------------------
GEO = "ES"
START_YEAR, END_YEAR = 2015, 2025
BASE_SLEEP = (3, 6)          # pausas aleatorias entre llamadas
BACKOFF_FACTOR = 2.0          # backoff cuando hay error
MAX_RETRIES = 6               # reintentos por llamada

GRUPOS: Dict[str, List[str]] = {
    "sofisticado": ["cayetana", "pija", "old money", "moda preppy", "minimalista elegante"],
    "urbano": ["choni", "trap style", "y2k outfit", "streetwear", "shein"],
    "consciente_compra": ["moda sostenible", "slow fashion", "ropa vintage", "comprar ropa online", "zara online"],
    "marcas": ["zara", "mango", "pull and bear", "shein", "stradivarius"],
}

CACHE_DIR = "data/trends_cache"
OUT_DIR = "data"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------ Utils ------------------
def jitter(slp=None):
    a, b = slp if slp else BASE_SLEEP
    time.sleep(random.uniform(a, b))

def cache_path(term: str, year: int) -> str:
    safe = term.replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{year}.csv")

def fetch_term_year(term: str, year: int, pytrends: TrendReq) -> pd.DataFrame:
    """Descarga un término para un año con reintentos/backoff y guarda en caché."""
    path = cache_path(term, year)
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"]).set_index("date")

    t0, t1 = f"{year}-01-01", f"{year}-12-31"
    retries, wait = 0, random.uniform(*BASE_SLEEP)
    while True:
        try:
            pytrends.build_payload([term], geo=GEO, timeframe=f"{t0} {t1}")
            df = pytrends.interest_over_time()
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            if df.empty:
                idx = pd.date_range(start=t0, end=t1, freq="D")
                df = pd.DataFrame({term: [float("nan")] * len(idx)}, index=idx)
            df.reset_index().rename(columns={"date": "date"}).to_csv(path, index=False)
            return df
        except KeyboardInterrupt:
            raise
        except Exception:
            retries += 1
            if retries > MAX_RETRIES:
                idx = pd.date_range(start=t0, end=t1, freq="D")
                df = pd.DataFrame({term: [float("nan")] * len(idx)}, index=idx)
                df.reset_index().rename(columns={"index": "date"}).to_csv(path, index=False)
                return df
            time.sleep(wait)
            wait *= BACKOFF_FACTOR  # backoff

def monthly_from_cache(term: str, term_bar: tqdm = None) -> pd.DataFrame:
    """Compone mensual a partir de todos los años (usa tqdm para progreso del término)."""
    pytrends = TrendReq(hl="es-ES", tz=60)
    parts = []
    years = list(range(START_YEAR, END_YEAR + 1))

    for y in years:
        df = fetch_term_year(term, y, pytrends)
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"], errors="ignore")
        parts.append(df)
        jitter()
        if term_bar:
            term_bar.update(1)

    daily = pd.concat(parts).sort_index()
    daily = daily[~daily.index.duplicated(keep="last")]
    monthly = (daily.to_period("M").to_timestamp()
                    .resample("M").mean()
                    .round(2))
    monthly.columns = [term]
    return monthly

def guardar_grupo(nombre: str, terms: List[str], total_bar: tqdm) -> pd.DataFrame:
    frames = []
    # barra por grupo (términos)
    with tqdm(total=len(terms), desc=f"Grupo {nombre}", leave=False) as group_bar:
        for t in terms:
            # barra por término (años)
            with tqdm(total=END_YEAR - START_YEAR + 1,
                      desc=f"  {t}", leave=False) as term_bar:
                m = monthly_from_cache(t, term_bar=term_bar)
            frames.append(m)
            group_bar.update(1)
            total_bar.update(1)
    df = pd.concat(frames, axis=1)
    out = os.path.join(OUT_DIR, f"googletrends_{nombre}_{START_YEAR}_{END_YEAR}.csv")
    df.reset_index().rename(columns={"index": "mes"}).to_csv(out, index=False, encoding="utf-8-sig")
    return df

def to_long(df: pd.DataFrame, grupo: str) -> pd.DataFrame:
    return (df.reset_index().rename(columns={"index": "mes"})
              .melt(id_vars="mes", var_name="termino", value_name="valor")
              .assign(grupo=grupo))

# ------------------ Main ------------------
def main():
    combined = []

    total_terms = sum(len(v) for v in GRUPOS.values())
    try:
        with tqdm(total=total_terms, desc="Progreso total (términos)") as total_bar:
            for nombre, terms in GRUPOS.items():
                df = guardar_grupo(nombre, terms, total_bar)
                combined.append(to_long(df, nombre))
                jitter((10, 18))  # pausa larga entre grupos
    except KeyboardInterrupt:
        print("\n⏹️  Interrumpido por el usuario. Fusionando lo disponible…")
    finally:
        if combined:
            total = (pd.concat(combined, ignore_index=True)
                     .sort_values(["mes", "grupo", "termino"]))
            out_total = os.path.join(OUT_DIR, f"googletrends_moda_total_long_{START_YEAR}_{END_YEAR}.csv")
            total.to_csv(out_total, index=False, encoding="utf-8-sig")
            print(f"📦 Combinado: {out_total}")

if __name__ == "__main__":
    main()
