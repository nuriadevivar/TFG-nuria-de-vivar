import pandas as pd
import numpy as np

def clean_percentage(value):
    """
    Convierte porcentajes a escala 0-100.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        if 0 <= value <= 1:
            return float(value * 100)
        return float(value)

    value = str(value).strip().replace("%", "").replace(",", ".")
    if value in ["", "nan", "NaN", "None"]:
        return np.nan

    num = float(value)
    if 0 <= num <= 1:
        return num * 100
    return num


def clean_numeric(value):
    """
    Limpia valores numéricos de tablas estadísticas.
    Convierte a NaN símbolos como ':', '..', '.', etc.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip().replace(",", ".")

    missing_tokens = {"", "nan", "NaN", "None", ":", "..", ".", "...", "-"}
    if value in missing_tokens:
        return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


def ensure_year_int(df, col="anio"):
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df