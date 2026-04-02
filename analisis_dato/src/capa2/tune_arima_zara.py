import os
import json
import warnings
import itertools
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# =====================================================
# Rutas
# =====================================================
INPUT_PATH = "analisis_dato/data/input/capa2/trends_marcas_clean.csv"
OUTPUT_DIR = "analisis_dato/data/analytic/capa2"
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# =====================================================
# Función MAPE
# =====================================================
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# =====================================================
# Carga y filtro Zara
# =====================================================
df = pd.read_csv(INPUT_PATH)
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

zara = (
    df[df["termino"] == "zara"]
    .copy()
    .sort_values("fecha")
    .set_index("fecha")
    .asfreq("MS")
)

series = zara["valor_trends"].dropna()

print("\n--- INFO ZARA ---")
print(series.head())
print(f"Inicio: {series.index.min()}")
print(f"Fin:    {series.index.max()}")
print(f"N observaciones: {len(series)}")

# =====================================================
# Split temporal
# =====================================================
train = series.loc[: "2024-12-01"]
test = series.loc["2025-01-01":]

print("\n--- TRAIN / TEST ---")
print(f"Train: {train.index.min()} -> {train.index.max()} | n={len(train)}")
print(f"Test:  {test.index.min()} -> {test.index.max()} | n={len(test)}")

# =====================================================
# Grid ARIMA
# =====================================================
p = [0, 1, 2]
d = [1]
q = [0, 1, 2]

param_grid = list(itertools.product(p, d, q))
results_list = []

print(f"\nModelos a probar: {len(param_grid)}")

for idx, (p_, d_, q_) in enumerate(param_grid, start=1):
    order = (p_, d_, q_)
    try:
        model = ARIMA(train, order=order)
        fitted = model.fit()

        pred = fitted.forecast(steps=len(test))

        mae = mean_absolute_error(test, pred)
        rmse = np.sqrt(mean_squared_error(test, pred))
        mape = mean_absolute_percentage_error(test, pred)

        results_list.append({
            "order": str(order),
            "aic": fitted.aic,
            "bic": fitted.bic,
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        })

        print(f"[{idx}/{len(param_grid)}] OK -> order={order}, "
              f"AIC={fitted.aic:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}")

    except Exception as e:
        print(f"[{idx}/{len(param_grid)}] ERROR -> order={order} | {e}")

# =====================================================
# Resultados
# =====================================================
results_df = pd.DataFrame(results_list)
results_df_sorted = results_df.sort_values(by=["rmse", "mape", "aic"]).reset_index(drop=True)

results_df.to_csv(os.path.join(REPORTS_DIR, "zara_arima_tuning_all_results.csv"), index=False)
results_df_sorted.to_csv(os.path.join(REPORTS_DIR, "zara_arima_tuning_sorted_results.csv"), index=False)

top10 = results_df_sorted.head(10)
top10.to_csv(os.path.join(METRICS_DIR, "zara_arima_top10.csv"), index=False)

with open(os.path.join(METRICS_DIR, "zara_arima_top10.json"), "w", encoding="utf-8") as f:
    json.dump(top10.to_dict(orient="records"), f, ensure_ascii=False, indent=4)

print("\n--- TOP MODELOS ZARA ---")
print(top10)

if not results_df_sorted.empty:
    best_model = results_df_sorted.iloc[0].to_dict()
    with open(os.path.join(REPORTS_DIR, "zara_best_arima_candidate.json"), "w", encoding="utf-8") as f:
        json.dump(best_model, f, ensure_ascii=False, indent=4)

    print("\n--- MEJOR CANDIDATO ZARA ---")
    print(best_model)
else:
    print("\nNo se ha podido ajustar ningún modelo.")