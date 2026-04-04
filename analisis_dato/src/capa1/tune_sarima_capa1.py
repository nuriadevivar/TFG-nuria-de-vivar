import os
import json
import warnings
import itertools
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# =====================================================
# Rutas
# =====================================================
INPUT_PATH = "data/input/capa1/capa1_master_mensual_analysis.csv"
OUTPUT_DIR = "data/analytic/capa1"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
# =====================================================
# Función MAPE
# =====================================================
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# =====================================================
# Carga y preparación
# =====================================================
df = pd.read_csv(INPUT_PATH)
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").set_index("fecha").asfreq("MS")

target_col = "indice_retail_moda"
series = df[target_col]

train = series.loc[: "2022-12-01"]
test = series.loc["2023-01-01":]

print("\n--- TRAIN / TEST ---")
print(f"Train: {train.index.min()} -> {train.index.max()} | n={len(train)}")
print(f"Test:  {test.index.min()} -> {test.index.max()} | n={len(test)}")

# =====================================================
# Grid de búsqueda razonable
# =====================================================
p = [0, 1, 2]
d = [0, 1]
q = [0, 1, 2]

P = [0, 1]
D = [0, 1]
Q = [0, 1]

seasonal_period = 12

param_grid = list(itertools.product(p, d, q, P, D, Q))

results_list = []

print(f"\nModelos a probar: {len(param_grid)}")

for idx, (p_, d_, q_, P_, D_, Q_) in enumerate(param_grid, start=1):
    order = (p_, d_, q_)
    seasonal_order = (P_, D_, Q_, seasonal_period)

    try:
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted = model.fit(disp=False)

        forecast_test = fitted.get_forecast(steps=len(test))
        pred_mean = forecast_test.predicted_mean

        mae = mean_absolute_error(test, pred_mean)
        rmse = np.sqrt(mean_squared_error(test, pred_mean))
        mape = mean_absolute_percentage_error(test, pred_mean)

        results_list.append({
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            "aic": fitted.aic,
            "bic": fitted.bic,
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        })

        print(f"[{idx}/{len(param_grid)}] OK -> order={order}, seasonal={seasonal_order}, "
              f"AIC={fitted.aic:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}")

    except Exception as e:
        print(f"[{idx}/{len(param_grid)}] ERROR -> order={order}, seasonal={seasonal_order} | {e}")

# =====================================================
# Resultados
# =====================================================
results_df = pd.DataFrame(results_list)

# Orden por RMSE y luego AIC
results_df_sorted = results_df.sort_values(by=["rmse", "mape", "aic"]).reset_index(drop=True)

results_df.to_csv(os.path.join(REPORTS_DIR, "capa1_sarima_tuning_all_results.csv"), index=False)
results_df_sorted.to_csv(os.path.join(REPORTS_DIR, "capa1_sarima_tuning_sorted_results.csv"), index=False)

top10 = results_df_sorted.head(10)
top10.to_csv(os.path.join(METRICS_DIR, "capa1_sarima_top10.csv"), index=False)

with open(os.path.join(METRICS_DIR, "capa1_sarima_top10.json"), "w", encoding="utf-8") as f:
    json.dump(top10.to_dict(orient="records"), f, ensure_ascii=False, indent=4)

print("\n--- TOP 10 MODELOS ---")
print(top10)

# Guardar mejor modelo en un txt
if not results_df_sorted.empty:
    best_model = results_df_sorted.iloc[0].to_dict()
    with open(os.path.join(REPORTS_DIR, "capa1_best_sarima_candidate.json"), "w", encoding="utf-8") as f:
        json.dump(best_model, f, ensure_ascii=False, indent=4)

    print("\n--- MEJOR CANDIDATO ---")
    print(best_model)
else:
    print("\nNo se ha podido ajustar ningún modelo.")