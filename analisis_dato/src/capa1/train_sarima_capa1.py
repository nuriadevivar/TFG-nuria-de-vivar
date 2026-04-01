import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# =====================================================
# Rutas
# =====================================================
INPUT_PATH = "analisis_dato/data/input/capa1/capa1_master_mensual_analysis.csv"
OUTPUT_DIR = "analisis_dato/data/analytic/capa1"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

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

# =====================================================
# Split temporal: train hasta 2022, test = 2023
# =====================================================
train = series.loc[: "2022-12-01"]
test = series.loc["2023-01-01":]

print("\n--- TRAIN / TEST ---")
print(f"Train: {train.index.min()} -> {train.index.max()} | n={len(train)}")
print(f"Test:  {test.index.min()} -> {test.index.max()} | n={len(test)}")

# =====================================================
# Modelo SARIMA inicial
# =====================================================
# Propuesta inicial razonable:
# order=(1,1,1)
# seasonal_order=(1,1,1,12)
# Si luego queremos, afinamos.
model = SARIMAX(
    train,
    order=(0, 1, 1),
    seasonal_order=(0, 1, 0, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)

# Resumen del modelo
with open(os.path.join(REPORTS_DIR, "sarima_model_summary.txt"), "w", encoding="utf-8") as f:
    f.write(results.summary().as_text())

print("\n--- RESUMEN MODELO SARIMA ---")
print(results.summary())

# =====================================================
# Predicción sobre test
# =====================================================
forecast_test = results.get_forecast(steps=len(test))
pred_mean = forecast_test.predicted_mean
pred_ci = forecast_test.conf_int()

# =====================================================
# Métricas
# =====================================================
mae = mean_absolute_error(test, pred_mean)
rmse = np.sqrt(mean_squared_error(test, pred_mean))
mape = mean_absolute_percentage_error(test, pred_mean)

metrics = {
    "model": "SARIMA(0,1,1)(0,1,0,12)",
    "target": target_col,
    "mae": float(mae),
    "rmse": float(rmse),
    "mape": float(mape)
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(os.path.join(METRICS_DIR, "capa1_sarima_metrics.csv"), index=False)

with open(os.path.join(METRICS_DIR, "capa1_sarima_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=4)

print("\n--- MÉTRICAS ---")
print(metrics_df)

# =====================================================
# Gráfico real vs predicción en test
# =====================================================
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Test real")
plt.plot(pred_mean.index, pred_mean, label="Predicción test")
plt.fill_between(
    pred_ci.index,
    pred_ci.iloc[:, 0],
    pred_ci.iloc[:, 1],
    color="gray",
    alpha=0.2,
    label="IC 95%"
)
plt.title("SARIMA - Real vs Predicción (test)")
plt.xlabel("Fecha")
plt.ylabel(target_col)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa1_sarima_real_vs_pred.png"), dpi=300)
plt.close()

# =====================================================
# Residuos del modelo en train
# =====================================================
residuals = results.resid

plt.figure(figsize=(12, 4))
plt.plot(residuals.index, residuals)
plt.axhline(0, linestyle="--")
plt.title("Residuos del modelo SARIMA")
plt.xlabel("Fecha")
plt.ylabel("Residuo")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa1_sarima_residuos.png"), dpi=300)
plt.close()

# =====================================================
# Forecast corto futuro (12 meses)
# =====================================================
future_forecast = results.get_forecast(steps=12)
future_mean = future_forecast.predicted_mean
future_ci = future_forecast.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(series.index, series, label="Serie histórica")
plt.plot(future_mean.index, future_mean, label="Forecast 12 meses")
plt.fill_between(
    future_ci.index,
    future_ci.iloc[:, 0],
    future_ci.iloc[:, 1],
    color="gray",
    alpha=0.2,
    label="IC 95%"
)
plt.title("Forecast corto SARIMA (12 meses)")
plt.xlabel("Fecha")
plt.ylabel(target_col)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa1_sarima_forecast_12m.png"), dpi=300)
plt.close()

# =====================================================
# Guardar predicciones
# =====================================================
predictions_df = pd.DataFrame({
    "fecha": test.index,
    "y_true": test.values,
    "y_pred": pred_mean.values,
    "residual": test.values - pred_mean.values
})
predictions_df.to_csv(os.path.join(REPORTS_DIR, "capa1_sarima_test_predictions.csv"), index=False)

future_df = pd.DataFrame({
    "fecha": future_mean.index,
    "forecast": future_mean.values,
    "lower_ci": future_ci.iloc[:, 0].values,
    "upper_ci": future_ci.iloc[:, 1].values
})
future_df.to_csv(os.path.join(REPORTS_DIR, "capa1_sarima_forecast_12m.csv"), index=False)

print("\nModelo SARIMA completado.")