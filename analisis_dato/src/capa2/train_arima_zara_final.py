import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# =====================================================
# Rutas
# =====================================================
INPUT_PATH = "analisis_dato/data/input/capa2/trends_marcas_clean.csv"
OUTPUT_DIR = "analisis_dato/data/analytic/capa2"
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

train = series.loc[: "2024-12-01"]
test = series.loc["2025-01-01":]

print("\n--- TRAIN / TEST ---")
print(f"Train: {train.index.min()} -> {train.index.max()} | n={len(train)}")
print(f"Test:  {test.index.min()} -> {test.index.max()} | n={len(test)}")

# =====================================================
# Modelo final
# =====================================================
model = ARIMA(train, order=(2, 1, 2))
results = model.fit()

with open(os.path.join(REPORTS_DIR, "zara_arima_model_summary.txt"), "w", encoding="utf-8") as f:
    f.write(results.summary().as_text())

print("\n--- RESUMEN MODELO ARIMA ZARA ---")
print(results.summary())

# =====================================================
# Predicción test
# =====================================================
pred = results.forecast(steps=len(test))

mae = mean_absolute_error(test, pred)
rmse = np.sqrt(mean_squared_error(test, pred))
mape = mean_absolute_percentage_error(test, pred)

metrics = {
    "model": "ARIMA(2,1,2)",
    "target": "valor_trends_zara",
    "mae": float(mae),
    "rmse": float(rmse),
    "mape": float(mape)
}

pd.DataFrame([metrics]).to_csv(
    os.path.join(METRICS_DIR, "zara_arima_final_metrics.csv"),
    index=False
)

with open(os.path.join(METRICS_DIR, "zara_arima_final_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=4)

print("\n--- MÉTRICAS FINALES ZARA ---")
print(pd.DataFrame([metrics]))

# =====================================================
# Real vs predicción
# =====================================================
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Test real")
plt.plot(test.index, pred, label="Predicción test")
plt.title("ARIMA Zara - Real vs Predicción")
plt.xlabel("Fecha")
plt.ylabel("Valor Trends")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "zara_arima_real_vs_pred.png"), dpi=300)
plt.close()

# =====================================================
# Residuos
# =====================================================
residuals = results.resid

plt.figure(figsize=(12, 4))
plt.plot(residuals.index, residuals)
plt.axhline(0, linestyle="--")
plt.title("Residuos modelo ARIMA Zara")
plt.xlabel("Fecha")
plt.ylabel("Residuo")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "zara_arima_residuos.png"), dpi=300)
plt.close()

# =====================================================
# Forecast 12 meses
# =====================================================
future_forecast = results.forecast(steps=12)
future_index = pd.date_range(start=series.index.max() + pd.offsets.MonthBegin(1), periods=12, freq="MS")

plt.figure(figsize=(12, 6))
plt.plot(series.index, series, label="Serie histórica")
plt.plot(future_index, future_forecast, label="Forecast 12 meses")
plt.title("Forecast ARIMA Zara (12 meses)")
plt.xlabel("Fecha")
plt.ylabel("Valor Trends")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "zara_arima_forecast_12m.png"), dpi=300)
plt.close()

# =====================================================
# Guardar predicciones
# =====================================================
predictions_df = pd.DataFrame({
    "fecha": test.index,
    "y_true": test.values,
    "y_pred": pred.values,
    "residual": test.values - pred.values
})
predictions_df.to_csv(os.path.join(REPORTS_DIR, "zara_arima_test_predictions.csv"), index=False)

future_df = pd.DataFrame({
    "fecha": future_index,
    "forecast": future_forecast.values
})
future_df.to_csv(os.path.join(REPORTS_DIR, "zara_arima_forecast_12m.csv"), index=False)

print("\nModelo ARIMA final para Zara completado.")