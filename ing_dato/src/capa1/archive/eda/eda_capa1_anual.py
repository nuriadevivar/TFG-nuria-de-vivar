import pandas as pd
import matplotlib.pyplot as plt
from config import DATA_PROCESSED, OUTPUT_FIGURES, OUTPUT_TABLES

# Cargar datos
file_path = DATA_PROCESSED / "capa1_master_anual_analysis.csv"
df = pd.read_csv(file_path)

OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

print("Dimensiones:", df.shape)
print("")
print("Preview:")
print(df)
print("")
print("Descriptivos:")
print(df.describe(include="all"))

# Guardar descriptivos
desc_path = OUTPUT_TABLES / "capa1_anual_descriptivos.csv"
df.describe().to_csv(desc_path)

# ---------- Gráfico 1: RRSS vs compra online ----------
plt.figure(figsize=(10, 6))
plt.plot(df["anio"], df["pct_usuarios_rrss"], marker="o", label="Usuarios RRSS (%)")
plt.plot(df["anio"], df["pct_personas_compra_online"], marker="o", label="Personas que compran online (%)")
plt.title("Evolución del uso de RRSS y la compra online en España")
plt.xlabel("Año")
plt.ylabel("Porcentaje")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / "capa1_anual_rrss_vs_compra_online.png")
plt.close()

# ---------- Gráfico 2: Facturación empresarial online ----------
plt.figure(figsize=(10, 6))
plt.plot(df["anio"], df["pct_facturacion_empresas_online"], marker="o")
plt.title("Porcentaje de facturación empresarial procedente de ventas online")
plt.xlabel("Año")
plt.ylabel("Porcentaje")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / "capa1_anual_facturacion_online_empresas.png")
plt.close()

# ---------- Gráfico 3: Empresas que venden online ----------
plt.figure(figsize=(10, 6))
plt.plot(df["anio"], df["pct_empresas_venden_ecommerce"], marker="o", label="Empresas que venden ecommerce (%)")
plt.plot(df["anio"], df["pct_empresas_venden_web_apps"], marker="o", label="Empresas que venden por web/apps (%)")
plt.title("Adopción empresarial del ecommerce")
plt.xlabel("Año")
plt.ylabel("Porcentaje")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / "capa1_anual_empresas_venden_online.png")
plt.close()

# ---------- Gráfico 4: Peso de ventas ecommerce ----------
plt.figure(figsize=(10, 6))
plt.plot(df["anio"], df["pct_ventas_ecommerce_sobre_total"], marker="o", label="Ecommerce sobre ventas totales (%)")
plt.plot(
    df["anio"],
    df["pct_ventas_ecommerce_sobre_total_empresas_que_venden"],
    marker="o",
    label="Ecommerce sobre ventas totales (empresas que venden) (%)"
)
plt.title("Peso del ecommerce en las ventas empresariales")
plt.xlabel("Año")
plt.ylabel("Porcentaje")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / "capa1_anual_peso_ventas_ecommerce.png")
plt.close()

print("")
print("EDA anual completado.")
print("Descriptivos guardados en:", desc_path)
print("Gráficos guardados en:", OUTPUT_FIGURES)