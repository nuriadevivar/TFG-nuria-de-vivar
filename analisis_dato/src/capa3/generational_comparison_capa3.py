import os
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Rutas
# =====================================================
INPUT_PATH = "analisis_dato/data/input/capa3/capa3_supervised_ready.csv"
OUTPUT_DIR = "analisis_dato/data/analytic/capa3"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# =====================================================
# Carga
# =====================================================
df = pd.read_csv(INPUT_PATH)

print("\n--- INFO GENERAL ---")
print(df.info())

print("\n--- DISTRIBUCIÓN POR GENERACIÓN ---")
print(df["grupo_edad"].value_counts())
print(df["grupo_edad"].value_counts(normalize=True).round(3))

# =====================================================
# Variables a comparar
# =====================================================
variables_clave = [
    "freq_compra_anual",
    "tiempo_rrss_dia",
    "freq_contenido_moda_rrss",
    "sigue_influencers_moda",
    "compra_ult_6m_por_rrss_bin",
    "indice_influencia_rrss",
    "indice_impulso_tendencia",
    "indice_confianza_influencers",
    "indice_escepticismo_influencers",
    "indice_difusion_fastfashion",
    "indice_riesgo_arrepentimiento"
]

# =====================================================
# Tabla de medias por generación
# =====================================================
gen_profile = df.groupby("grupo_edad")[variables_clave].mean().round(3)
gen_profile.to_csv(os.path.join(REPORTS_DIR, "capa3_generational_profile_means.csv"))

print("\n--- MEDIAS POR GENERACIÓN ---")
print(gen_profile)

# =====================================================
# Gráfico 1: variables digitales e influencia
# =====================================================
vars_digital = [
    "indice_influencia_rrss",
    "freq_contenido_moda_rrss",
    "sigue_influencers_moda",
    "tiempo_rrss_dia"
]

plot_df = gen_profile[vars_digital]
plot_df.T.plot(kind="bar", figsize=(12, 6))
plt.title("Comparativa generacional: influencia digital y exposición a moda")
plt.xlabel("Variable")
plt.ylabel("Valor medio")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa3_generational_digital_comparison.png"), dpi=300)
plt.close()

# =====================================================
# Gráfico 2: compra e impulso
# =====================================================
vars_compra = [
    "freq_compra_anual",
    "compra_ult_6m_por_rrss_bin",
    "indice_impulso_tendencia",
    "indice_confianza_influencers",
    "indice_riesgo_arrepentimiento"
]

plot_df = gen_profile[vars_compra]
plot_df.T.plot(kind="bar", figsize=(12, 6))
plt.title("Comparativa generacional: compra, impulso e influencia")
plt.xlabel("Variable")
plt.ylabel("Valor medio")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa3_generational_purchase_comparison.png"), dpi=300)
plt.close()

print("\nComparativa generacional completada.")