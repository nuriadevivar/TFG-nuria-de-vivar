import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Rutas
# =====================================================
INPUT_PATH = "analisis_dato/data/input/capa2/instagram_posts_clean.csv"
OUTPUT_DIR = "analisis_dato/data/analytic/capa2"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# =====================================================
# Carga de datos
# =====================================================
df = pd.read_csv(INPUT_PATH)

print("\n--- INFO GENERAL ---")
print(df.info())

print("\n--- PRIMERAS FILAS ---")
print(df.head())

print("\n--- NULOS POR COLUMNA ---")
print(df.isnull().sum())

print("\n--- DUPLICADOS EXACTOS ---")
print(df.duplicated().sum())

# =====================================================
# Limpieza mínima
# =====================================================
# Eliminar duplicados exactos si los hubiera
df = df.drop_duplicates().copy()

# Parsear fecha
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

# =====================================================
# Distribuciones base
# =====================================================
print("\n--- DISTRIBUCIÓN TIPO POST ---")
print(df["tipo_post"].value_counts())

print("\n--- DISTRIBUCIÓN MARCA ---")
print(df["marca"].value_counts())

print("\n--- DESCRIPTIVO ENGAGEMENT ---")
print(df["engagement_total_post"].describe())

# Histograma de engagement
plt.figure(figsize=(10, 5))
plt.hist(df["engagement_total_post"], bins=50)
plt.title("Distribución de engagement_total_post")
plt.xlabel("Engagement total post")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "instagram_engagement_hist.png"), dpi=300)
plt.close()

# Boxplot engagement por marca
plt.figure(figsize=(10, 5))
df.boxplot(column="engagement_total_post", by="marca", grid=False)
plt.title("Engagement por marca")
plt.suptitle("")
plt.xlabel("Marca")
plt.ylabel("Engagement total post")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "instagram_engagement_por_marca.png"), dpi=300)
plt.close()

# Boxplot engagement por tipo_post
plt.figure(figsize=(10, 5))
df.boxplot(column="engagement_total_post", by="tipo_post", grid=False)
plt.title("Engagement por tipo de post")
plt.suptitle("")
plt.xlabel("Tipo de post")
plt.ylabel("Engagement total post")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "instagram_engagement_por_tipo_post.png"), dpi=300)
plt.close()

# =====================================================
# Creación del target
# =====================================================
threshold_p75 = df["engagement_total_post"].quantile(0.75)
df["alto_engagement"] = (df["engagement_total_post"] > threshold_p75).astype(int)

print("\n--- UMBRAL P75 ---")
print(threshold_p75)

print("\n--- DISTRIBUCIÓN TARGET alto_engagement ---")
print(df["alto_engagement"].value_counts())

# Gráfico del target
target_counts = df["alto_engagement"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(target_counts.index.astype(str), target_counts.values)
plt.title("Distribución del target alto_engagement")
plt.xlabel("alto_engagement")
plt.ylabel("Número de posts")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "instagram_target_distribution.png"), dpi=300)
plt.close()

# Alto engagement por marca
target_by_brand = pd.crosstab(df["marca"], df["alto_engagement"], normalize="index")
target_by_brand.to_csv(os.path.join(REPORTS_DIR, "instagram_target_by_brand.csv"))

# Alto engagement por tipo_post
target_by_post = pd.crosstab(df["tipo_post"], df["alto_engagement"], normalize="index")
target_by_post.to_csv(os.path.join(REPORTS_DIR, "instagram_target_by_tipo_post.csv"))

# =====================================================
# Dataset final para modelado
# =====================================================
model_df = df[[
    "marca",
    "anio",
    "mes_num",
    "tipo_post",
    "engagement_total_post",
    "alto_engagement"
]].copy()

model_df.to_csv(os.path.join(OUTPUT_DIR, "instagram_model_input.csv"), index=False)

print("\nDataset final guardado en analytic/capa2/instagram_model_input.csv")
print(model_df.head())