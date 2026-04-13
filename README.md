# TFG — Transformación de los Hábitos del Consumidor en la Era de la Moda Digital

Análisis del mercado retail, señales digitales y comportamiento del consumidor de fast fashion en España.

**Autora:** Nuria de Vivar Adrada  
**Grado:** Business Analytics · Universidad Francisco de Vitoria  
**Curso:** 2025–2026

---

## Sobre el proyecto

El sector de la moda en España ha cambiado mucho en la última década, impulsado por la digitalización, el auge del fast fashion y el papel creciente de las redes sociales en las decisiones de compra. Este TFG intenta entender esa transformación desde tres ángulos distintos pero complementarios: la evolución del mercado, la presencia digital de las marcas y el comportamiento real del consumidor.

El análisis se estructura en tres capas:

**Capa 1 — Mercado retail:** evolución del índice de comercio minorista de moda en España entre 2015 y 2024, usando datos de Eurostat y fuentes de comercio electrónico. Se aplican modelos SARIMA y Holt-Winters para proyectar la tendencia a 12 meses.

**Capa 2 — Señales digitales:** análisis de la huella digital de Zara, Mango, H&M, Shein y Massimo Dutti a través de Google Trends e Instagram. Incluye text mining sobre captions y modelado ARIMA por marca.

**Capa 3 — Consumidor:** encuesta propia sobre hábitos de compra de fast fashion, segmentada por generación (Gen Z, Millennials, Gen X). Se aplica clustering K-Means y clasificación supervisada (Regresión Logística y Random Forest) para identificar perfiles de consumidor.

---

## Estructura del repositorio

```
TFG-nuria-de-vivar/
│
├── ing_dato/                        # Ingeniería de datos
│   ├── src/
│   │   ├── capa1/                   # ETL mercado retail
│   │   ├── capa2/                   # ETL señales digitales
│   │   ├── capa3/                   # ETL encuesta consumidor
│   │   └── common/                  # Config y utilidades
│   ├── data/
│   │   ├── raw/                     # Datos originales
│   │   ├── processed/               # Datos limpios por capa
│   │   └── databases/               # SQLite (capa1.db, capa2.db, capa3.db)
│   ├── outputs/
│   │   ├── figures/                 # Gráficos EDA y control de calidad
│   │   └/tables/                  # Tablas descriptivas
│   └── requirements.txt
│
├── analisis_dato/                   # Modelado y análisis estadístico
│   ├── src/
│   │   ├── capa1/                   # Modelos temporales
│   │   ├── capa2/                   # Modelos digitales y text mining
│   │   ├── capa3/                   # Clustering y clasificación
│   │   └── common/                  # Tabla maestra de resultados
│   └── data/
│       ├── input/                   # Datasets listos para modelar
│       └── analytic/                # Figuras, métricas e informes
│
└── analisis_de_negocio/             # Interpretación y conclusiones
```

---

## Tecnologías

- **Python 3.11**
- Series temporales: `statsmodels` (SARIMA, ARIMA, Holt-Winters)
- Machine learning: `scikit-learn` (Logistic Regression, Random Forest, K-Means)
- Text mining: TF-IDF con `scikit-learn`
- Visualización: `matplotlib`, `seaborn`
- Datos: `pandas`, `numpy`
- Fuentes: Eurostat, Google Trends (pytrends), Apify (Instagram scraping)
- Base de datos: SQLite

---

## Instalación

```bash
git clone https://github.com/nuriadevivar/TFG-nuria-de-vivar.git
cd TFG-nuria-de-vivar

python -m venv venv
venv\Scripts\activate          # Windows
pip install -r ing_dato/requirements.txt
```

## Orden de ejecución

Cada capa sigue la secuencia `build_` → `transform_` → `eda_`, primero en `ing_dato/src/` y después los modelos en `analisis_dato/src/`.

```
ing_dato:       capa1 → capa2 → capa3
analisis_dato:  capa1 → capa2 → capa3
```

---

## Fuentes de datos

| Fuente | Descripción | Capa |
|--------|-------------|------|
| Eurostat | Índice de comercio minorista de moda y retail total en España | 1 |
| INE / ONTSI | Indicadores de comercio electrónico y digitalización | 1 |
| Google Trends | Búsquedas mensuales de marcas y estéticas (2015–2025) | 2 |
| Apify | Posts de Instagram de las 5 marcas analizadas | 2 |
| Encuesta propia | Hábitos de consumo fast fashion por generación | 3 |

---

## Resultados

Los outputs del análisis están en `analisis_dato/data/analytic/`. La tabla comparativa de todos los modelos se encuentra en:

```
analisis_dato/data/analytic/summary/master_results_table.xlsx
```