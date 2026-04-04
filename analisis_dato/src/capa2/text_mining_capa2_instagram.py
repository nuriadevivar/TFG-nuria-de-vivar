"""
text_mining_capa2_instagram.py — Text Mining sobre captions de Instagram
=========================================================================

MARCO TEÓRICO
-------------
El text mining (minería de texto) es la disciplina que extrae información
estructurada y conocimiento a partir de texto no estructurado mediante técnicas
de Procesamiento del Lenguaje Natural (PLN) y aprendizaje automático
(Feldman & Sanger, 2007).

En el contexto de este TFG, los captions de los posts de Instagram de marcas
de moda constituyen una fuente de señal complementaria al engagement numérico:
el texto comunica la identidad de marca, la llamada a la acción y el tono
emocional del contenido. Este análisis responde a la pregunta:
  ¿Existe relación entre el tipo de lenguaje utilizado en los captions y
  el engagement generado por los posts?

PIPELINE DE TEXT MINING:
  1. Preparación del texto:
     - Normalización: minúsculas, eliminación de URLs, menciones (@), hashtags (#),
       emojis y puntuación. Los hashtags se analizan por separado.
     - Tokenización por espacio (inglés/español mezclados en las marcas analizadas).
     - Eliminación de stopwords (español + inglés) usando NLTK.
     - Longitud del caption como feature numérica.

  2. Análisis exploratorio de texto:
     - Distribución de longitud de captions por marca.
     - WordCloud por marca para visualizar términos más frecuentes.
     - Top N términos por marca (frecuencia absoluta y relativa).
     - Análisis de hashtags más frecuentes por marca.

  3. Modelo de clasificación con features de texto:
     MODELO A — TF-IDF + Regresión Logística (baseline):
       TF-IDF (Term Frequency–Inverse Document Frequency) pondera cada término
       por su frecuencia en el documento y su rareza en el corpus. Penaliza
       términos muy frecuentes que no aportan información discriminativa
       (Salton & Buckley, 1988). Se combina con Regresión Logística para obtener
       un modelo interpretable: los coeficientes indican qué términos son
       predictores positivos/negativos del alto engagement.

     MODELO B — TF-IDF + Random Forest (avanzado):
       Extiende el modelo baseline capturando interacciones entre términos
       que la regresión logística no puede modelar. La feature importance
       revela los términos más predictivos según el criterio de impureza Gini.

  MÉTRICAS: Accuracy, Precision, Recall, F1-Score, ROC-AUC (igual que capa2
  Instagram engagement — permite comparación directa entre modelo con/sin texto).

REFERENCIAS:
  - Feldman, R. & Sanger, J. (2007). The Text Mining Handbook. Cambridge UP.
  - Salton, G. & Buckley, C. (1988). Term-weighting approaches in automatic
    text retrieval. Information Processing & Management, 24(5), 513-523.
  - Jaakonmäki et al. (2017). The Impact of Content, Context, and Creator on
    User Engagement in Social Media Marketing. HICSS.
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
)

warnings.filterwarnings("ignore")

# =====================================================
# Rutas
# =====================================================
INPUT_PATH  = "data/analytic/capa2/instagram_model_input.csv"
RAW_POSTS   = "data/input/capa2/instagram_posts_clean.csv"
OUTPUT_DIR  = "data/analytic/capa2"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

for d in [FIGURES_DIR, METRICS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# =====================================================
# Stopwords combinadas (español + inglés — marcas bilingües)
# =====================================================
STOPWORDS_ES = {
    "de","la","el","en","y","a","los","las","un","una","con","por",
    "para","que","del","se","su","sus","lo","le","al","es","más",
    "pero","como","si","ya","mi","me","te","nos","os","todo","muy",
    "bien","ser","fue","ha","he","hay","este","esta","esto","eso","ese",
    "cuando","también","así","ver","hacer","puede","nuestro","nuestra",
    "tu","tus","vos","vuestro","entre","sobre","hasta","desde","cada",
}
STOPWORDS_EN = {
    "the","a","an","and","or","but","in","on","at","to","for","of",
    "with","is","are","was","were","be","been","have","has","had",
    "this","that","it","its","we","you","our","your","they","their",
    "new","now","all","more","by","from","as","up","out","so","not",
    "can","will","just","my","me","us","do","did","get","go","one",
}
STOPWORDS = STOPWORDS_ES | STOPWORDS_EN

# =====================================================
# Funciones de limpieza
# =====================================================
def clean_text(text: str) -> str:
    if pd.isna(text) or str(text).strip() in ("", "nan"):
        return ""
    t = str(text).lower()
    t = re.sub(r"http\S+|www\S+", " ", t)          # URLs
    t = re.sub(r"@\w+", " ", t)                     # menciones
    t = re.sub(r"#\w+", " ", t)                     # hashtags (se analizan por separado)
    t = re.sub(r"[^\w\sáéíóúüñ]", " ", t)           # puntuación y emojis
    t = re.sub(r"\d+", " ", t)                      # números
    tokens = [w for w in t.split() if len(w) > 2 and w not in STOPWORDS]
    return " ".join(tokens)

def extract_hashtags(text: str) -> list:
    if pd.isna(text):
        return []
    return [h.lower() for h in re.findall(r"#(\w+)", str(text))]

# =====================================================
# Carga de datos
# =====================================================
print("=" * 65)
print("CAPA 2 — TEXT MINING: CAPTIONS DE INSTAGRAM")
print("=" * 65)

# Cargar posts raw para tener los captions completos
try:
    df_raw = pd.read_csv(RAW_POSTS)
    has_caption = "caption" in df_raw.columns
except FileNotFoundError:
    df_raw = None
    has_caption = False

# Cargar dataset de modelado (tiene el target alto_engagement)
df_model = pd.read_csv(INPUT_PATH)

if has_caption and df_raw is not None:
    df = df_raw.copy()

    # Si el target no está en el raw, reconstruirlo con el mismo criterio que prepare_instagram_model.py
    if "alto_engagement" not in df.columns:
        threshold_p75 = df["engagement_total_post"].quantile(0.75)
        df["alto_engagement"] = (df["engagement_total_post"] > threshold_p75).astype(int)

    # Asegurar columnas mínimas necesarias
    if "tipo_post" not in df.columns:
        df["tipo_post"] = ""
    if "anio" not in df.columns:
        df["anio"] = pd.to_datetime(df["fecha"], errors="coerce").dt.year
    if "mes_num" not in df.columns:
        df["mes_num"] = pd.to_datetime(df["fecha"], errors="coerce").dt.month

    # La columna hashtags del raw está vacía; la mantenemos, pero extraeremos hashtags desde caption
    df["caption"] = df["caption"].fillna("").astype(str)
    if "hashtags" not in df.columns:
        df["hashtags"] = ""
    else:
        df["hashtags"] = df["hashtags"].fillna("").astype(str)

else:
    df = df_model.copy()
    df["caption"] = ""
    df["hashtags"] = ""

df["caption"] = df["caption"].fillna("")
df["hashtags"] = df["hashtags"].fillna("")

print(f"\n[Dataset]")
print(f"  Posts totales:  {len(df)}")
print(f"  Con caption:    {(df['caption'].str.strip() != '').sum()}")

hashtags_from_caption = df["caption"].apply(lambda x: len(extract_hashtags(x)) > 0)
print(f"  Con hashtags:   {int(hashtags_from_caption.sum())}")

# Si no hay captions, generar texto sintético a partir de features
# para que el pipeline funcione con cualquier estado de los datos
if (df["caption"] == "").all():
    print("\n  [INFO] Captions no disponibles — construyendo texto a partir de features")
    df["caption_clean"] = (
        df["marca"].fillna("") + " " +
        df["tipo_post"].fillna("") + " " +
        df["marca"].fillna("")
    )
    has_real_text = False
else:
    df["caption_clean"] = df["caption"].apply(clean_text)
    has_real_text = True

print(f"  Captions con contenido tras limpieza: {(df['caption_clean'] != '').sum()}")

# =====================================================
# ANÁLISIS EXPLORATORIO DE TEXTO
# =====================================================

# Longitud de captions
df["caption_len"] = df["caption"].apply(lambda x: len(str(x).split()))
df["caption_clean_len"] = df["caption_clean"].apply(lambda x: len(x.split()) if x else 0)

# Estadísticos de longitud por marca
len_by_brand = df.groupby("marca")["caption_len"].describe().round(2)
len_by_brand.to_csv(os.path.join(REPORTS_DIR, "capa2_text_caption_length_by_brand.csv"))

print(f"\n[Longitud de captions por marca]")
print(len_by_brand[["mean", "50%", "max"]])

# Top términos por marca
top_terms_all = {}
for marca in df["marca"].unique():
    sub = df[df["marca"] == marca]["caption_clean"]
    all_words = " ".join(sub).split()
    counter = Counter(all_words)
    top_terms_all[marca] = counter.most_common(20)

# Guardar top términos
top_terms_records = []
for marca, terms in top_terms_all.items():
    for term, freq in terms:
        top_terms_records.append({"marca": marca, "termino": term, "frecuencia": freq})
pd.DataFrame(top_terms_records).to_csv(
    os.path.join(REPORTS_DIR, "capa2_text_top_terms_by_brand.csv"), index=False
)

# Gráfico top términos por marca
marcas = df["marca"].unique()
n_marcas = len(marcas)
fig, axes = plt.subplots(
    (n_marcas + 1) // 2, 2,
    figsize=(14, 4 * ((n_marcas + 1) // 2))
)
axes = axes.flatten()
for i, marca in enumerate(sorted(marcas)):
    terms = top_terms_all.get(marca, [])
    if terms:
        labels = [t[0] for t in terms[:10]][::-1]
        freqs  = [t[1] for t in terms[:10]][::-1]
        axes[i].barh(labels, freqs, color="#2196F3")
        axes[i].set_title(f"Top términos — {marca}")
        axes[i].set_xlabel("Frecuencia")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Top 10 términos en captions por marca (tras limpieza)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa2_text_top_terms_by_brand.png"), dpi=300)
plt.close()

# =====================================================
# Top hashtags por marca
# =====================================================
hashtag_records = []
for marca in sorted(df["marca"].unique()):
    sub = df[df["marca"] == marca]["caption"]
    all_tags = []
    for text in sub:
        all_tags.extend(extract_hashtags(str(text)))
    counter = Counter(all_tags)
    for tag, freq in counter.most_common(15):
        hashtag_records.append({"marca": marca, "hashtag": tag, "frecuencia": freq})

hashtag_df = pd.DataFrame(hashtag_records)
hashtag_df.to_csv(os.path.join(REPORTS_DIR, "capa2_text_top_hashtags_by_brand.csv"), index=False)

if not hashtag_df.empty:
    print(f"\n[Top hashtags globales]")
    global_tags = hashtag_df.groupby("hashtag")["frecuencia"].sum().sort_values(ascending=False).head(10)
    print(global_tags)

# =====================================================
# MODELO CON FEATURES DE TEXTO
# =====================================================
print("\n" + "-" * 50)
print("MODELOS DE CLASIFICACIÓN CON TF-IDF (text features)")
print("-" * 50)

# Filtrar posts con texto limpio no vacío
df_model_text = df[df["caption_clean"].str.strip() != ""].copy()
if len(df_model_text) < 50:
    print(f"\n  [WARN] Solo {len(df_model_text)} posts con texto válido.")
    print("  Se usa el corpus completo con texto sintetizado de features.")
    df_model_text = df.copy()
    df_model_text["caption_clean"] = df_model_text["caption_clean"].replace("", "sin_caption")

X_text = df_model_text["caption_clean"]
y_text = df_model_text["alto_engagement"].astype(int)

print(f"  Posts para modelado: {len(df_model_text)}")
print(f"  Clase 1 (alto engagement): {y_text.sum()} ({y_text.mean()*100:.1f}%)")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_text, y_text, test_size=0.2, random_state=42, stratify=y_text
)

tfidf_params = dict(
    max_features=500,
    ngram_range=(1, 2),     # unigramas + bigramas
    min_df=2,               # término en al menos 2 documentos
    sublinear_tf=True,      # log(TF) para suavizar términos muy frecuentes
)

text_models = {
    "A_TFIDF_LogisticRegression": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ]),
    "B_TFIDF_RandomForest": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   RandomForestClassifier(
            n_estimators=300, max_depth=10,
            class_weight="balanced", random_state=42
        )),
    ]),
}

text_results = []

for mname, mpipe in text_models.items():
    mpipe.fit(X_tr, y_tr)

    y_tr_pred = mpipe.predict(X_tr)
    y_tr_prob = mpipe.predict_proba(X_tr)[:, 1]
    y_te_pred = mpipe.predict(X_te)
    y_te_prob = mpipe.predict_proba(X_te)[:, 1]

    res = {
        "model":            mname,
        "train_f1":         f1_score(y_tr, y_tr_pred, zero_division=0),
        "train_auc":        roc_auc_score(y_tr, y_tr_prob),
        "test_accuracy":    accuracy_score(y_te, y_te_pred),
        "test_precision":   precision_score(y_te, y_te_pred, zero_division=0),
        "test_recall":      recall_score(y_te, y_te_pred, zero_division=0),
        "test_f1":          f1_score(y_te, y_te_pred, zero_division=0),
        "test_auc":         roc_auc_score(y_te, y_te_prob),
        "gap_f1":           f1_score(y_tr, y_tr_pred, zero_division=0) - f1_score(y_te, y_te_pred, zero_division=0),
        "gap_auc":          roc_auc_score(y_tr, y_tr_prob) - roc_auc_score(y_te, y_te_prob),
    }
    text_results.append(res)

    print(f"\n  [{mname}]")
    print(f"  Test → Acc={res['test_accuracy']:.3f} | F1={res['test_f1']:.3f} | "
          f"AUC={res['test_auc']:.3f}")
    gap_flag = "⚠ SOBREAJUSTE" if res["gap_f1"] > 0.1 else "✓ OK"
    print(f"  Gap F1={res['gap_f1']:.3f} | {gap_flag}")

    # Matriz de confusión
    cm = confusion_matrix(y_te, y_te_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bajo", "Alto"])
    disp.plot(colorbar=False)
    plt.title(f"Matriz de Confusión — {mname}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"capa2_text_{mname}_confusion_matrix.png"), dpi=300)
    plt.close()

    # ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_te, y_te_prob, ax=ax, name=mname)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title(f"Curva ROC — {mname}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"capa2_text_{mname}_roc_curve.png"), dpi=300)
    plt.close()

# Top términos predictivos (coeficientes Regresión Logística)
lr_pipe = text_models["A_TFIDF_LogisticRegression"]
vocab   = lr_pipe.named_steps["tfidf"].get_feature_names_out()
coefs   = lr_pipe.named_steps["clf"].coef_[0]

coef_df = pd.DataFrame({"term": vocab, "coef": coefs}) \
            .sort_values("coef", ascending=False)

top_pos = coef_df.head(15)
top_neg = coef_df.tail(15)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.barh(top_pos["term"][::-1], top_pos["coef"][::-1], color="#2ca02c")
ax1.set_title("Términos → Alto engagement")
ax1.set_xlabel("Coeficiente logístico")
ax2.barh(top_neg["term"], top_neg["coef"], color="#d62728")
ax2.set_title("Términos → Bajo engagement")
ax2.set_xlabel("Coeficiente logístico")
plt.suptitle("TF-IDF + Regresión Logística — Términos más predictivos", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa2_text_lr_coeficientes.png"), dpi=300)
plt.close()

coef_df.to_csv(os.path.join(REPORTS_DIR, "capa2_text_lr_coeficientes.csv"), index=False)

# Guardar métricas text mining
text_results_df = pd.DataFrame(text_results).sort_values("test_auc", ascending=False)
text_results_df.to_csv(os.path.join(METRICS_DIR, "capa2_text_modelos_comparativa.csv"), index=False)
with open(os.path.join(METRICS_DIR, "capa2_text_modelos_comparativa.json"), "w", encoding="utf-8") as f:
    json.dump(text_results, f, ensure_ascii=False, indent=4)

# =====================================================
# Interpretación y conclusiones
# =====================================================
text_winner = text_results_df.iloc[0]
interpretation = {
    "objetivo": "Predecir alto engagement a partir del texto del caption",
    "modelo_ganador": text_winner["model"],
    "test_auc": text_winner["test_auc"],
    "test_f1":  text_winner["test_f1"],
    "tiene_texto_real": has_real_text,
    "interpretacion": (
        "El análisis TF-IDF sobre los captions de Instagram identifica los términos "
        "más asociados al alto engagement. Los coeficientes de la regresión logística "
        "son directamente interpretables: términos con coeficiente positivo predicen "
        "alto engagement, mientras que los negativos predicen bajo engagement. "
        "Este análisis complementa los modelos de features estructuradas al incorporar "
        "la dimensión semántica del contenido publicado."
    ),
    "comparacion_con_features_estructuradas": (
        "Comparar el AUC de los modelos TF-IDF con el AUC de los modelos de features "
        "estructuradas (marca, tipo_post, mes) permite cuantificar el valor añadido "
        "del contenido textual en la predicción del engagement."
    ),
}

with open(os.path.join(REPORTS_DIR, "capa2_text_mining_interpretacion.json"), "w", encoding="utf-8") as f:
    json.dump(interpretation, f, ensure_ascii=False, indent=4)

print("\n" + "=" * 65)
print("CAPA 2 — TEXT MINING COMPLETADO")
print(f"  Modelo A: TF-IDF + Logistic Regression | AUC={text_results[0]['test_auc']:.3f}")
print(f"  Modelo B: TF-IDF + Random Forest       | AUC={text_results[1]['test_auc']:.3f}")
print(f"  Ganador:  {text_winner['model']}")
print("Outputs:")
print("  figures/ → top_terms_by_brand, lr_coeficientes, confusion_matrix, roc_curve")
print("  metrics/ → capa2_text_modelos_comparativa.csv/.json")
print("  reports/ → top_terms, top_hashtags, lr_coeficientes, interpretacion")
print("=" * 65)