from pathlib import Path

# =========================
# ROOT
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
DOCS_DIR = BASE_DIR / "docs"


# =========================
# DATA
# =========================

DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_DATABASES = DATA_DIR / "databases"


# =========================
# OUTPUTS
# =========================

OUTPUT_TABLES = OUTPUTS_DIR / "tables"
OUTPUT_FIGURES = OUTPUTS_DIR / "figures"


# =========================
# CAPA 1
# =========================

RAW_CAPA1 = DATA_RAW / "capa1"
PROCESSED_CAPA1 = DATA_PROCESSED / "capa1"
DB_CAPA1 = DATA_DATABASES / "capa1.db"

TABLES_CAPA1 = OUTPUT_TABLES / "capa1"
TABLES_CAPA1_CONTROL = TABLES_CAPA1 / "control"
TABLES_CAPA1_MASTERS = TABLES_CAPA1 / "masters"
TABLES_CAPA1_EDA = TABLES_CAPA1 / "eda"

FIGURES_CAPA1 = OUTPUT_FIGURES / "capa1"
FIGURES_CAPA1_CONTROL = FIGURES_CAPA1 / "control"
FIGURES_CAPA1_EDA = FIGURES_CAPA1 / "eda"


# =========================
# CAPA 2
# =========================

RAW_CAPA2 = DATA_RAW / "capa2"
PROCESSED_CAPA2 = DATA_PROCESSED / "capa2"
DB_CAPA2 = DATA_DATABASES / "capa2.db"

TABLES_CAPA2 = OUTPUT_TABLES / "capa2"
TABLES_CAPA2_CONTROL = TABLES_CAPA2 / "control"
TABLES_CAPA2_MASTERS = TABLES_CAPA2 / "masters"
TABLES_CAPA2_EDA = TABLES_CAPA2 / "eda"

FIGURES_CAPA2 = OUTPUT_FIGURES / "capa2"
FIGURES_CAPA2_CONTROL = FIGURES_CAPA2 / "control"
FIGURES_CAPA2_EDA = FIGURES_CAPA2 / "eda"


# =========================
# CAPA 3
# =========================

RAW_CAPA3 = DATA_RAW / "capa3"
PROCESSED_CAPA3 = DATA_PROCESSED / "capa3"
DB_CAPA3 = DATA_DATABASES / "capa3.db"

TABLES_CAPA3 = OUTPUT_TABLES / "capa3"
TABLES_CAPA3_CONTROL = TABLES_CAPA3 / "control"
TABLES_CAPA3_MASTERS = TABLES_CAPA3 / "masters"
TABLES_CAPA3_EDA = TABLES_CAPA3 / "eda"

FIGURES_CAPA3 = OUTPUT_FIGURES / "capa3"
FIGURES_CAPA3_CONTROL = FIGURES_CAPA3 / "control"
FIGURES_CAPA3_EDA = FIGURES_CAPA3 / "eda"