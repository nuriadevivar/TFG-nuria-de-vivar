from pathlib import Path

# =========================
# ROOT
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
ANALYTIC_DIR = DATA_DIR / "analytic"
OUTPUTS_DIR = BASE_DIR / "outputs"

# =========================
# INPUTS
# =========================

INPUT_CAPA1 = INPUT_DIR / "capa1"
INPUT_CAPA2 = INPUT_DIR / "capa2"
INPUT_CAPA3 = INPUT_DIR / "capa3"

# =========================
# ANALYTIC
# =========================

ANALYTIC_CAPA1 = ANALYTIC_DIR / "capa1"
ANALYTIC_CAPA2 = ANALYTIC_DIR / "capa2"
ANALYTIC_CAPA3 = ANALYTIC_DIR / "capa3"

# =========================
# OUTPUTS
# =========================

FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"

FIGURES_CAPA1 = FIGURES_DIR / "capa1"
FIGURES_CAPA2 = FIGURES_DIR / "capa2"
FIGURES_CAPA3 = FIGURES_DIR / "capa3"

METRICS_CAPA1 = METRICS_DIR / "capa1"
METRICS_CAPA2 = METRICS_DIR / "capa2"
METRICS_CAPA3 = METRICS_DIR / "capa3"

TABLES_CAPA1 = TABLES_DIR / "capa1"
TABLES_CAPA2 = TABLES_DIR / "capa2"
TABLES_CAPA3 = TABLES_DIR / "capa3"

MODELS_CAPA1 = MODELS_DIR / "capa1"
MODELS_CAPA2 = MODELS_DIR / "capa2"
MODELS_CAPA3 = MODELS_DIR / "capa3"