"""Constantes centrais do projeto CMAPSS RUL Prediction."""

from pathlib import Path

# Raiz do projeto (2 niveis acima deste arquivo)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Diretorios de dados
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# Diretorios auxiliares
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"

# Reprodutibilidade
SEED: int = 42

# Subsets do C-MAPSS
SUBSETS: list[str] = ["FD001", "FD002", "FD003", "FD004"]

# Schema de colunas do C-MAPSS (nao vem no arquivo)
COLUNAS_META: list[str] = ["unit_id", "cycle"]
COLUNAS_OP_SETTINGS: list[str] = [f"op_setting_{i}" for i in range(1, 4)]
COLUNAS_SENSORES: list[str] = [f"sensor_{i}" for i in range(1, 22)]
COLUNAS_CMAPSS: list[str] = COLUNAS_META + COLUNAS_OP_SETTINGS + COLUNAS_SENSORES

# Sensores constantes (sem variancia) — remover antes de modelar
SENSORES_CONSTANTES: list[int] = [1, 5, 6, 10, 16, 18, 19]

# Piecewise RUL cap (padrao da literatura — Heimes 2008, Zheng 2017)
RUL_CAP: int = 125

# Janela padrao para features rolling
ROLLING_WINDOW: int = 30

# MLflow — backend SQLite (MLflow 3.x deprecou o backend de arquivos)
_MLFLOW_DB: Path = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI: str = f"sqlite:///{_MLFLOW_DB.as_posix()}"
