"""Helper para inicializar experimentos MLflow com configuracao consistente."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

import mlflow

# Suprime warnings de Git ausente (sem impacto funcional no tracking)
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from src.utils.config import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)


def configurar_tracking(tracking_uri: str = MLFLOW_TRACKING_URI) -> None:
    """Configura o URI de tracking do MLflow.

    Chamar uma vez no inicio de cada script/notebook. Por default aponta para
    `mlruns/` na raiz do projeto.
    """
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)


@contextmanager
def start_run(
    experiment: str,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> Iterator[mlflow.ActiveRun]:
    """Context manager para iniciar um experimento MLflow com padroes do projeto.

    Args:
        experiment: Nome do experimento (ex: "cmapss_baseline").
        run_name: Nome opcional da execucao.
        tags: Tags adicionais para a execucao (ex: {"model": "xgboost", "subset": "FD001"}).
        tracking_uri: URI de tracking. Default: `mlruns/` local.

    Yields:
        `mlflow.ActiveRun` — a execucao ativa.

    Example:
        >>> with start_run("cmapss_baseline", tags={"model": "linear"}) as run:
        ...     mlflow.log_param("alpha", 1.0)
        ...     mlflow.log_metric("rmse", 15.3)
    """
    configurar_tracking(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run
