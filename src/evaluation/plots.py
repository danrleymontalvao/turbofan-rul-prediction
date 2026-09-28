"""Plots de avaliacao para modelos de RUL.

TODO: implementar na fase 04_evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_predito_vs_real(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Predito vs Real",
) -> None:
    """Scatter com diagonal y=x + banda de erro tipico."""
    raise NotImplementedError("A implementar na fase 04_evaluation")


def plot_residuos(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Histograma dos residuos (y_pred - y_true)."""
    raise NotImplementedError("A implementar na fase 04_evaluation")


def plot_erro_por_faixa_rul(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    n_bins: int = 5,
) -> None:
    """Boxplot do erro absoluto por faixa de RUL verdadeiro."""
    raise NotImplementedError("A implementar na fase 04_evaluation")
