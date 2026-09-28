"""Metricas de avaliacao para RUL prediction.

Referencias:
  - Saxena et al. (2008), "Damage Propagation Modeling for Aircraft Engine
    RUL Estimation" (definicao do PHM Score).
"""

from __future__ import annotations

import numpy as np


def _as_array(x: object) -> np.ndarray:
    return np.asarray(x, dtype=float)


def rmse(y_true: object, y_pred: object) -> float:
    """Root Mean Squared Error."""
    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mae(y_true: object, y_pred: object) -> float:
    """Mean Absolute Error."""
    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def phm_score(y_true: object, y_pred: object) -> float:
    """PHM Score (Saxena 2008).

    Penaliza previsoes tardias (predito > real) exponencialmente mais que
    adiantadas (predito < real):
      - Se d < 0 (adiantado): s = exp(-d/13) - 1
      - Se d >= 0 (atrasado): s = exp(d/10)  - 1
    onde d = predito - real.

    Menor e melhor. Score = 0 significa previsao perfeita.

    Args:
        y_true: RUL verdadeiro.
        y_pred: RUL predito.

    Returns:
        Soma do score por amostra.

    Examples:
        >>> round(phm_score([50, 50], [50, 50]), 6)
        0.0
        >>> round(phm_score([50], [40]), 4)  # adiantado em 10
        1.1503
        >>> round(phm_score([50], [60]), 4)  # atrasado em 10
        1.7183
    """
    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)
    d = y_pred_arr - y_true_arr

    scores = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1.0,
        np.exp(d / 10.0) - 1.0,
    )
    return float(np.sum(scores))
