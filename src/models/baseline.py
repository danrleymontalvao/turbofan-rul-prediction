"""Baseline linear para RUL prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.utils.config import SEED


def treinar_baseline_linear(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    alpha: float = 1.0,
) -> Ridge:
    """Treina Ridge regression como baseline.

    Args:
        X_train: Features de treino.
        y_train: RUL alvo (piecewise capped).
        alpha: Regularizacao L2. Default: 1.0.

    Returns:
        Modelo Ridge treinado.
    """
    model = Ridge(alpha=alpha, random_state=SEED)
    model.fit(X_train, y_train)
    return model
