"""Modelos baseados em arvores: Random Forest, XGBoost, LightGBM e Stacking."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.utils.config import SEED


def treinar_random_forest(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    **params: Any,
) -> RandomForestRegressor:
    """Treina RandomForestRegressor com seed fixo do projeto.

    Args:
        X_train: Features de treino.
        y_train: RUL alvo.
        **params: Hiperparametros passados diretamente ao RandomForestRegressor.

    Returns:
        Modelo treinado.
    """
    defaults: dict[str, Any] = {"random_state": SEED, "n_jobs": -1}
    defaults.update(params)
    model = RandomForestRegressor(**defaults)
    model.fit(X_train, y_train)
    return model


def treinar_xgboost(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    **params: Any,
) -> XGBRegressor:
    """Treina XGBRegressor com seed fixo e output suprimido.

    Args:
        X_train: Features de treino.
        y_train: RUL alvo.
        **params: Hiperparametros passados diretamente ao XGBRegressor.

    Returns:
        Modelo treinado.
    """
    defaults: dict[str, Any] = {
        "random_state": SEED,
        "n_jobs": -1,
        "eval_metric": "rmse",
        "verbosity": 0,
    }
    defaults.update(params)
    model = XGBRegressor(**defaults)
    model.fit(X_train, y_train)
    return model


def treinar_lightgbm(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    **params: Any,
) -> LGBMRegressor:
    """Treina LGBMRegressor com seed fixo e output suprimido.

    Args:
        X_train: Features de treino.
        y_train: RUL alvo.
        **params: Hiperparametros passados diretamente ao LGBMRegressor.

    Returns:
        Modelo treinado.
    """
    defaults: dict[str, Any] = {
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    defaults.update(params)
    model = LGBMRegressor(**defaults)
    model.fit(X_train, y_train)
    return model


def treinar_stacking(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray,
    rf_params: dict[str, Any] | None = None,
    xgb_params: dict[str, Any] | None = None,
    lgb_params: dict[str, Any] | None = None,
) -> StackingRegressor:
    """Treina StackingRegressor com RF + XGB + LGB e meta-modelo Ridge.

    Usa GroupKFold(5) pre-computado para garantir que motores do treino
    nao vazem para o conjunto de validacao interna do stacking.

    Args:
        X_train: Features de treino.
        y_train: RUL alvo.
        groups: Array de unit_id (um valor por linha) para GroupKFold.
        rf_params: Hiperparametros do Random Forest.
        xgb_params: Hiperparametros do XGBoost.
        lgb_params: Hiperparametros do LightGBM.

    Returns:
        StackingRegressor treinado.
    """
    cv_splits = list(GroupKFold(n_splits=5).split(X_train, y_train, groups))

    estimators = [
        (
            "rf",
            RandomForestRegressor(
                random_state=SEED, n_jobs=1, **(rf_params or {})
            ),
        ),
        (
            "xgb",
            XGBRegressor(
                random_state=SEED,
                n_jobs=1,
                eval_metric="rmse",
                verbosity=0,
                **(xgb_params or {}),
            ),
        ),
        (
            "lgb",
            LGBMRegressor(
                random_state=SEED, n_jobs=1, verbose=-1, **(lgb_params or {})
            ),
        ),
    ]

    model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(),
        cv=cv_splits,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
