"""Features rolling (janelas moveis) por unit_id."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import ROLLING_WINDOW


def _slope(x: np.ndarray) -> float:
    """Coeficiente angular (slope) de regressao linear nos pontos de uma janela."""
    n = len(x)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=float)
    t_c = t - t.mean()
    x_c = x - x.mean()
    denom = (t_c * t_c).sum()
    return float((t_c * x_c).sum() / denom) if denom > 0 else 0.0


def rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int = ROLLING_WINDOW,
    group_col: str = "unit_id",
) -> pd.DataFrame:
    """Calcula estatisticas rolling por unit_id para cada sensor.

    Para cada sensor em `sensor_cols`, produz as colunas:
      - {sensor}_mean_{window}
      - {sensor}_std_{window}
      - {sensor}_min_{window}
      - {sensor}_max_{window}
      - {sensor}_slope_{window}   (coef angular de regressao linear na janela)
      - {sensor}_delta_baseline   (valor atual - media dos primeiros `window` ciclos)

    Args:
        df: DataFrame com colunas `group_col`, `cycle` e as `sensor_cols`.
            Deve estar ordenado por (group_col, cycle).
        sensor_cols: Nomes dos sensores a agregar.
        window: Tamanho da janela em numero de ciclos. Default: `ROLLING_WINDOW`.
        group_col: Coluna que identifica o motor. Default: "unit_id".

    Returns:
        DataFrame original com as novas colunas concatenadas (sem alterar a ordem).
    """
    result = df.copy()

    for sensor in sensor_cols:
        g = result.groupby(group_col, sort=False)[sensor]
        roll = g.rolling(window=window, min_periods=1)

        result[f"{sensor}_mean_{window}"] = roll.mean().values
        # ddof=0 evita NaN na primeira linha de cada motor
        result[f"{sensor}_std_{window}"] = roll.std(ddof=0).fillna(0.0).values
        result[f"{sensor}_min_{window}"] = roll.min().values
        result[f"{sensor}_max_{window}"] = roll.max().values

        result[f"{sensor}_slope_{window}"] = (
            g.rolling(window=window, min_periods=2)
            .apply(_slope, raw=True)
            .fillna(0.0)
            .values
        )

        # Baseline = media dos primeiros `window` ciclos do motor
        baseline = g.transform(lambda x: x.iloc[:window].mean())
        result[f"{sensor}_delta_baseline"] = result[sensor] - baseline

    return result
