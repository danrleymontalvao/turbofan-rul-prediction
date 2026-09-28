"""Normalizacao de features por regime operacional.

Critico para FD002 e FD004 (multi-regime). Para FD001/FD003, usar n_regimes=1
que aplica um StandardScaler global.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils.config import COLUNAS_OP_SETTINGS


def _default_op_cols() -> list[str]:
    return list(COLUNAS_OP_SETTINGS)


@dataclass
class RegimeNormalizer:
    """Normalizador z-score por cluster de operating settings.

    Para n_regimes=1 (FD001/FD003), aplica um StandardScaler global.
    Para n_regimes=6 (FD002/FD004), usa KMeans para identificar regimes e
    aplica um StandardScaler independente por regime.

    Attributes:
        n_regimes: Numero de clusters. Use 1 para FD001/FD003, 6 para FD002/FD004.
        op_cols: Colunas usadas para clusterizar. Default: op_setting_1..3.
    """

    n_regimes: int = 1
    op_cols: list[str] | None = None

    # Atributos populados por fit()
    _scalers: dict[int, StandardScaler] = field(
        default_factory=dict, init=False, repr=False
    )
    _kmeans: KMeans | None = field(default=None, init=False, repr=False)
    _sensor_cols: list[str] = field(default_factory=list, init=False, repr=False)

    def _resolve_op_cols(self) -> list[str]:
        return self.op_cols if self.op_cols is not None else _default_op_cols()

    def _predict_regime(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_regimes == 1:
            return np.zeros(len(df), dtype=int)
        assert self._kmeans is not None
        return self._kmeans.predict(df[self._resolve_op_cols()].values)

    def fit(self, df: pd.DataFrame, sensor_cols: list[str]) -> "RegimeNormalizer":
        """Aprende clusters e estatisticas por regime.

        Args:
            df: DataFrame de treino com colunas de op_settings e sensores.
            sensor_cols: Colunas de sensores a normalizar.

        Returns:
            Self (para encadeamento fit().transform()).
        """
        self._sensor_cols = list(sensor_cols)

        if self.n_regimes > 1:
            self._kmeans = KMeans(
                n_clusters=self.n_regimes, random_state=42, n_init="auto"
            )
            self._kmeans.fit(df[self._resolve_op_cols()].values)

        regimes = self._predict_regime(df)
        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            scaler = StandardScaler()
            scaler.fit(df.loc[mask, self._sensor_cols].values)
            self._scalers[regime_id] = scaler

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica normalizacao z-score dentro de cada regime.

        Args:
            df: DataFrame com as mesmas colunas usadas em fit().

        Returns:
            Copia do DataFrame com as colunas de sensor normalizadas.
        """
        result = df.copy()
        regimes = self._predict_regime(result)

        # Garante que as colunas de sensor sao float antes de atribuir valores normalizados
        result[self._sensor_cols] = result[self._sensor_cols].astype(float)

        for regime_id, scaler in self._scalers.items():
            mask = regimes == regime_id
            if mask.sum() == 0:
                continue
            result.loc[mask, self._sensor_cols] = scaler.transform(
                result.loc[mask, self._sensor_cols].values
            )

        return result

    def fit_transform(self, df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        return self.fit(df, sensor_cols).transform(df)
