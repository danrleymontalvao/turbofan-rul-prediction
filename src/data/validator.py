"""Validacoes de qualidade de dados C-MAPSS.

TODO: implementar quando iniciar a fase 01_eda.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RelatorioValidacao:
    """Resultado agregado de checagens de qualidade."""

    ok: bool = True
    avisos: list[str] = field(default_factory=list)
    erros: list[str] = field(default_factory=list)


def validar_cmapss(df: pd.DataFrame, split: str = "train") -> RelatorioValidacao:
    """Valida integridade de um DataFrame do C-MAPSS.

    Checagens:
      - unit_id inteiro e sem nulos
      - cycle monotono crescente por unit_id
      - sensor_* sem nulos
      - Para train: pelo menos 1 motor com ciclos suficientes

    Args:
        df: DataFrame carregado via `src.data.loader.load_cmapss`.
        split: "train" ou "test" — diferencia checagens especificas.

    Returns:
        RelatorioValidacao com listas de avisos e erros.
    """
    raise NotImplementedError("A implementar na fase 01_eda")
