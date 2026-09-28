"""Piecewise RUL cap — trata os primeiros ciclos como RUL constante.

Justificativa: nas fases iniciais de vida os sensores nao carregam informacao
util sobre a falha. Modelar RUL alto como constante evita que o modelo tente
aprender ruido nesse regime. Padrao da literatura (Heimes 2008, Zheng 2017).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import RUL_CAP


def apply_piecewise_rul(
    rul: pd.Series | np.ndarray,
    cap: int = RUL_CAP,
) -> pd.Series | np.ndarray:
    """Aplica teto linear ao RUL.

    Args:
        rul: Serie ou array com valores de RUL derivados (cycles restantes).
        cap: Valor maximo permitido. Default: `RUL_CAP` do config (125).

    Returns:
        RUL truncado no cap; preserva tipo de entrada (Series ou ndarray).

    Examples:
        >>> import pandas as pd
        >>> apply_piecewise_rul(pd.Series([200, 130, 100, 50, 10]), cap=125).tolist()
        [125, 125, 100, 50, 10]
    """
    if isinstance(rul, pd.Series):
        return rul.clip(upper=cap)
    return np.minimum(rul, cap)
