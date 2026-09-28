"""Analise de interpretabilidade via SHAP.

TODO: implementar na fase 04_evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def shap_summary(
    model: Any,
    X: pd.DataFrame,
    output_path: Path,
) -> None:
    """Gera summary plot SHAP (importancia global) e salva em `output_path`."""
    raise NotImplementedError("A implementar na fase 04_evaluation")


def shap_waterfall(
    model: Any,
    X: pd.DataFrame,
    instance_idx: int,
    output_path: Path,
) -> None:
    """Waterfall SHAP para uma instancia especifica (ex: motor com menor RUL predito)."""
    raise NotImplementedError("A implementar na fase 04_evaluation")
