"""Loader canonico para os arquivos do dataset NASA C-MAPSS."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from src.utils.config import COLUNAS_CMAPSS, RAW_DIR, SUBSETS

logger = logging.getLogger(__name__)

Subset = Literal["FD001", "FD002", "FD003", "FD004"]
Split = Literal["train", "test", "rul"]


def _arquivo_esperado(subset: Subset, split: Split, raw_dir: Path) -> Path:
    """Constroi caminho para um arquivo especifico do C-MAPSS."""
    if split == "rul":
        nome = f"RUL_{subset}.txt"
    else:
        nome = f"{split}_{subset}.txt"
    return raw_dir / nome


def load_cmapss(
    subset: Subset,
    split: Split,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Carrega um arquivo do dataset C-MAPSS.

    Args:
        subset: Um de "FD001", "FD002", "FD003", "FD004".
        split: "train", "test" ou "rul".
        raw_dir: Diretorio dos arquivos brutos. Default: `src.utils.config.RAW_DIR`.

    Returns:
        DataFrame com colunas canonicas:
          - Para "train" e "test": ["unit_id", "cycle", "op_setting_1..3",
            "sensor_1..21"].
          - Para "rul": ["rul"] (uma linha por motor de teste, na mesma ordem
            de aparicao em test_FD00X.txt).

    Raises:
        ValueError: se `subset` ou `split` for invalido.
        FileNotFoundError: se o arquivo esperado nao existir.
    """
    if subset not in SUBSETS:
        raise ValueError(f"subset invalido: {subset!r}. Esperado um de {SUBSETS}")
    if split not in ("train", "test", "rul"):
        raise ValueError(f"split invalido: {split!r}. Esperado train, test ou rul")

    raw_dir = raw_dir or RAW_DIR
    caminho = _arquivo_esperado(subset, split, raw_dir)

    if not caminho.is_file():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {caminho}. "
            f"Baixe o C-MAPSS conforme instrucoes no README."
        )

    logger.info("Carregando %s", caminho.name)

    if split == "rul":
        df = pd.read_csv(caminho, sep=r"\s+", header=None, names=["rul"])
    else:
        df = pd.read_csv(caminho, sep=r"\s+", header=None, names=COLUNAS_CMAPSS)
        df["unit_id"] = df["unit_id"].astype(int)
        df["cycle"] = df["cycle"].astype(int)

    return df


def load_subset_completo(
    subset: Subset,
    raw_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Carrega os tres arquivos de um subset (train, test, rul)."""
    return {s: load_cmapss(subset, s, raw_dir) for s in ("train", "test", "rul")}
