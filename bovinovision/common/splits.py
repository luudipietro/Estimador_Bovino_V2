"""Split train/val/test por animal (nunca por imagen): todas las fotos de un animal caen en el mismo conjunto."""
import numpy as np
import pandas as pd

from config import SEED, SPLIT_FRACCIONES


def asignar_split(animal_ids, fracciones=SPLIT_FRACCIONES, seed=SEED) -> pd.Series:
    """Serie animal_id -> 'train'|'val'|'test', determinista para un seed dado."""
    unicos = np.array(sorted(pd.unique(pd.Series(animal_ids))))
    rng = np.random.default_rng(seed)
    rng.shuffle(unicos)
    n = len(unicos)
    n_train = int(round(n * fracciones["train"]))
    n_val = int(round(n * fracciones["val"]))
    etiquetas = np.empty(n, dtype=object)
    etiquetas[:n_train] = "train"
    etiquetas[n_train:n_train + n_val] = "val"
    etiquetas[n_train + n_val:] = "test"
    return pd.Series(etiquetas, index=unicos, name="split")


def verificar_sin_fuga(df: pd.DataFrame, col_animal="animal_id", col_split="split") -> None:
    """AssertionError si algún animal aparece en más de un split."""
    n_splits = df.groupby(col_animal)[col_split].nunique()
    malos = n_splits[n_splits > 1]
    assert malos.empty, f"Fuga: {len(malos)} animales en más de un split (ej. {list(malos.index[:3])})"
