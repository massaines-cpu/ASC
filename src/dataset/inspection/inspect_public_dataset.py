"""Inventorie un dataset EEG public avant de préparer son pré-entraînement.

Le script ne transforme aucune donnée. Il répond d'abord aux questions de
compatibilité : formats, formes, nombre de canaux, longueur temporelle et taille
sur disque. Les fichiers MATLAB sont inspectés sans supposer le nom de leur
variable EEG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import whosmat

from src.config.settings import (
    PUBLIC_DATASET_ROOT,
    REPORT_OUTPUT_ROOT,
)


SUPPORTED_SUFFIXES = {".npy", ".mat"}


def inspect_npy(file_path: Path) -> list[dict[str, object]]:
    """Lit uniquement l'en-tête logique d'un tableau NumPy via mmap."""

    array = np.load(file_path, mmap_mode="r")
    return [{
        "file_path": str(file_path),
        "file_type": "npy",
        "variable_name": "array",
        "shape": str(tuple(array.shape)),
        "number_of_dimensions": array.ndim,
        "dtype": str(array.dtype),
        "size_megabytes": file_path.stat().st_size / (1024 ** 2),
    }]


def inspect_mat(file_path: Path) -> list[dict[str, object]]:
    """Liste les variables MATLAB sans charger tout le signal en mémoire."""

    rows = []
    for variable_name, shape, matlab_type in whosmat(file_path):
        rows.append({
            "file_path": str(file_path),
            "file_type": "mat",
            "variable_name": variable_name,
            "shape": str(tuple(shape)),
            "number_of_dimensions": len(shape),
            "dtype": matlab_type,
            "size_megabytes": file_path.stat().st_size / (1024 ** 2),
        })
    return rows


def inspect_file(file_path: Path) -> list[dict[str, object]]:
    """Route l'inspection selon l'extension."""

    if file_path.suffix.lower() == ".npy":
        return inspect_npy(file_path)
    if file_path.suffix.lower() == ".mat":
        return inspect_mat(file_path)
    return []


def main() -> None:
    """Crée l'inventaire CSV du dataset public sélectionné."""

    if PUBLIC_DATASET_ROOT is None:
        raise ValueError(
            "Renseigne PUBLIC_DATASET_ROOT dans settings.py après avoir "
            "vérifié la licence et téléchargé le dataset."
        )
    if not PUBLIC_DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset public introuvable : {PUBLIC_DATASET_ROOT}"
        )

    files = sorted(
        path
        for path in PUBLIC_DATASET_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise ValueError(
            f"Aucun fichier .npy ou .mat dans {PUBLIC_DATASET_ROOT}."
        )

    rows = []
    for file_path in files:
        rows.extend(inspect_file(file_path))

    inventory = pd.DataFrame(rows)
    output_directory = REPORT_OUTPUT_ROOT / "public_pretraining"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "public_dataset_inventory.csv"
    inventory.to_csv(output_path, index=False)

    print(inventory.head(30).to_string(index=False))
    print(f"\nFichiers inspectés : {len(files)}")
    print(f"Inventaire         : {output_path}")
    print(
        "Étape suivante : identifier dans la documentation la variable EEG, "
        "l'ordre des canaux, la fréquence, les sujets et les labels."
    )


if __name__ == "__main__":
    main()
