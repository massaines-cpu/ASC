"""Lecture et validation homogènes des résultats LODO ASC."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config.settings import EXPECTED_DYADS, ExperimentSpec


STANDARD_HISTORY_COLUMNS = {
    "train_loss": "train_loss",
    "train_accuracy": "train_accuracy",
    "validation_loss": "validation_loss",
    "validation_accuracy": "validation_accuracy",
}

SIGNAL_JEPA_HISTORY_COLUMNS = {
    "train_loss": "train_window_loss",
    "train_accuracy": "train_window_accuracy",
    "validation_loss": "validation_participant_loss",
    "validation_accuracy": "validation_participant_accuracy",
}


def require_file(path: Path) -> Path:
    """Arrête l'analyse avec un message précis lorsqu'un fichier manque."""

    if not path.exists():
        raise FileNotFoundError(f"Fichier requis introuvable : {path}")
    return path


def load_summary(experiment: ExperimentSpec) -> pd.DataFrame:
    """Charge le résumé LODO et vérifie la présence des huit dyades."""

    summary_path = require_file(
        experiment.result_directory / "lodo_cv_summary.csv"
    )
    table = pd.read_csv(summary_path)

    required_columns = {
        "validation_dyad",
        "best_epoch",
        "best_validation_loss",
        "best_validation_accuracy",
    }
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(
            f"Colonnes absentes de {summary_path} : "
            + ", ".join(sorted(missing_columns))
        )

    found_dyads = set(table["validation_dyad"].astype(str))
    expected_dyads = set(EXPECTED_DYADS)
    missing_dyads = expected_dyads - found_dyads
    unexpected_dyads = found_dyads - expected_dyads

    if missing_dyads or unexpected_dyads or len(table) != len(EXPECTED_DYADS):
        raise ValueError(
            f"LODO incomplet pour '{experiment.label}'. "
            f"Manquantes={sorted(missing_dyads)}, "
            f"inattendues={sorted(unexpected_dyads)}, lignes={len(table)}."
        )

    ordered_table = table.set_index("validation_dyad").loc[
        list(EXPECTED_DYADS)
    ]
    return ordered_table.reset_index()


def load_history(
    experiment: ExperimentSpec,
    validation_dyad: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Charge un historique et renvoie les colonnes scientifiques communes."""

    history_path = require_file(
        experiment.result_directory
        / f"fold_{validation_dyad}"
        / "history.csv"
    )
    history = pd.read_csv(history_path)

    if set(STANDARD_HISTORY_COLUMNS.values()).issubset(history.columns):
        column_mapping = STANDARD_HISTORY_COLUMNS
    elif set(SIGNAL_JEPA_HISTORY_COLUMNS.values()).issubset(history.columns):
        column_mapping = SIGNAL_JEPA_HISTORY_COLUMNS
    else:
        raise ValueError(
            f"Format d'historique non reconnu : {history_path}. "
            f"Colonnes trouvées : {list(history.columns)}"
        )

    if "epoch" not in history.columns:
        history.insert(0, "epoch", range(1, len(history) + 1))

    return history, column_mapping


def load_experiment_config(experiment: ExperimentSpec) -> dict:
    """Lit la configuration enregistrée, ou renvoie un dictionnaire vide."""

    config_path = experiment.result_directory / "experiment_config.json"
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def find_checkpoint(
    experiment: ExperimentSpec,
    validation_dyad: str,
    models_root: Path,
) -> Path | None:
    """Recherche le checkpoint sans inventer son emplacement."""

    checkpoint_name = f"best_model_fold_{validation_dyad}.pt"
    exact_candidate = (
        models_root
        / experiment.result_directory_name
        / checkpoint_name
    )
    if exact_candidate.exists():
        return exact_candidate

    matches = list(models_root.rglob(checkpoint_name))
    matching_experiment = [
        path
        for path in matches
        if experiment.result_directory_name in str(path.parent)
    ]
    if len(matching_experiment) == 1:
        return matching_experiment[0]

    return None

