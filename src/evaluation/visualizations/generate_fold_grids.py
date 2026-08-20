"""Crée les grilles train/validation des huit folds LODO.

Deux figures sont créées par expérience :
- une grille 2 × 4 pour la loss ;
- une grille 2 × 4 pour l'accuracy.

Pour SignalJEPA, train et validation sont comparés au niveau fenêtre. La ligne
verticale rappelle cependant l'époque sélectionnée avec la loss participant.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.result_io import load_history, load_summary
from src.config.settings import (
    CHECKPOINT_SELECTION,
    COMPARISON_GROUPS,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MONITOR,
    EXPECTED_DYADS,
    REPORT_OUTPUT_ROOT,
    ExperimentSpec,
)


def read_patience(experiment: ExperimentSpec) -> int:
    """Lit la patience enregistrée, avec repli pour les anciens runs."""

    config_path = experiment.result_directory / "experiment_config.json"
    if not config_path.exists():
        return DEFAULT_EARLY_STOPPING_PATIENCE

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    return int(
        config.get(
            "early_stopping_patience",
            config.get("patience", DEFAULT_EARLY_STOPPING_PATIENCE),
        )
    )


def select_curve_columns(
    history: pd.DataFrame,
    metric_name: str,
) -> tuple[str, str, str]:
    """Sélectionne des courbes train/validation calculées au même niveau."""

    if metric_name == "loss":
        if {"train_loss", "validation_loss"}.issubset(history.columns):
            return "train_loss", "validation_loss", "participant"
        return "train_window_loss", "validation_window_loss", "fenêtre"

    if {"train_accuracy", "validation_accuracy"}.issubset(history.columns):
        return "train_accuracy", "validation_accuracy", "participant"
    return (
        "train_window_accuracy",
        "validation_window_accuracy",
        "fenêtre",
    )


def save_metric_grid(
    experiment: ExperimentSpec,
    metric_name: str,
    output_directory: Path,
) -> Path:
    """Dessine les huit folds d'une expérience dans une grille commune."""

    summary = load_summary(experiment).set_index("validation_dyad")
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(18, 8),
        sharex=False,
        sharey=False,
    )

    aggregation_levels = set()

    for axis, validation_dyad in zip(
        axes.flat,
        EXPECTED_DYADS,
        strict=True,
    ):
        history, _ = load_history(experiment, validation_dyad)
        train_column, validation_column, aggregation_level = (
            select_curve_columns(history, metric_name)
        )
        aggregation_levels.add(aggregation_level)

        best_epoch = int(summary.loc[validation_dyad, "best_epoch"])

        axis.plot(
            history["epoch"],
            history[train_column],
            linestyle="--",
            color="#3B82F6",
            label="Train",
        )
        axis.plot(
            history["epoch"],
            history[validation_column],
            color="#E76F51",
            label="Validation",
        )
        axis.axvline(
            best_epoch,
            color="#2A9D8F",
            linestyle=":",
            linewidth=2,
            label=f"Best epoch {best_epoch}",
        )

        axis.set_title(f"Validation {validation_dyad}", weight="bold")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric_name.capitalize())
        axis.grid(alpha=0.25)

        if metric_name == "accuracy":
            axis.set_ylim(0, 1)

        axis.legend(fontsize=8)

    patience = read_patience(experiment)
    aggregation_text = ", ".join(sorted(aggregation_levels))
    figure.suptitle(
        f"{experiment.label} — {metric_name} train/validation par fold",
        fontsize=16,
        weight="bold",
    )
    figure.text(
        0.5,
        0.012,
        (
            f"Niveau des courbes : {aggregation_text}. "
            f"Early stopping : patience={patience}, "
            f"métrique={EARLY_STOPPING_MONITOR}, "
            f"checkpoint={CHECKPOINT_SELECTION}."
        ),
        ha="center",
        fontsize=9,
        color="#374151",
    )
    figure.tight_layout(rect=[0, 0.045, 1, 0.95])

    output_directory.mkdir(parents=True, exist_ok=True)
    safe_name = experiment.result_directory_name.replace("/", "_")
    output_path = output_directory / f"{safe_name}_{metric_name}_8folds.png"
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def unique_experiments() -> list[ExperimentSpec]:
    """Retourne chaque répertoire de résultat une seule fois."""

    experiments_by_directory = {}
    for group in COMPARISON_GROUPS.values():
        for experiment in group:
            experiments_by_directory[experiment.result_directory_name] = (
                experiment
            )
    return list(experiments_by_directory.values())


def main() -> None:
    """Produit les grilles loss et accuracy de toutes les expériences."""

    output_directory = REPORT_OUTPUT_ROOT / "fold_grids"

    for experiment in unique_experiments():
        for metric_name in ("loss", "accuracy"):
            output_path = save_metric_grid(
                experiment=experiment,
                metric_name=metric_name,
                output_directory=output_directory,
            )
            print(f"Figure sauvegardée : {output_path}")


if __name__ == "__main__":
    main()

