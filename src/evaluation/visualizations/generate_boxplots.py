"""Produit les boxplots LODO demandés par Amel.

Chaque boîte est construite avec exactement huit accuracies : une par dyade de
validation. Les points individuels restent visibles afin de ne pas masquer la
variabilité inter-dyades derrière une moyenne.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.result_io import load_summary
from src.config.settings import (
    CHECKPOINT_SELECTION,
    COMPARISON_GROUPS,
    EXPECTED_DYADS,
    REPORT_OUTPUT_ROOT,
    ExperimentSpec,
)


GROUP_TITLES = {
    "mlp_hidden_size": "MLP non linéaire — taille de la couche cachée",
    "mlp_dropout": "MLP 32 neurones — effet du Dropout",
    "architectures": "Architectures from scratch — comparaison LODO",
    "signal_jepa": "SignalJEPA PreLocal — stratégies de transfert",
    "signal_jepa_channels": "SignalJEPA PreLocal — 19 vs 32 canaux (scratch)",
}


def calculate_statistics(
    experiment: ExperimentSpec,
    summary: pd.DataFrame,
) -> dict[str, object]:
    """Calcule les statistiques descriptives des huit folds."""

    accuracies = summary["best_validation_accuracy"].astype(float)
    losses = summary["best_validation_loss"].astype(float)

    return {
        "family": experiment.family,
        "experiment": experiment.label,
        "result_directory": experiment.result_directory_name,
        "number_of_folds": len(summary),
        "accuracy_mean": accuracies.mean(),
        "accuracy_std": accuracies.std(ddof=1),
        "accuracy_median": accuracies.median(),
        "accuracy_q1": accuracies.quantile(0.25),
        "accuracy_q3": accuracies.quantile(0.75),
        "accuracy_min": accuracies.min(),
        "accuracy_max": accuracies.max(),
        "loss_mean": losses.mean(),
        "loss_std": losses.std(ddof=1),
        "checkpoint_selection": CHECKPOINT_SELECTION,
    }


def save_group_boxplot(
    group_name: str,
    experiments: tuple[ExperimentSpec, ...],
    output_directory: Path,
) -> tuple[Path, list[dict[str, object]]]:
    """Enregistre un boxplot et renvoie les statistiques associées."""

    summaries = [load_summary(experiment) for experiment in experiments]
    accuracy_values = [
        summary["best_validation_accuracy"].to_numpy(dtype=float) * 100.0
        for summary in summaries
    ]

    figure_width = max(9.0, len(experiments) * 2.3)
    figure, axis = plt.subplots(figsize=(figure_width, 6.5))

    boxplot = axis.boxplot(
        accuracy_values,
        tick_labels=[experiment.label for experiment in experiments],
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 6,
        },
        medianprops={"color": "#132238", "linewidth": 2},
    )

    colors = ("#77B6EA", "#F4A261", "#74C69D", "#C9A0DC")
    for patch, color in zip(boxplot["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Décalage fixe et reproductible : chaque point correspond à une dyade.
    jitter = np.linspace(-0.12, 0.12, len(EXPECTED_DYADS))
    for experiment_index, values in enumerate(accuracy_values, start=1):
        axis.scatter(
            experiment_index + jitter,
            values,
            color="#132238",
            s=28,
            alpha=0.8,
            zorder=3,
        )

        for x_position, value, dyad in zip(
            experiment_index + jitter,
            values,
            EXPECTED_DYADS,
            strict=True,
        ):
            axis.annotate(
                dyad,
                (x_position, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#374151",
            )

    axis.set_title(GROUP_TITLES[group_name], fontsize=14, weight="bold")
    axis.set_xlabel("Configuration expérimentale")
    axis.set_ylabel("Accuracy au meilleur checkpoint (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=12)

    figure.text(
        0.5,
        0.015,
        (
            "Chaque point = une dyade de validation LODO (n = 8). "
            f"Checkpoint retenu : {CHECKPOINT_SELECTION}. "
            "Losange blanc = moyenne."
        ),
        ha="center",
        fontsize=9,
        color="#374151",
    )
    figure.tight_layout(rect=[0, 0.055, 1, 1])

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"boxplot_{group_name}.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    statistics = [
        calculate_statistics(experiment, summary)
        for experiment, summary in zip(experiments, summaries, strict=True)
    ]
    return output_path, statistics


def main() -> None:
    """Génère tous les boxplots configurés et une table récapitulative."""

    output_directory = REPORT_OUTPUT_ROOT / "boxplots"
    all_statistics = []

    for group_name, experiments in COMPARISON_GROUPS.items():
        output_path, statistics = save_group_boxplot(
            group_name=group_name,
            experiments=experiments,
            output_directory=output_directory,
        )
        all_statistics.extend(statistics)
        print(f"Figure sauvegardée : {output_path}")

    statistics_table = pd.DataFrame(all_statistics)
    statistics_path = output_directory / "comparison_statistics.csv"
    statistics_table.to_csv(statistics_path, index=False)
    print(f"Statistiques sauvegardées : {statistics_path}")


if __name__ == "__main__":
    main()
