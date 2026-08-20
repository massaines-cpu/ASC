"""Vérifie que les métriques publiées correspondent au meilleur checkpoint.

L'early stopping ASC surveille la validation loss. Par conséquent :

1. ``best_epoch`` doit être la dernière époque améliorant la validation loss
   d'au moins ``min_delta`` ;
2. ``best_validation_loss`` doit être la loss de cette époque ;
3. ``best_validation_accuracy`` doit être l'accuracy observée à cette même
   époque, pas nécessairement l'accuracy maximale de toute la courbe ;
4. un fichier ``best_model_fold_<dyade>.pt`` doit exister.
"""

from __future__ import annotations

import math

import pandas as pd

from src.evaluation.result_io import (
    find_checkpoint,
    load_experiment_config,
    load_history,
    load_summary,
)
from src.config.settings import (
    COMPARISON_GROUPS,
    EXPECTED_DYADS,
    MODELS_ROOT,
    REPORT_OUTPUT_ROOT,
    ExperimentSpec,
)


ABSOLUTE_TOLERANCE = 1e-7


def unique_experiments() -> list[ExperimentSpec]:
    """Évite d'auditer deux fois une expérience présente dans deux groupes."""

    by_directory = {}
    for experiment_group in COMPARISON_GROUPS.values():
        for experiment in experiment_group:
            by_directory[experiment.result_directory_name] = experiment
    return list(by_directory.values())


def audit_fold(
    experiment: ExperimentSpec,
    validation_dyad: str,
    summary_row: pd.Series,
) -> dict[str, object]:
    """Rejoue la règle de sélection de l'early stopping sur l'historique."""

    history, columns = load_history(experiment, validation_dyad)
    validation_loss_column = columns["validation_loss"]
    validation_accuracy_column = columns["validation_accuracy"]

    config = load_experiment_config(experiment)
    minimum_delta = float(
        config.get("early_stopping_min_delta", 1e-4)
    )

    # L'early stopping ne sauvegarde un nouveau checkpoint que si la baisse
    # dépasse min_delta. Le minimum numérique absolu peut donc être légèrement
    # inférieur au checkpoint retenu sans constituer une incohérence.
    selected_row_index = None
    selected_loss = float("inf")
    for row_index, row in history.iterrows():
        candidate_loss = float(row[validation_loss_column])
        if candidate_loss < selected_loss - minimum_delta:
            selected_loss = candidate_loss
            selected_row_index = row_index

    if selected_row_index is None:
        raise RuntimeError("Historique vide ou invalide.")

    selected_row = history.loc[selected_row_index]
    expected_best_epoch = int(selected_row["epoch"])
    expected_best_loss = float(selected_row[validation_loss_column])
    expected_accuracy_at_best_loss = float(
        selected_row[validation_accuracy_column]
    )

    reported_best_epoch = int(summary_row["best_epoch"])
    reported_best_loss = float(summary_row["best_validation_loss"])
    reported_best_accuracy = float(
        summary_row["best_validation_accuracy"]
    )

    checkpoint_path = find_checkpoint(
        experiment=experiment,
        validation_dyad=validation_dyad,
        models_root=MODELS_ROOT,
    )

    epoch_matches = reported_best_epoch == expected_best_epoch
    loss_matches = math.isclose(
        reported_best_loss,
        expected_best_loss,
        rel_tol=0.0,
        abs_tol=ABSOLUTE_TOLERANCE,
    )
    accuracy_matches = math.isclose(
        reported_best_accuracy,
        expected_accuracy_at_best_loss,
        rel_tol=0.0,
        abs_tol=ABSOLUTE_TOLERANCE,
    )
    checkpoint_exists = checkpoint_path is not None

    return {
        "experiment": experiment.label,
        "result_directory": experiment.result_directory_name,
        "validation_dyad": validation_dyad,
        "reported_best_epoch": reported_best_epoch,
        "history_selected_epoch": expected_best_epoch,
        "reported_best_validation_loss": reported_best_loss,
        "history_selected_validation_loss": expected_best_loss,
        "reported_accuracy_at_checkpoint": reported_best_accuracy,
        "history_accuracy_at_selected_loss": expected_accuracy_at_best_loss,
        "early_stopping_min_delta": minimum_delta,
        "epoch_matches": epoch_matches,
        "loss_matches": loss_matches,
        "accuracy_matches": accuracy_matches,
        "checkpoint_exists": checkpoint_exists,
        "checkpoint_path": str(checkpoint_path or ""),
        "audit_passed": (
            epoch_matches
            and loss_matches
            and accuracy_matches
            and checkpoint_exists
        ),
    }


def main() -> None:
    """Audite tous les folds de toutes les expériences configurées."""

    audit_rows = []

    for experiment in unique_experiments():
        summary = load_summary(experiment).set_index("validation_dyad")

        for validation_dyad in EXPECTED_DYADS:
            audit_rows.append(
                audit_fold(
                    experiment=experiment,
                    validation_dyad=validation_dyad,
                    summary_row=summary.loc[validation_dyad],
                )
            )

    audit_table = pd.DataFrame(audit_rows)
    REPORT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_OUTPUT_ROOT / "best_checkpoint_audit.csv"
    audit_table.to_csv(output_path, index=False)

    failed_rows = audit_table[~audit_table["audit_passed"]]
    print(
        audit_table.groupby("experiment")["audit_passed"]
        .agg(["sum", "count"])
        .to_string()
    )
    print(f"\nRapport complet : {output_path}")

    if not failed_rows.empty:
        print("\nPoints à vérifier :")
        print(
            failed_rows[
                [
                    "experiment",
                    "validation_dyad",
                    "epoch_matches",
                    "loss_matches",
                    "accuracy_matches",
                    "checkpoint_exists",
                ]
            ].to_string(index=False)
        )
        raise RuntimeError(
            "Certaines métriques ne correspondent pas au meilleur checkpoint."
        )


if __name__ == "__main__":
    main()
