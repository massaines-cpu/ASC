"""Figures de l'expérience SignalJEPA PreLocal."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_fold_history_and_plots(
    validation_dyad: str,
    history: dict[str, list[float]],
    results_dir: Path,
) -> None:
    """Sauvegarde les métriques fenêtre et participant sans les confondre."""

    fold_dir = results_dir / f"fold_{validation_dyad}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    history_table = pd.DataFrame(history)
    history_table.insert(
        0,
        "epoch",
        np.arange(1, len(history_table) + 1),
    )
    history_table.to_csv(fold_dir / "history.csv", index=False)

    epochs = history_table["epoch"]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(
        epochs,
        history_table["train_window_loss"],
        "--",
        label="Train — fenêtres",
    )
    axes[0].plot(
        epochs,
        history_table["validation_window_loss"],
        label="Validation — fenêtres",
    )
    axes[0].plot(
        epochs,
        history_table["validation_participant_loss"],
        label="Validation participants",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        history_table["train_window_accuracy"],
        "--",
        label="Train — fenêtres",
    )
    axes[1].plot(
        epochs,
        history_table["validation_window_accuracy"],
        label="Validation — fenêtres",
    )
    axes[1].set_title("Accuracy par fenêtre")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        epochs,
        history_table["validation_participant_accuracy"],
        color="tab:green",
        label="Validation participants",
    )
    axes[2].set_title("Accuracy scientifique finale")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy participant")
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    figure.suptitle(
        f"SignalJEPA PreLocal validation {validation_dyad}"
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    figure.savefig(
        fold_dir / "training_curves.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_confusion_matrix_plot(
    validation_dyad: str,
    matrix: np.ndarray,
    results_dir: Path,
) -> None:
    """Affiche la matrice calculée après agrégation des cinq fenêtres."""

    if matrix.shape != (2, 2):
        raise ValueError(
            f"La matrice doit être 2 × 2, reçu {matrix.shape}."
        )

    fold_dir = results_dir / f"fold_{validation_dyad}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Nombre de participants")

    axis.set_xticks([0, 1], labels=["YO", "YF"])
    axis.set_yticks([0, 1], labels=["YO", "YF"])
    axis.set_xlabel("Classe prédite")
    axis.set_ylabel("Classe réelle")
    axis.set_title(
        f"Matrice participant validation {validation_dyad}"
    )

    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(2):
        for column_index in range(2):
            value = int(matrix[row_index, column_index])
            color = "white" if value > threshold else "black"
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=color,
            )

    figure.tight_layout()
    figure.savefig(
        fold_dir / "confusion_matrix_participant.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_global_fold_comparison(
    histories: dict[str, dict[str, list[float]]],
    results_dir: Path,
) -> None:
    """Compare train et validation de tous les folds exécutés.

    ne jamais présenter la validation
    seule, toujours l'accompagner du train pour montrer l'écart entre les
    deux (signe de surapprentissage). Le train est mesuré au niveau
    fenêtre (l'unité d'entraînement de PreLocal) et la validation au niveau
    participant (l'unité d'évaluation scientifique du projet) : les deux
    échelles ne sont pas directement comparables valeur à valeur, donc le
    titre de chaque axe le précise explicitement plutôt que de laisser
    croire à une correspondance directe fenêtre/participant.
    """

    if not histories:
        raise ValueError("Aucun historique de fold à représenter.")

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for dyad_index, (validation_dyad, history) in enumerate(histories.items()):
        color = color_cycle[dyad_index % len(color_cycle)]
        train_epochs = np.arange(1, len(history["train_window_loss"]) + 1)
        validation_epochs = np.arange(
            1, len(history["validation_participant_loss"]) + 1
        )

        axes[0].plot(
            train_epochs,
            history["train_window_loss"],
            "--",
            color=color,
            alpha=0.6,
        )
        axes[0].plot(
            validation_epochs,
            history["validation_participant_loss"],
            color=color,
            label=validation_dyad,
        )

        axes[1].plot(
            train_epochs,
            history["train_window_accuracy"],
            "--",
            color=color,
            alpha=0.6,
        )
        axes[1].plot(
            validation_epochs,
            history["validation_participant_accuracy"],
            color=color,
            label=validation_dyad,
        )

    axes[0].set_title("Loss train (pointillé) vs validation (participant)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].grid(alpha=0.3)
    axes[0].legend(title="Dyade (validation)")

    axes[1].set_title("Accuracy train ( pointillé) vs validation (participant)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend(title="Dyade (validation)")

    figure.suptitle("SignalJEPA PreLocal Leave-One-Dyad-Out (train vs validation)")
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    figure.savefig(
        results_dir / "all_folds_participant_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)
