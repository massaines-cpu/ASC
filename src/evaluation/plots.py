"""Création des figures homogènes pour les expériences LODO."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_fold_plots(
    validation_dyad: str,
    history: dict[str, list[float]],
    results_dir: Path,
) -> None:
    """Enregistre l'historique et les courbes train/validation d'un fold."""

    required_keys = {
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
    }
    missing_keys = required_keys - set(history)
    if missing_keys:
        raise ValueError(
            "Clés manquantes dans history : "
            + ", ".join(sorted(missing_keys))
        )

    lengths = {len(history[key]) for key in required_keys}
    if len(lengths) != 1:
        raise ValueError("Toutes les séries de history doivent avoir la même taille.")

    number_of_epochs = lengths.pop()
    if number_of_epochs == 0:
        raise ValueError("Aucune epoch à représenter.")

    epochs = np.arange(1, number_of_epochs + 1)
    fold_dir = results_dir / f"fold_{validation_dyad}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    history_table = pd.DataFrame(history)
    history_table.insert(0, "epoch", epochs)
    history_table.to_csv(fold_dir / "history.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], "--", label="Train")
    axes[0].plot(epochs, history["validation_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], "--", label="Train")
    axes[1].plot(epochs, history["validation_accuracy"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.suptitle(f"Apprentissage — validation {validation_dyad}")
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(fold_dir / "training_curves.png", dpi=150)
    plt.close(figure)


def save_confusion_matrix_plot(
    validation_dyad: str,
    matrix: np.ndarray,
    results_dir: Path,
) -> None:
    """Enregistre une matrice de confusion lisible sans dépendre de seaborn."""

    if matrix.shape != (2, 2):
        raise ValueError(
            f"La matrice doit avoir la forme (2, 2), reçu {matrix.shape}."
        )

    fold_dir = results_dir / f"fold_{validation_dyad}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Nombre d'exemples")

    axis.set_xticks([0, 1], labels=["YO", "YF"])
    axis.set_yticks([0, 1], labels=["YO", "YF"])
    axis.set_xlabel("Classe prédite")
    axis.set_ylabel("Classe réelle")
    axis.set_title(f"Matrice de confusion — validation {validation_dyad}")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(2):
        for column_index in range(2):
            value = int(matrix[row_index, column_index])
            text_color = "white" if value > threshold else "black"
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
            )

    figure.tight_layout()
    figure.savefig(fold_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def save_global_comparison(
    all_histories: dict[str, dict[str, list[float]]],
    results_dir: Path,
) -> None:
    """Compare les courbes de validation des différents folds LODO."""

    if not all_histories:
        raise ValueError("Aucun historique de fold à représenter.")

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))

    for validation_dyad, history in all_histories.items():
        loss_epochs = np.arange(1, len(history["validation_loss"]) + 1)
        accuracy_epochs = np.arange(
            1,
            len(history["validation_accuracy"]) + 1,
        )

        axes[0].plot(
            loss_epochs,
            history["validation_loss"],
            label=validation_dyad,
        )
        axes[1].plot(
            accuracy_epochs,
            history["validation_accuracy"],
            label=validation_dyad,
        )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title("Validation loss par fold")
    axes[0].grid(alpha=0.3)
    axes[0].legend(title="Dyade")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Validation accuracy par fold")
    axes[1].grid(alpha=0.3)
    axes[1].legend(title="Dyade")

    figure.suptitle("Leave-One-Dyad-Out — comparaison des folds")
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(results_dir / "all_folds_comparison.png", dpi=150)
    plt.close(figure)
