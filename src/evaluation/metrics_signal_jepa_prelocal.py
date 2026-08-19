"""Métriques adaptées aux cinq fenêtres d'un participant ASC.

Le modèle prédit chaque fenêtre de deux secondes. La question scientifique
reste cependant : « quelle est la condition du participant ? ». Les cinq
probabilités YF d'un même participant sont donc moyennées avant de calculer
l'accuracy, la matrice de confusion et le rapport de classification.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.nn import functional as functional

from src.models.signal_jepa_prelocal_model import prepare_binary_logits


CLASS_NAMES = ["YO", "YF"]
EXPECTED_WINDOWS_PER_PARTICIPANT = 5


def collect_window_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
) -> pd.DataFrame:
    """Collecte logits et probabilités sans encore fusionner les fenêtres."""

    model.eval()
    rows = []

    with torch.no_grad():
        for eeg, labels, sample_ids, window_indices in loader:
            eeg = eeg.to(device)
            labels = labels.to(device)

            logits = prepare_binary_logits(model(eeg))
            probabilities_yf = torch.sigmoid(logits)
            predictions = (probabilities_yf >= 0.5).long()

            labels_cpu = labels.detach().cpu().numpy()
            logits_cpu = logits.detach().cpu().numpy()
            probabilities_cpu = probabilities_yf.detach().cpu().numpy()
            predictions_cpu = predictions.detach().cpu().numpy()
            window_indices_cpu = window_indices.detach().cpu().numpy()

            for row_index, sample_id in enumerate(sample_ids):
                rows.append({
                    "sample_id": str(sample_id),
                    "window_index": int(window_indices_cpu[row_index]),
                    "true_label": int(labels_cpu[row_index]),
                    "true_class": CLASS_NAMES[int(labels_cpu[row_index])],
                    "logit_yf": float(logits_cpu[row_index]),
                    "probability_yo": float(
                        1.0 - probabilities_cpu[row_index]
                    ),
                    "probability_yf": float(probabilities_cpu[row_index]),
                    "predicted_label": int(predictions_cpu[row_index]),
                    "predicted_class": CLASS_NAMES[
                        int(predictions_cpu[row_index])
                    ],
                })

    if not rows:
        raise ValueError("Impossible d'évaluer un DataLoader vide.")

    return pd.DataFrame(rows)


def aggregate_windows_by_participant(
    window_table: pd.DataFrame,
) -> pd.DataFrame:
    """Moyenne les cinq probabilités d'un participant avant décision."""

    participant_rows = []

    for sample_id, group in window_table.groupby("sample_id", sort=False):
        unique_labels = group["true_label"].unique()
        if len(unique_labels) != 1:
            raise ValueError(
                f"Plusieurs labels sont associés au participant {sample_id}."
            )

        number_of_windows = len(group)
        if number_of_windows != EXPECTED_WINDOWS_PER_PARTICIPANT:
            raise ValueError(
                f"{sample_id} possède {number_of_windows} fenêtres au lieu "
                f"de {EXPECTED_WINDOWS_PER_PARTICIPANT}."
            )

        true_label = int(unique_labels[0])
        probability_yf = float(group["probability_yf"].mean())
        probability_yo = 1.0 - probability_yf
        predicted_label = int(probability_yf >= 0.5)

        participant_rows.append({
            "sample_id": sample_id,
            "true_label": true_label,
            "true_class": CLASS_NAMES[true_label],
            "number_of_windows": number_of_windows,
            "probability_yo": probability_yo,
            "probability_yf": probability_yf,
            "predicted_label": predicted_label,
            "predicted_class": CLASS_NAMES[predicted_label],
            "correct": predicted_label == true_label,
        })

    return pd.DataFrame(participant_rows)


def calculate_validation_metrics(
    window_table: pd.DataFrame,
    participant_table: pd.DataFrame,
) -> dict[str, float]:
    """Calcule séparément les métriques fenêtre et participant."""

    window_logits = torch.tensor(
        window_table["logit_yf"].to_numpy(),
        dtype=torch.float32,
    )
    window_targets = torch.tensor(
        window_table["true_label"].to_numpy(),
        dtype=torch.float32,
    )
    window_loss = functional.binary_cross_entropy_with_logits(
        window_logits,
        window_targets,
    ).item()
    window_accuracy = float(
        (
            window_table["predicted_label"].to_numpy()
            == window_table["true_label"].to_numpy()
        ).mean()
    )

    # Après moyenne des probabilités, la BCE est calculée au niveau de
    # l'observation scientifique réelle : un participant.
    participant_probabilities = np.clip(
        participant_table["probability_yf"].to_numpy(dtype=np.float64),
        1e-7,
        1.0 - 1e-7,
    )
    participant_targets = participant_table["true_label"].to_numpy(
        dtype=np.float64
    )
    participant_loss = float(
        -np.mean(
            participant_targets * np.log(participant_probabilities)
            + (1.0 - participant_targets)
            * np.log(1.0 - participant_probabilities)
        )
    )
    participant_accuracy = float(participant_table["correct"].mean())

    return {
        "window_loss": window_loss,
        "window_accuracy": window_accuracy,
        "participant_loss": participant_loss,
        "participant_accuracy": participant_accuracy,
    }


def evaluate_validation_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict[str, float]:
    """Évalue une epoch et retourne les métriques utilisées par l'arrêt."""

    window_table = collect_window_predictions(model, loader, device)
    participant_table = aggregate_windows_by_participant(window_table)
    return calculate_validation_metrics(window_table, participant_table)


def evaluate_and_save_fold(
    validation_dyad: str,
    model: nn.Module,
    loader,
    results_dir: Path,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    """Évalue le meilleur checkpoint et sauvegarde tous les détails."""

    window_table = collect_window_predictions(model, loader, device)
    participant_table = aggregate_windows_by_participant(window_table)
    metrics = calculate_validation_metrics(window_table, participant_table)

    labels = participant_table["true_label"].to_numpy()
    predictions = participant_table["predicted_label"].to_numpy()
    correct_mask = labels == predictions
    confidences = participant_table[
        ["probability_yo", "probability_yf"]
    ].to_numpy().max(axis=1)

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    fold_dir = results_dir / f"fold_{validation_dyad}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    window_table.to_csv(
        fold_dir / "window_predictions.csv",
        index=False,
    )
    participant_table.to_csv(
        fold_dir / "participant_predictions.csv",
        index=False,
    )

    matrix_table = pd.DataFrame(
        matrix,
        index=["true_YO", "true_YF"],
        columns=["predicted_YO", "predicted_YF"],
    )
    matrix_table.to_csv(fold_dir / "confusion_matrix.csv")

    mean_confidence = float(confidences.mean())
    mean_correct_confidence = (
        float(confidences[correct_mask].mean())
        if correct_mask.any()
        else float("nan")
    )
    mean_error_confidence = (
        float(confidences[~correct_mask].mean())
        if (~correct_mask).any()
        else float("nan")
    )

    metrics.update({
        "mean_confidence": mean_confidence,
        "mean_correct_confidence": mean_correct_confidence,
        "mean_error_confidence": mean_error_confidence,
        "number_of_participants": float(len(participant_table)),
        "number_of_windows": float(len(window_table)),
    })

    lines = [
        f"Dyade de validation : {validation_dyad}",
        (
            "Accuracy participant : "
            f"{metrics['participant_accuracy']:.4f}"
        ),
        f"Loss participant : {metrics['participant_loss']:.4f}",
        f"Accuracy fenêtre : {metrics['window_accuracy']:.4f}",
        f"Loss fenêtre : {metrics['window_loss']:.4f}",
        f"Participants évalués : {len(participant_table)}",
        f"Fenêtres évaluées : {len(window_table)}",
        f"Confiance moyenne : {mean_confidence:.4f}",
        f"Prédictions YO : {int((predictions == 0).sum())}",
        f"Prédictions YF : {int((predictions == 1).sum())}",
    ]

    report_path = fold_dir / "evaluation_report.txt"
    with report_path.open("w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))
        report_file.write("\n\nMatrice de confusion :\n")
        report_file.write(str(matrix))
        report_file.write("\n\nClassification report :\n")
        report_file.write(report)

    print("\n".join(lines))
    print("\nMatrice de confusion participant :")
    print(matrix)
    print("\nClassification report participant :")
    print(report)

    return metrics, matrix
