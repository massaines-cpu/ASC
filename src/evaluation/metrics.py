"""Métriques et rapports d'évaluation pour un fold LODO."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn


CLASS_NAMES = ["YO", "YF"]


def collect_predictions(
    model: nn.Module,
    loader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne les labels, classes prédites et probabilités.

    Le réseau produit deux logits bruts. ``softmax`` est appliqué ici,
    uniquement pendant l'évaluation, afin d'obtenir les probabilités YO/YF.
    """

    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for eeg, labels in loader:
            logits = model(eeg)

            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(
                    "L'évaluation attend deux logits par exemple, "
                    f"mais a reçu une sortie de forme {tuple(logits.shape)}."
                )

            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_labels.append(labels.detach().cpu().numpy())
            all_predictions.append(predictions.detach().cpu().numpy())
            all_probabilities.append(probabilities.detach().cpu().numpy())

    if not all_labels:
        raise ValueError("Impossible d'évaluer un DataLoader vide.")

    return (
        np.concatenate(all_labels),
        np.concatenate(all_predictions),
        np.concatenate(all_probabilities),
    )


def evaluate_fold(
    validation_dyad: str,
    model: nn.Module,
    loader,
    results_dir: Path,
) -> dict[str, float]:
    """Évalue le meilleur checkpoint et enregistre les rapports du fold."""

    labels, predictions, probabilities = collect_predictions(model, loader)
    correct_mask = predictions == labels
    predicted_confidences = probabilities.max(axis=1)

    accuracy = float(correct_mask.mean())
    mean_confidence = float(predicted_confidences.mean())
    mean_correct_confidence = (
        float(predicted_confidences[correct_mask].mean())
        if correct_mask.any()
        else float("nan")
    )
    mean_error_confidence = (
        float(predicted_confidences[~correct_mask].mean())
        if (~correct_mask).any()
        else float("nan")
    )

    # labels=[0, 1] garantit une matrice 2 × 2 même si une classe n'est
    # jamais prédite dans un fold.
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )
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

    prediction_table = pd.DataFrame({
        "true_label": labels,
        "true_class": [CLASS_NAMES[label] for label in labels],
        "predicted_label": predictions,
        "predicted_class": [CLASS_NAMES[label] for label in predictions],
        "probability_yo": probabilities[:, 0],
        "probability_yf": probabilities[:, 1],
        "correct": correct_mask,
    })
    prediction_table.to_csv(
        fold_dir / "predictions.csv",
        index=False,
    )

    matrix_table = pd.DataFrame(
        matrix,
        index=["true_YO", "true_YF"],
        columns=["predicted_YO", "predicted_YF"],
    )
    matrix_table.to_csv(fold_dir / "confusion_matrix.csv")

    lines = [
        f"Dyade de validation : {validation_dyad}",
        f"Accuracy : {accuracy:.4f}",
        f"Confiance moyenne : {mean_confidence:.4f}",
        (
            "Confiance moyenne des bonnes prédictions : "
            f"{mean_correct_confidence:.4f}"
        ),
        (
            "Confiance moyenne des erreurs : "
            f"{mean_error_confidence:.4f}"
        ),
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
    print("\nMatrice de confusion :")
    print(matrix)
    print("\nClassification report :")
    print(report)

    return {
        "evaluation_accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "mean_correct_confidence": mean_correct_confidence,
        "mean_error_confidence": mean_error_confidence,
    }
