"""
Correctifs ciblés pour le fold J8 (biais systématique vers la classe YO).

Deux changements testés, un par un, uniquement sur J8 :

  A. Patience augmentée : le fold J8 s'arrêtait à l'epoch 8 dans le run
     précédent (le plus précoce de tous les folds). Avec seulement 32
     exemples de validation, la validation loss est bruitée — on vérifie
     si le modèle continue réellement à s'améliorer au-delà.

  B. Sélection du meilleur epoch sur l'ACCURACY ÉQUILIBRÉE (moyenne des
     recalls par classe) plutôt que sur la validation loss brute. Avec un
     petit set de validation, la loss peut se dégrader légèrement sans
     que l'accuracy en pâtisse, ou inversement masquer un biais de classe
     (le modèle peut avoir une loss basse tout en prédisant presque
     toujours la même classe).

Chaque option est entraînée et évaluée séparément pour voir laquelle,
si l'une des deux, corrige effectivement le biais.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

from src.dataset.labels import prepare_classification_table
from src.dataset.dataloader_participant import create_participant_dataloaders
from src.models.eegNET_model import EEGNet

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_DYAD = "J8"
TRAIN_DYADS = ["J1", "J2", "J4", "J5", "J7", "J10"]
TEST_DYADS = ["J15"]

BATCH_SIZE = 5
NUMBER_OF_EPOCHS = 150
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
EARLY_STOPPING_MIN_DELTA = 1e-4

criterion = nn.CrossEntropyLoss()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


metadata = pd.read_csv(PROJECT_ROOT / "data" / "all_metadata.csv")
classification_table = prepare_classification_table(
    metadata=metadata,
    target_column="eyes_code",
    allowed_classes=["YO", "YF"],
    label_map={"YO": 0, "YF": 1},
)


def run_epoch(model, loader, optimizer=None):
    """Exécute une passe (train si optimizer fourni, sinon validation).

    Retourne aussi les labels/prédictions bruts pour pouvoir calculer
    l'accuracy équilibrée en plus de la loss.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_examples = 0
    all_labels, all_predictions = [], []

    gradient_context = torch.enable_grad() if training else torch.no_grad()

    with gradient_context:
        for eeg, labels in loader:
            if training:
                optimizer.zero_grad()

            predictions = model(eeg)
            loss = criterion(predictions, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            predicted_classes = predictions.argmax(dim=1)
            all_labels.append(labels.numpy())
            all_predictions.append(predicted_classes.numpy())

    average_loss = total_loss / total_examples
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)

    accuracy = (all_predictions == all_labels).mean()
    # Moyenne des recalls par classe : un modèle qui prédit toujours la
    # même classe aura une balanced_accuracy proche de 0.5, même si sa
    # loss ou son accuracy brute semblent correctes.
    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)

    return average_loss, accuracy, balanced_accuracy, all_labels, all_predictions


def train_j8(selection_metric: str, patience: int):
    """Entraîne un modèle neuf pour le fold J8.

    selection_metric : "loss" ou "balanced_accuracy"
        Critère utilisé pour choisir le meilleur epoch.
    patience : nombre d'epochs sans amélioration avant early stopping.
    """
    print("\n" + "#" * 70)
    print(f"Run : sélection = {selection_metric} | patience = {patience}")
    print("#" * 70)

    train_loader, validation_loader, _ = create_participant_dataloaders(
        classification_table=classification_table,
        dataset_root=PROJECT_ROOT / "data" / "data_toy_repaired",
        train_dyads=TRAIN_DYADS,
        validation_dyads=[VALIDATION_DYAD],
        test_dyads=TEST_DYADS,
        batch_size=BATCH_SIZE,
    )

    set_seed(RANDOM_SEED)
    model = EEGNet(n_channels=32, n_classes=2, n_samples=5120)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # "best" est celui qui MAXIMISE la métrique choisie ; pour la loss,
    # on inverse le signe pour garder une logique de maximisation commune.
    best_score = -float("inf")
    best_epoch = None
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch_index in range(NUMBER_OF_EPOCHS):
        epoch_number = epoch_index + 1

        train_loss, train_accuracy, train_balanced_acc, _, _ = run_epoch(
            model, train_loader, optimizer
        )
        val_loss, val_accuracy, val_balanced_acc, val_labels, val_predictions = run_epoch(
            model, validation_loader
        )

        current_score = (
            -val_loss if selection_metric == "loss" else val_balanced_acc
        )

        improved = current_score > best_score + EARLY_STOPPING_MIN_DELTA

        if improved:
            best_score = current_score
            best_epoch = epoch_number
            best_state_dict = {
                key: value.clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch_number:03d} | "
            f"Train loss: {train_loss:.4f} acc: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} acc: {val_accuracy:.4f} "
            f"balanced_acc: {val_balanced_acc:.4f}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping à l'epoch {epoch_number}.")
            break

    print(f"\nMeilleure epoch : {best_epoch} (score = {best_score:.4f})")

    # Recharge le meilleur état et évalue en détail.
    model.load_state_dict(best_state_dict)
    _, _, _, labels, predictions = run_epoch(model, validation_loader)

    print("Accuracy :", (predictions == labels).mean())
    print("Balanced accuracy :", balanced_accuracy_score(labels, predictions))
    print(confusion_matrix(labels, predictions))
    print(classification_report(labels, predictions, digits=3, zero_division=0))


if __name__ == "__main__":
    # Option A : patience augmentée, sélection classique sur la loss.
    train_j8(selection_metric="loss", patience=30)

    # Option B : sélection sur l'accuracy équilibrée, patience standard.
    train_j8(selection_metric="balanced_accuracy", patience=15)