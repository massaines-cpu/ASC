"""
Leave-One-Dyad-Out Cross-Validation — protocole B.

OBJECTIF
--------
Évaluer la capacité du modèle à généraliser vers une dyade qui n'a jamais
été utilisée pendant l'entraînement.

PROTOCOLE B
-----------
Les dyades de développement sont utilisées dans une cross-validation LODO :

    - une dyade est placée en validation ;
    - toutes les autres dyades de développement sont utilisées en train ;
    - on recommence jusqu'à ce que chaque dyade de développement ait été
      utilisée une fois comme validation.

La dyade J15 reste totalement séparée :

    - elle ne participe à aucun entraînement ;
    - elle ne participe à aucune validation ;
    - elle sera utilisée une seule fois à la fin du projet pour évaluer
      le modèle et l'architecture retenus.

Exemple pour le fold J1 :

    Train      : J2, J4, J5, J7, J8, J10
    Validation : J1
    Test final : J15, non utilisé à ce stade
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, classification_report

from src.dataset.labels import prepare_classification_table
from src.dataset.dataloader_participant import create_participant_dataloaders
from src.models.petit_eeg_cnn import Small_CNN_EEG

# ==================================================================
# 1. CONFIGURATION GÉNÉRALE
# ==================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dyades utilisées pour développer et comparer les modèles.
# Chaque dyade sera utilisée exactement une fois comme validation.
DEVELOPMENT_DYADS = ["J1", "J2", "J4", "J5", "J7", "J8", "J10"]

# J15 reste en dehors de toute la cross-validation (test final unique).
TEST_DYADS = ["J15"]

BATCH_SIZE = 5
NUMBER_OF_EPOCHS = 100
#nombre d’epochs consécutives autorisées sans amélioration significative de la validation loss.
EARLY_STOPPING_PATIENCE = 15
# diminution de loss doit être supérieure à 0.0001 pour remettre le compteur à zéro.
EARLY_STOPPING_MIN_DELTA = 1e-4
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

EXPERIMENT_NAME = "protocol_B_small_cnn_standardized"
RESULTS_DIR = PROJECT_ROOT / "results" / EXPERIMENT_NAME
MODELS_DIR = PROJECT_ROOT / "models" / EXPERIMENT_NAME

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# YO -> 0, YF -> 1 (logits[:, 0] = score YO, logits[:, 1] = score YF)
criterion = nn.CrossEntropyLoss()


# ==================================================================
# 2. REPRODUCTIBILITÉ
# ==================================================================

def set_seed(seed: int) -> None:
    """Fixe les générateurs aléatoires (Python, NumPy, PyTorch) pour
    limiter les variations entre exécutions."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================================================================
# 3. CHARGEMENT DES MÉTADONNÉES ET CRÉATION DES LABELS
# ==================================================================

metadata_path = PROJECT_ROOT / "data" / "all_metadata.csv"
metadata = pd.read_csv(metadata_path)

classification_table = prepare_classification_table(
    metadata=metadata,
    target_column="eyes_code",
    allowed_classes=["YO", "YF"],
    label_map={"YO": 0, "YF": 1},
)


# ==================================================================
# 4. EXÉCUTION D'UNE EPOCH
# ==================================================================

def run_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None = None,
):
    """Exécute une passe (train si optimizer fourni, sinon validation)."""

    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

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

            current_batch_size = labels.size(0)
            # Pondération par la taille réelle du batch (le dernier peut
            # être plus petit) pour obtenir une moyenne exacte à la fin.
            total_loss += loss.item() * current_batch_size

            predicted_classes = predictions.argmax(dim=1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_examples += current_batch_size

    if total_examples == 0:
        raise ValueError(
            "Le DataLoader ne contient aucun exemple. "
            "Vérifie les dyades et la classification_table."
        )

    average_loss = total_loss / total_examples
    accuracy = correct_predictions / total_examples
    return average_loss, accuracy


# ==================================================================
# 5. ÉVALUATION DÉTAILLÉE D'UN FOLD (matrice de confusion, confiance)
# ==================================================================

def collect_predictions(model: nn.Module, loader):
    """Calcule labels, prédictions et probabilités sur tout un loader."""
    model.eval()

    all_labels, all_predictions, all_probabilities = [], [], []

    with torch.no_grad():
        for eeg, labels in loader:
            logits = model(eeg)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_labels.append(labels.numpy())
            all_predictions.append(predictions.numpy())
            all_probabilities.append(probabilities.numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_predictions),
        np.concatenate(all_probabilities),
    )


def evaluate_fold(validation_dyad: str, model: nn.Module, loader) -> None:
    """Évalue le meilleur modèle d'un fold : accuracy, confiance des
    prédictions, matrice de confusion et classification report.

    Le rapport est affiché en console ET sauvegardé dans
    results/<experiment>/fold_<dyade>/evaluation_report.txt
    """
    labels, predictions, probabilities = collect_predictions(model, loader)
    correct_mask = predictions == labels

    lines = [
        f"Accuracy : {correct_mask.mean():.4f}",
        f"Confiance moyenne : {probabilities.max(axis=1).mean():.4f}",
    ]

    # Confiance séparée bonnes prédictions / erreurs : un modèle fiable
    # doit être plus confiant quand il a raison que quand il se trompe.
    if correct_mask.any():
        lines.append(
            "Confiance moyenne (bonnes prédictions) : "
            f"{probabilities[correct_mask].max(axis=1).mean():.4f}"
        )
    if (~correct_mask).any():
        lines.append(
            "Confiance moyenne (erreurs) : "
            f"{probabilities[~correct_mask].max(axis=1).mean():.4f}"
        )

    lines.append(f"Prédictions classe 0 (YO) : {(predictions == 0).sum()}")
    lines.append(f"Prédictions classe 1 (YF) : {(predictions == 1).sum()}")

    matrix = confusion_matrix(labels, predictions)
    report = classification_report(
        labels, predictions, digits=3, zero_division=0
    )

    print("\n".join(lines))
    print(matrix)
    print(report)

    fold_dir = RESULTS_DIR / f"fold_{validation_dyad}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    with open(fold_dir / "evaluation_report.txt", "w") as report_file:
        report_file.write("\n".join(lines))
        report_file.write("\n\nMatrice de confusion :\n")
        report_file.write(str(matrix))
        report_file.write("\n\nClassification report :\n")
        report_file.write(report)


# ==================================================================
# 6. SAUVEGARDE DES COURBES D'UN FOLD
# ==================================================================

def save_fold_results(
    validation_dyad: str,
    history: dict[str, list[float]],
) -> None:
    """Sauvegarde l'historique (CSV) et les courbes (PNG) d'un fold."""

    fold_dir = RESULTS_DIR / f"fold_{validation_dyad}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    number_of_recorded_epochs = len(history["train_loss"])
    epochs = range(1, number_of_recorded_epochs + 1)

    history_table = pd.DataFrame(history)
    history_table.insert(loc=0, column="epoch", value=epochs)
    history_table.to_csv(fold_dir / "history.csv", index=False)

    # --- Loss ---
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["validation_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss — dyade de validation : {validation_dyad}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fold_dir / "loss_curve.png", dpi=150)
    plt.close()

    # --- Accuracy ---
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(
        epochs, history["validation_accuracy"], label="Validation Accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy — dyade de validation : {validation_dyad}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fold_dir / "accuracy_curve.png", dpi=150)
    plt.close()


# ==================================================================
# 7. ENTRAÎNEMENT D'UN SEUL FOLD
# ==================================================================

def train_one_fold(
    validation_dyad: str,
    train_dyads: list[str],
) -> tuple[dict, float, float, int]:
    """Entraîne un modèle neuf pour un fold de la LODO, puis évalue le
    meilleur checkpoint sur la dyade de validation."""

    # Vérifications anti fuite de données.
    if validation_dyad in train_dyads:
        raise ValueError(
            f"{validation_dyad} est présente à la fois "
            "dans train_dyads et dans validation_dyad."
        )
    overlap_train_test = set(train_dyads) & set(TEST_DYADS)
    if overlap_train_test:
        raise ValueError(
            "Certaines dyades sont présentes à la fois dans le train "
            f"et le test : {overlap_train_test}"
        )

    train_loader, validation_loader, _ = create_participant_dataloaders(
        classification_table=classification_table,
        dataset_root=PROJECT_ROOT / "data" / "data_toy_repaired",
        train_dyads=train_dyads,
        validation_dyads=[validation_dyad],
        test_dyads=TEST_DYADS,
        batch_size=BATCH_SIZE,
    )

    # Même seed avant chaque fold : initialisation comparable d'un fold
    # à l'autre.
    set_seed(RANDOM_SEED)

    # Modèle et optimiseur neufs à chaque fold, pour éviter toute fuite
    # d'information d'un fold vers le suivant.
    model = Small_CNN_EEG(
        number_of_eeg_channels=32,
        number_of_classes=2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Nombre de paramètres entraînables : "
        f"{number_of_parameters:,}"
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    best_validation_loss = float("inf")
    best_validation_accuracy = None
    best_epoch = None
    # Nombre d'epochs consécutives sans amélioration suffisante de la validation loss.
    epochs_without_improvement = 0
    best_model_path = MODELS_DIR / f"best_model_fold_{validation_dyad}.pt"

    for epoch_index in range(NUMBER_OF_EPOCHS):
        epoch_number = epoch_index + 1

        train_loss, train_accuracy = run_epoch(
            model=model, loader=train_loader, optimizer=optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model=model, loader=validation_loader
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        # Le meilleur modèle est choisi sur la validation loss (elle
        # reflète aussi la confiance des prédictions, pas juste l'accuracy).
        # if validation_loss < best_validation_loss:
        #     best_validation_loss = validation_loss
        #     best_validation_accuracy = validation_accuracy
        #     best_epoch = epoch_number
        #     torch.save(model.state_dict(), best_model_path)


        # Une amélioration est considérée comme significative seulement si
        # la validation loss diminue d'au moins EARLY_STOPPING_MIN_DELTA.
        improved = (
                validation_loss
                < best_validation_loss - EARLY_STOPPING_MIN_DELTA
        )

        if improved:

            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch_number

            torch.save(model.state_dict(),best_model_path)

            # Une amélioration a été observée :
            # on remet le compteur à zéro.
            epochs_without_improvement = 0

        else:

            # La validation loss ne s'est pas suffisamment améliorée.
            epochs_without_improvement += 1

        print(
            f"Fold {validation_dyad} | "
            f"Epoch {epoch_number:03d}/{NUMBER_OF_EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.4f} | "
            f"Val loss: {validation_loss:.4f} | "
            f"Val acc: {validation_accuracy:.4f}"
        )
        print(
            f"Epochs sans amélioration : "
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"\nEarly stopping — fold {validation_dyad}"
            )

            print(
                f"Aucune amélioration de la validation loss "
                f"pendant {EARLY_STOPPING_PATIENCE} epochs."
            )

            print(
                f"Meilleure epoch : {best_epoch} | "
                f"Meilleure validation loss : "
                f"{best_validation_loss:.4f}"
            )

            break

    if best_epoch is None or best_validation_accuracy is None:
        raise RuntimeError("Aucun meilleur modèle n'a été enregistré.")

    save_fold_results(validation_dyad=validation_dyad, history=history)

    print(
        f"\n[Fold validation={validation_dyad}] "
        f"meilleure validation loss = {best_validation_loss:.4f} | "
        f"accuracy correspondante = {best_validation_accuracy:.4f} | "
        f"epoch = {best_epoch}\n"
    )

    # Recharge les poids du MEILLEUR epoch (pas ceux du dernier) avant
    # de calculer la matrice de confusion et le classification report.
    model.load_state_dict(torch.load(best_model_path))
    evaluate_fold(
        validation_dyad=validation_dyad, model=model, loader=validation_loader
    )

    return history, best_validation_loss, best_validation_accuracy, best_epoch


# ==================================================================
# 8. GRAPHIQUES GLOBAUX DE COMPARAISON
# ==================================================================

def save_global_comparison(
    all_histories: dict[str, dict[str, list[float]]],
) -> None:
    """Compare les performances de validation de tous les folds."""

    # epochs = range(1, NUMBER_OF_EPOCHS + 1)

    plt.figure(figsize=(10, 6))

    for validation_dyad, history in all_histories.items():
        fold_epochs = range(
            1,
            len(history["validation_loss"]) + 1,
        )

        plt.plot(
            fold_epochs,
            history["validation_loss"],
            label=f"Dyade {validation_dyad}",
        )

    plt.figure(figsize=(10, 6))

    for validation_dyad, history in all_histories.items():
        fold_epochs = range(
            1,
            len(history["validation_accuracy"]) + 1,
        )

        plt.plot(
            fold_epochs,
            history["validation_accuracy"],
            label=f"Dyade {validation_dyad}",
        )


# ==================================================================
# 9. PROGRAMME PRINCIPAL
# ==================================================================

def main() -> None:
    """Exécute les sept folds du protocole B."""

    set_seed(RANDOM_SEED)

    all_histories = {}
    fold_summary = []

    for validation_dyad in DEVELOPMENT_DYADS:
        train_dyads = [
            dyad for dyad in DEVELOPMENT_DYADS if dyad != validation_dyad
        ]

        print("=" * 70)
        print(f"FOLD - dyade de validation : {validation_dyad}")
        print(f"Train dyads      : {train_dyads}")
        print(f"Validation dyad  : {[validation_dyad]}")
        print(f"Test dyads isolés: {TEST_DYADS}")
        print("=" * 70)

        (
            history,
            best_validation_loss,
            best_validation_accuracy,
            best_epoch,
        ) = train_one_fold(
            validation_dyad=validation_dyad, train_dyads=train_dyads
        )

        all_histories[validation_dyad] = history

        # "best" = même epoch que la meilleure validation loss.
        # "final" = dernière epoch, gardée pour observer l'overfitting.
        fold_summary.append(
            {
                "validation_dyad": validation_dyad,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "best_validation_accuracy": best_validation_accuracy,
                "final_train_loss": history["train_loss"][-1],
                "final_train_accuracy": history["train_accuracy"][-1],
                "final_validation_loss": history["validation_loss"][-1],
                "final_validation_accuracy": history["validation_accuracy"][-1],
            }
        )

    summary_table = pd.DataFrame(fold_summary)
    summary_table.to_csv(RESULTS_DIR / "lodo_cv_summary.csv", index=False)

    mean_accuracy = summary_table["best_validation_accuracy"].mean()
    std_accuracy = summary_table["best_validation_accuracy"].std()
    mean_loss = summary_table["best_validation_loss"].mean()
    std_loss = summary_table["best_validation_loss"].std()

    print("\n" + "=" * 70)
    print("RÉSUMÉ GLOBAL DE LA CROSS-VALIDATION")
    print("=" * 70)
    print(summary_table.to_string(index=False))
    print()
    print(f"Best validation accuracy moyenne : {mean_accuracy:.4f}")
    print(f"Écart-type de l'accuracy : {std_accuracy:.4f}")
    print(f"Best validation loss moyenne : {mean_loss:.4f}")
    print(f"Écart-type de la loss : {std_loss:.4f}")

    save_global_comparison(all_histories)


# Ne s'exécute que si le fichier est lancé directement (pas à l'import).
if __name__ == "__main__":
    main()