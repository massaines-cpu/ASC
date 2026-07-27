# Entraînement du modèle au niveau participant
#
# Ce fichier :
# 1. charge les métadonnées
# 2. prépare les labels
# 3. crée les DataLoaders
# 4. crée le modèle
# 5. entraîne le modèle pendant plusieurs époques
# 6. valide le modèle après chaque époque
# 7. sauvegarde le meilleur modèle + l'historique des métriques
# 8. affiche les courbes d'apprentissage

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from src.dataset.labels import prepare_classification_table
from src.dataset.dataloader_participant import create_participant_dataloaders
from src.models.participant_linear_model import SimpleParticipantClassifier


# ------------------------------------------------------------------
# 1. Racine du projet
# ------------------------------------------------------------------
# __file__ = chemin de ce fichier
# .resolve() -> chemin absolu
# .parents[2] -> on remonte jusqu'à la racine du projet ASC
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------
# 2. Métadonnées et labels
# ------------------------------------------------------------------
metadata = pd.read_csv(PROJECT_ROOT / "data" / "all_metadata.csv")

# YO -> 0 (yeux ouverts)
# YF -> 1 (yeux fermés)
classification_table = prepare_classification_table(
    metadata=metadata,
    target_column="eyes_code",
    allowed_classes=["YO", "YF"],
    label_map={"YO": 0, "YF": 1},
)


# ------------------------------------------------------------------
# 3. DataLoaders
# ------------------------------------------------------------------
# Important : les dyades (séances d'enregistrement) ne doivent jamais
# être partagées entre train / validation / test, sinon le modèle peut
# apprendre des caractéristiques propres à une séance -> fuite
# d'information (data leakage) et validation artificiellement bonne.


#Même si le modèle apprend au niveau participant, je garde le découpage par dyade pour éviter le data leakag
train_loader, validation_loader, test_loader = create_participant_dataloaders(
    classification_table=classification_table,
    dataset_root=PROJECT_ROOT / "data" / "data_toy",
    train_dyads=["J2", "J4", "J5", "J7", "J8", "J1"],
    validation_dyads=["J10"],
    test_dyads=["J15"],
    batch_size=5,
)


# ------------------------------------------------------------------
# 4. Modèle, loss, optimiseur
# ------------------------------------------------------------------
model = SimpleParticipantClassifier()

# CrossEntropyLoss attend :
#   predictions : (batch_size, number_of_classes) -> ici (batch_size, 2)
#   labels      : (batch_size,)
criterion = nn.CrossEntropyLoss()

# Adam ajuste progressivement les poids du modèle pour réduire la loss.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ------------------------------------------------------------------
# 5. Paramètres d'entraînement et historique des métriques
# ------------------------------------------------------------------
number_of_epochs = 100

train_losses, train_accuracies = [], []
validation_losses, validation_accuracies = [], []

# Suivi du meilleur modèle (défini AVANT la boucle : sinon il serait
# réinitialisé à chaque époque et on ne saurait jamais quel est le
# vrai meilleur modèle sur l'ensemble de l'entraînement).
best_validation_loss = float("inf")
best_epoch = None
best_model_path = PROJECT_ROOT / "models" / "best_model.pt"
best_model_path.parent.mkdir(parents=True, exist_ok=True)


def run_epoch(loader, training: bool):
    """
    Exécute une époque complète.

    Parameters
    ----------
    loader :
        DataLoader utilisé (train ou validation).

    training :
        True  -> apprentissage
        False -> validation

    Returns
    -------
    average_loss : float

    accuracy : float
    """
    model.train() if training else model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    # torch.no_grad() : pas de calcul de gradients en validation,
    # car on ne fait ni backward() ni optimizer.step().
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for eeg, labels in loader:
            # eeg : (batch_size, 32, 5120) -> predictions : (batch_size, 2)
            predictions = model(eeg)
            loss = criterion(predictions, labels)

            if training:
                optimizer.zero_grad()  # PyTorch accumule les gradients par défaut
                loss.backward()        # calcule les gradients
                optimizer.step()       # met à jour les poids

            total_loss += loss.item()

            # Logits -> classe prédite, ex : [2.1, -0.5] -> classe 0
            predicted_classes = predictions.argmax(dim=1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_examples += labels.size(0)  # le dernier batch peut être plus petit

    average_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_examples
    return average_loss, accuracy


# ------------------------------------------------------------------
# 6. Boucle d'entraînement
# ------------------------------------------------------------------
for epoch in range(number_of_epochs):

    train_loss, train_accuracy = run_epoch(train_loader, training=True)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    validation_loss, validation_accuracy = run_epoch(validation_loader, training=False)
    validation_losses.append(validation_loss)
    validation_accuracies.append(validation_accuracy)

    # Sauvegarde du modèle seulement s'il améliore la meilleure loss
    # de validation observée jusqu'ici.
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), best_model_path)

    print(f"Epoch {epoch + 1}/{number_of_epochs}")
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Accuracy : {train_accuracy:.4f}")
    print(f"Validation Loss : {validation_loss:.4f}")
    print(f"Validation Accuracy : {validation_accuracy:.4f}")
    print("-" * 50)

print(f"Meilleur modèle : époque {best_epoch} (validation loss = {best_validation_loss:.4f})")


# ------------------------------------------------------------------
# 7. Courbes d'apprentissage
# ------------------------------------------------------------------
epochs = range(1, number_of_epochs + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, validation_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss au cours de l'entraînement")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, validation_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy au cours de l'entraînement")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()