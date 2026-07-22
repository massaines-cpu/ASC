# Entraînement du modèle au niveau participant

# Ce fichier :
# 1. charge les métadonnées
# 2. prépare les labels
# 3. crée les DataLoaders
# 4. crée le modèle
# 5. entraîne le modèle pendant une époque.

from pathlib import Path

import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from src.dataset.labels import prepare_classification_table
from src.dataset.dataloader_participant import (
    create_participant_dataloaders,
)
from src.models.participant_model import (
    SimpleParticipantClassifier,
)


#racine du projet ASC

PROJECT_ROOT = Path(__file__).resolve().parents[2]


#lecture des métadonnées

metadata = pd.read_csv(
    PROJECT_ROOT / "data" / "all_metadata.csv"
)


#préparation de la classification :

# YO -> 0
# YF -> 1

classification_table = prepare_classification_table(
    metadata=metadata,
    target_column="eyes_code",
    allowed_classes=["YO", "YF"],
    label_map={
        "YO": 0,
        "YF": 1,
    },
)


#création des DataLoaders

train_loader, validation_loader, test_loader = (
    create_participant_dataloaders(
        classification_table=classification_table,
        dataset_root=PROJECT_ROOT / "data" / "data_toy",
        train_dyads=[
            "J2",
            "J4",
            "J5",
            "J7",
            "J8",
            "J1",
        ],
        validation_dyads=["J10"],
        test_dyads=["J15"],
        batch_size=5,
    )
)



#création du modèle

model = SimpleParticipantClassifier()


#fonction de perte

#crossEntropyLoss attend :

# predictions : (batch_size, number_of_classes)
# labels      : (batch_size)

criterion = nn.CrossEntropyLoss()


# Optimiseur

# Adam modifiera les paramètres du modèle afin de réduire
# progressivement la loss.

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)


# Entraînement pendant une époque

train_losses = []
train_accuracies = []
validation_losses = []
validation_accuracies = []
# Active le mode entraînement

model.train()
# Nombre de passages complets sur le jeu d'entraînement
number_of_epochs = 100

for epoch in range(number_of_epochs):
    # Somme des losses de tous les batches.

    total_loss = 0.0
    # Nombre total de prédictions correctes.

    number_of_correct_predictions = 0
    number_of_examples = 0
    # Parcours de tous les batches du train_loader.

    for eeg, labels in train_loader:

        # 1. Remise à zéro des gradients

        optimizer.zero_grad()

        # 2. Forward pass

        # eeg :
        # (batch_size, 32, 5120)

        # predictions :
        # (batch_size, 2)

        predictions = model(eeg)

        # 3. Calcul de la loss

        loss = criterion(
            predictions,
            labels,
        )

        # 4. Backpropagation

        # Calcul des gradients de la loss par rapport à chaque
        # paramètre du modèle.

        loss.backward()

        # 5. Mise à jour des paramètres

        optimizer.step()

        # Enregistrement de la loss du batch

        total_loss += loss.item()

        # Conversion des logits en classes prédites

        # Exemple :

        # [2.1, -0.5] -> classe 0
        # [0.3,  1.7] -> classe 1

        predicted_classes = predictions.argmax(dim=1)

        # Nombre de prédictions correctes dans ce batch

        number_of_correct_predictions += (
            predicted_classes == labels
        ).sum().item()

        # Nombre de participants dans ce batch

        # Le dernier batch peut contenir moins de 5 exemples.

        number_of_examples += labels.size(0)


# Résultats moyens de l'époque

    average_loss = total_loss / len(train_loader)

    accuracy = (
        number_of_correct_predictions
        / number_of_examples
    )
    train_losses.append(average_loss)
    train_accuracies.append(accuracy)

    print(
        f"Epoch {epoch + 1}/{number_of_epochs}"
    )

    print(
        f"Loss : {average_loss:.4f}"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print("-" * 40)

plt.figure(figsize=(8, 5))

plt.plot(
    train_accuracies,
    label="Train Accuracy",
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Accuracy au cours de l'entraînement")

plt.grid(True)

plt.legend()

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    label="Train Loss",
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Loss au cours de l'entraînement")

plt.grid(True)

plt.legend()

plt.show()