# Entraînement du modèle au niveau participant

# Ce fichier :
# 1. charge les métadonnées
# 2. prépare les labels
# 3. crée les DataLoaders
# 4. crée le modèle
# 5. entraîne le modèle pendant plusieurs époques
# 6. valide le modèle après chaque époque
# 7. sauvegarde l'historique des métriques
# 8. affiche les courbes d'apprentissage

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from src.dataset.labels import prepare_classification_table
from src.dataset.dataloader_participant import (
    create_participant_dataloaders,
)
from src.models.participant_model import (
    SimpleParticipantClassifier,
)


# 1. Racine du projet

# __file__ correspond au chemin de ce fichier Python.
#
# .resolve() transforme ce chemin en chemin absolu.
#
# .parents[2] permet de remonter jusqu'à la racine du projet ASC.
PROJECT_ROOT = Path(__file__).resolve().parents[2]



# 2. Lecture des métadonnées

metadata = pd.read_csv(
    PROJECT_ROOT / "data" / "all_metadata.csv"
)



# 3. Préparation de la table de classification


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


# 4. Création des DataLoaders


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



# 5. Création du modèle

model = SimpleParticipantClassifier()

# 6. Fonction de perte


# CrossEntropyLoss attend :

# predictions : (batch_size, number_of_classes)
# labels      : (batch_size)

# Ici :

# predictions : (batch_size, 2)
# labels      : (batch_size)

criterion = nn.CrossEntropyLoss()

# 7. Optimiseur

# Adam modifiera progressivement les paramètres du modèle
# afin de réduire la loss d'entraînement.

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)



# 8. Paramètres de l'entraînement


# Nombre de passages complets sur le jeu d'entraînement.
number_of_epochs = 100



# 9. Historique des métriques

# Ces listes contiendront une valeur par époque.

train_losses = []
train_accuracies = []

validation_losses = []
validation_accuracies = []


# 10. Boucle sur les époques


for epoch in range(number_of_epochs):

    # ========================================================
    # PARTIE ENTRAÎNEMENT
    # ========================================================

    # Active le mode entraînement.
    #
    # Cela sera important plus tard pour les couches comme :
    # - Dropout
    # - BatchNorm
    model.train()

    # Somme des losses de tous les batches d'entraînement.
    total_train_loss = 0.0

    # Nombre total de prédictions correctes sur le train.
    train_correct_predictions = 0

    # Nombre total d'exemples vus pendant l'entraînement.
    train_examples = 0

    # Parcours de tous les batches du train_loader.
    for eeg, labels in train_loader:

        # 1. Remise à zéro des gradients

        # PyTorch additionne les gradients par défaut.
        # Il faut donc les remettre à zéro avant chaque batch.
        optimizer.zero_grad()

        # 2. Forward pass

        # eeg :
        # (batch_size, 32, 5120)
        #
        # predictions :
        # (batch_size, 2)
        predictions = model(eeg)

        # 3. Calcul de la loss


        loss = criterion(
            predictions,
            labels,
        )

        # 4. Backpropagation

        # Calcule les gradients de la loss par rapport
        # à tous les paramètres entraînables du modèle.
        loss.backward()

        # 5. Mise à jour des paramètres

        # Adam utilise les gradients calculés pour modifier
        # les poids du modèle.
        optimizer.step()

        # 6. Accumulation de la loss

        # loss.item() transforme le tenseur contenant la loss
        # en nombre Python.
        total_train_loss += loss.item()

        # 7. Conversion des logits en classes

        # Exemple :
        #
        # [2.1, -0.5] -> classe 0
        # [0.3,  1.7] -> classe 1
        predicted_classes = predictions.argmax(dim=1)


        # 8. Nombre de bonnes prédictions

        train_correct_predictions += (
            predicted_classes == labels
        ).sum().item()

        # 9. Nombre d'exemples du batch

        # Le dernier batch peut contenir moins de 5 exemples.
        train_examples += labels.size(0)

    # RÉSULTATS MOYENS DU TRAIN POUR CETTE ÉPOQUE

    average_train_loss = (
        total_train_loss
        / len(train_loader)
    )

    train_accuracy = (
        train_correct_predictions
        / train_examples
    )

    # Sauvegarde des résultats de cette époque.
    train_losses.append(
        average_train_loss
    )

    train_accuracies.append(
        train_accuracy
    )


    # PARTIE VALIDATION

    # Active le mode évaluation.
    #
    # Le modèle ne doit pas modifier son comportement comme
    # pendant l'entraînement.
    model.eval()

    # Somme des losses de tous les batches de validation.
    total_validation_loss = 0.0

    # Nombre total de prédictions correctes en validation.
    validation_correct_predictions = 0

    # Nombre total d'exemples de validation.
    validation_examples = 0

    # Désactive le calcul des gradients.
    #
    # En validation :
    # - pas de backward()
    # - pas de optimizer.step()
    #
    # On veut seulement tester le modèle.
    with torch.no_grad():

        # Parcours de tous les batches du validation_loader.
        for eeg, labels in validation_loader:

            # 1. Forward pass

            predictions = model(eeg)

            # 2. Calcul de la loss de validation

            loss = criterion(
                predictions,
                labels,
            )

            # 3. Accumulation de la loss

            total_validation_loss += loss.item()

            # 4. Classes prédites

            predicted_classes = predictions.argmax(dim=1)

            # ------------------------------------------------
            # 5. Nombre de bonnes prédictions
            # ------------------------------------------------

            validation_correct_predictions += (
                predicted_classes == labels
            ).sum().item()

            # ------------------------------------------------
            # 6. Nombre d'exemples
            # ------------------------------------------------

            validation_examples += labels.size(0)

    # ========================================================
    # RÉSULTATS MOYENS DE VALIDATION POUR CETTE ÉPOQUE
    # ========================================================

    average_validation_loss = (
        total_validation_loss
        / len(validation_loader)
    )

    validation_accuracy = (
        validation_correct_predictions
        / validation_examples
    )

    # Sauvegarde des résultats de validation.
    validation_losses.append(
        average_validation_loss
    )

    validation_accuracies.append(
        validation_accuracy
    )


    # ========================================================
    # AFFICHAGE DES RÉSULTATS DE L'ÉPOQUE
    # ========================================================

    print(
        f"Epoch {epoch + 1}/{number_of_epochs}"
    )

    print(
        f"Train Loss : {average_train_loss:.4f}"
    )

    print(
        f"Train Accuracy : {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss : {average_validation_loss:.4f}"
    )

    print(
        f"Validation Accuracy : {validation_accuracy:.4f}"
    )

    print("-" * 50)


# ============================================================
# 11. Axe des époques
# ============================================================

# range commence à 0, mais on veut afficher les époques
# de 1 à number_of_epochs.
epochs = range(
    1,
    number_of_epochs + 1,
)


# ============================================================
# 12. Graphique des losses
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    train_losses,
    label="Train Loss",
)

plt.plot(
    epochs,
    validation_losses,
    label="Validation Loss",
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Loss au cours de l'entraînement"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 13. Graphique des accuracies
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    train_accuracies,
    label="Train Accuracy",
)

plt.plot(
    epochs,
    validation_accuracies,
    label="Validation Accuracy",
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Accuracy au cours de l'entraînement"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()