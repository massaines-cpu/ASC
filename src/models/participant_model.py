import torch
from torch import nn
from pathlib import Path
import pandas as pd

from src.dataset.labels import prepare_classification_table
from src.dataset.dataloader_participant import create_participant_dataloaders


# Récupère automatiquement le chemin vers la racine du projet.

# Cela permet d'accéder aux fichiers de données sans dépendre
# du dossier depuis lequel le script est lancé.

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Lecture du fichier contenant toutes les métadonnées EEG.

metadata = pd.read_csv(
    PROJECT_ROOT / "data" / "all_metadata.csv"
)


# Création de la table de classification.

# On conserve uniquement les deux conditions :

# YO -> label 0
# YF -> label 1

classification_table = prepare_classification_table(
    metadata,
    target_column="eyes_code",
    allowed_classes=["YO", "YF"],
    label_map={
        "YO": 0,
        "YF": 1,
    },
)


# Création des DataLoaders.

# Les données sont séparées par dyade afin d'éviter qu'une
# même dyade apparaisse à la fois dans l'entraînement et
# dans le test.

train_loader, validation_loader, test_loader = (
    create_participant_dataloaders(
        classification_table=classification_table,
        dataset_root=PROJECT_ROOT / "data" / "data_toy",
        train_dyads=["J2", "J4", "J5", "J7", "J8", "J1"],
        validation_dyads=["J10"],
        test_dyads=["J15"],
        batch_size=5,
    )
)


# Modèle de classification le plus simple possible.

# Architecture :
#
# EEG (32 × 5120)
#        ↓
# Flatten
#        ↓
# Couche linéaire
#        ↓
# Deux scores de classes :
#
# classe 0 -> YO
# classe 1 -> YF

class SimpleParticipantClassifier(nn.Module):

    def __init__(
        self,
        number_of_channels=32,
        number_of_timepoints=5120,
    ):
        super().__init__()

        # Transforme chaque EEG :

        # (32, 5120)

        # en un vecteur de taille :

        # 32 × 5120 = 163840

        # Une couche linéaire attend une entrée sous forme
        # de vecteur.

        self.flatten = nn.Flatten()


        # Couche de classification.

        # Entrée :
        # 163840 valeurs

        # Sortie :
        # 2 logits (un score par classe)

        # Ces deux valeurs correspondent aux scores associés
        # aux classes YO et YF.

        self.classifier = nn.Linear(
            number_of_channels * number_of_timepoints,
            2,
        )

    def forward(self, eeg):
        """
        Réalise le passage avant (forward pass).

        Paramètre
        ---------
        eeg :
            Batch de taille :

            (batch_size, 32, 5120)

        Retour
        ------
        predictions :
            Scores de taille :

            (batch_size, 2)
        """

        # Mise à plat de chaque EEG
        eeg = self.flatten(eeg)

        # Calcul des scores de classification
        predictions = self.classifier(eeg)

        return predictions


# Test du modèle

# Cette partie permet simplement de vérifier que :

# - le DataLoader fonctionne ;
# - le modèle accepte les données ;
# - les dimensions des sorties sont correctes.

# Récupération du premier batch d'entraînement
eeg, labels = next(iter(train_loader))

# Création du modèle
model = SimpleParticipantClassifier()

# Passage du batch dans le réseau
predictions = model(eeg)

# Vérification des dimensions obtenues
print("EEG :", eeg.shape)
print("Prédictions :", predictions.shape)
print("Labels :", labels.shape)