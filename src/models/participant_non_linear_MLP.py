import torch
from torch import nn
from pathlib import Path
import pandas as pd

from src.dataset.labels import prepare_classification_table
from src.dataset.dataloader_participant import (
    create_participant_dataloaders,
)


# Récupération automatique du chemin vers la racine du projet.
#
# Cette méthode permet d'accéder aux données indépendamment
# du dossier depuis lequel le script Python est exécuté.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Lecture du fichier contenant les métadonnées EEG.
metadata = pd.read_csv(
    PROJECT_ROOT / "data" / "all_metadata.csv"
)


# Création de la table utilisée pour la classification.
#
# Seules les conditions expérimentales YO et YF sont conservées :
#
# YO : Yeux Ouverts -> classe 0
# YF : Yeux Fermés -> classe 1
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
#
# Les données sont séparées au niveau des dyades.
# Cette séparation empêche qu'une même dyade soit présente
# simultanément dans l'entraînement et dans l'évaluation.
#
# Cette partie est uniquement utilisée pour vérifier le modèle.
# Pour les expériences finales, il faudra utiliser la boucle LODO
# commune aux autres architectures.

class NonLinearParticipantMLP(nn.Module):
    """
    MLP non linéaire pour la classification de signaux EEG.

    Architecture
    ------------
    EEG de forme (32, 5120)
            ↓
    Mise à plat
            ↓
    Couche linéaire : 163840 -> 128
            ↓
    Activation ReLU
            ↓
    Couche linéaire : 128 -> 2
            ↓
    Deux logits correspondant aux classes YO et YF

    Paramètres
    ----------
    number_of_channels : int
        Nombre de canaux EEG présents dans chaque exemple.

    number_of_timepoints : int
        Nombre de points temporels présents dans chaque canal.

    hidden_layer_size : int
        Nombre de neurones dans la couche cachée.

    number_of_classes : int
        Nombre de classes à prédire.

    Remarque scientifique
    ---------------------
    La fonction ReLU introduit une non-linéarité dans le modèle.
    Contrairement au modèle linéaire, ce réseau peut donc apprendre
    des relations plus complexes entre les amplitudes EEG et la
    condition expérimentale.
    """

    def __init__(
        self,
        number_of_channels: int = 32,
        number_of_timepoints: int = 5120,
        hidden_layer_size: int = 128,
        number_of_classes: int = 2,
    ):
        super().__init__()

        self.number_of_channels = number_of_channels
        self.number_of_timepoints = number_of_timepoints
        self.hidden_layer_size = hidden_layer_size
        self.number_of_classes = number_of_classes

        # Chaque EEG possède 32 canaux et 5120 points temporels.
        #
        # La taille du vecteur transmis au MLP est donc :
        #
        # 32 × 5120 = 163840 valeurs.
        flattened_input_size = (
            number_of_channels * number_of_timepoints
        )

        # La mise à plat transforme un EEG de forme :
        #
        # (32, 5120)
        #
        # en un vecteur contenant 163840 valeurs.
        #
        # Cette opération ne modifie pas les données :
        # elle change uniquement leur organisation.
        self.flatten = nn.Flatten()

        # La première couche apprend une représentation plus compacte
        # du signal EEG complet.
        #
        # Elle transforme les 163840 valeurs d'entrée en 128 valeurs.
        self.hidden_layer = nn.Linear(
            in_features=flattened_input_size,
            out_features=hidden_layer_size,
        )

        # ReLU introduit la non-linéarité dans le réseau.
        #
        # Sans cette fonction d'activation, la combinaison de plusieurs
        # couches linéaires resterait mathématiquement équivalente à
        # une seule couche linéaire.
        self.activation = nn.ReLU()

        # La dernière couche transforme la représentation cachée
        # en deux scores de classification :
        #
        # indice 0 -> YO
        # indice 1 -> YF
        self.classifier = nn.Linear(
            in_features=hidden_layer_size,
            out_features=number_of_classes,
        )

    def forward(self, eeg):
        """
        Réalise le passage avant du MLP.

        Paramètres
        ----------
        eeg : torch.Tensor
            Batch de signaux EEG de forme :

            (batch_size, 32, 5120)

        Retour
        ------
        logits : torch.Tensor
            Scores de classification de forme :

            (batch_size, 2)

        Les logits ne sont pas transformés en probabilités dans
        le modèle, car CrossEntropyLoss applique directement
        l'opération nécessaire pendant l'entraînement.
        """


        # Passage de :
        #
        # (batch_size, 32, 5120)
        #
        # à :
        #
        # (batch_size, 163840)
        eeg = self.flatten(eeg)

        # Construction d'une représentation cachée du signal.
        hidden_representation = self.hidden_layer(eeg)

        # Introduction de la non-linéarité.
        hidden_representation = self.activation(
            hidden_representation
        )

        # Calcul des deux scores correspondant à YO et YF.
        logits = self.classifier(hidden_representation)

        return logits
