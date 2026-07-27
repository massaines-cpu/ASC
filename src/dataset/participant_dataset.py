# ============================================================
# Dataset PyTorch au niveau du participant
#
# Contrairement au MultiBrainDataset :
#
#   MultiBrainDataset :
#       un exemple = une dyade complète
#       retour = eeg_a, eeg_b, label
#
#   ParticipantDataset :
#       un exemple = un seul participant
#       retour = eeg, label
#
# Un fichier .npy contient deux participants :
#
#   data.shape = (2, 32, 5120)
#
# Le Dataset crée donc deux exemples à partir de chaque fichier :
#
#   participant 0
#   participant 1
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dataset.labels import prepare_classification_table

PROJECT_ROOT = Path(__file__).resolve().parents[2]

metadata_path = PROJECT_ROOT / "data" / "all_metadata.csv"


class ParticipantDataset(Dataset):

    def __init__(self, classification_table, dataset_root):
        """
        Initialise le Dataset au niveau du participant.

        Paramètres
        ----------
        classification_table : pandas.DataFrame
            Table contenant les fichiers EEG et leurs labels.

        dataset_root : str ou Path
            Dossier contenant les dossiers des dyades.
        """

        # On remet les indices de la table à 0, 1, 2, ...
        self.classification_table = (
            classification_table.reset_index(drop=True)
        )

        # On convertit le chemin en objet Path
        self.dataset_root = Path(dataset_root)

    def __len__(self):
        """
        Retourne le nombre total de participants.

        Chaque ligne de classification_table correspond
        à un fichier contenant deux participants.

        Donc :

            nombre de participants
            =
            nombre de fichiers × 2
        """

        return len(self.classification_table) * 2

    def __getitem__(self, idx):
        """
        Charge un seul participant.

        Retour
        ------
        eeg : torch.Tensor
            EEG standardisé de forme (32, 5120).

        label : torch.Tensor
            Label numérique de la condition.
        """

        file_index = idx // 2
        participant_index = idx % 2

        row = self.classification_table.iloc[file_index]

        file_path = (
                self.dataset_root
                / row["dyad_id"]
                / "epochs"
                / row["filename"]
        )

        # Forme attendue : (2, 32, 5120)
        data = np.load(file_path)

        # Sélection d'un participant : (32, 5120)
        eeg = data[participant_index].astype(np.float32)

        # ----------------------------------------------------------
        # Standardisation canal par canal
        # ----------------------------------------------------------
        #
        # Pour chaque électrode, on calcule la moyenne et l'écart-type
        # sur les 5120 points temporels.
        #
        # Après standardisation, chaque canal aura approximativement :
        #
        # moyenne = 0
        # écart-type = 1
        channel_mean = eeg.mean(
            axis=1,
            keepdims=True,
        )

        channel_std = eeg.std(
            axis=1,
            keepdims=True,
        )

        eeg = (eeg - channel_mean) / (channel_std + 1e-8)

        # Comme eeg est déjà en float32, torch.from_numpy évite une copie
        # supplémentaire inutile.
        eeg = torch.from_numpy(eeg)

        label = torch.tensor(
            row["label"],
            dtype=torch.long,
        )

        return eeg, label




if __name__ == "__main__":

    # Lecture des métadonnées
    all_metadata = pd.read_csv(metadata_path)

    # Préparation de la classification YO / YF
    classification_table = prepare_classification_table(
        metadata=all_metadata,
        target_column="eyes_code",
        allowed_classes=["YO", "YF"],
        label_map={
            "YO": 0,
            "YF": 1,
        },
    )

    # Création du Dataset participant
    participant_dataset = ParticipantDataset(
        classification_table=classification_table,
        dataset_root=PROJECT_ROOT / "data" / "data_toy",
    )

    print(
        "Nombre de fichiers :",
        len(classification_table),
    )

    print(
        "Nombre de participants :",
        len(participant_dataset),
    )

    # Charge le premier participant du premier fichier
    eeg_0, label_0 = participant_dataset[0]

    # Charge le deuxième participant du premier fichier
    eeg_1, label_1 = participant_dataset[1]

    print("Participant 0 :", eeg_0.shape)
    print("Label 0 :", label_0)

    print("Participant 1 :", eeg_1.shape)
    print("Label 1 :", label_1)