#lit csv

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from labels import prepare_classification_table

# faire le lien entre :
#
#   all_metadata.csv
#           ↓
#   fichier .npy correspondant
#           ↓
#   chargement de l'EEG
#           ↓
#   conversion en tenseurs PyTorch
#           ↓
#   retour de (eeg_a, eeg_b, label)
#
# Cette classe sera ensuite utilisée par le DataLoader pour
# construire les batches d'entraînement

PROJECT_ROOT = Path(__file__).resolve().parents[2]

metadata_path = PROJECT_ROOT / "data" / "all_metadata.csv"

metadata = pd.read_csv(metadata_path)

# cette classe représente l'ensemble des exemples du dataset.
# elle ne charge PAS tous les EEG en mémoire.
#
# elle sait simplement :
#   - combien d'exemples existent
#   - comment charger UN exemple lorsqu'on le demande
class MultiBrainDataset(Dataset):
    def __init__(self, dataframe, dataset_root):
        self.dataframe = dataframe.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # récupère la ligne correspondant à l'indice demandé, exemple
        # filnename, dyad, label
        row = self.dataframe.iloc[idx]
        # construit automatiquement le chemin vers le fichier

        file_path = (
            self.dataset_root
            / row["dyad_id"]
            / "epochs"
            / row["filename"]
        )
        # Charge le fichier .npy.
        data = np.load(file_path)

        eeg_a = data[0]
        eeg_b = data[1]
        # Conversion NumPy → Tensor PyTorch
        eeg_a = torch.tensor(
            eeg_a,
            dtype=torch.float32,
        )

        eeg_b = torch.tensor(
            eeg_b,
            dtype=torch.float32,
        )
        # conversion du label en entier PyTorch

        label = torch.tensor(
            row["label"],
            dtype=torch.long,
        )
        # Retourne un exemple complet
        return eeg_a, eeg_b, label




if __name__ == "__main__":

    # préparation de la table de classification
    metadata1 = prepare_classification_table(
        metadata,
        target_column="eyes_code",
        allowed_classes=["YO", "YF"],
        label_map={
            "YO": 0,
            "YF": 1,
        },
    )

    # création du dataset
    dataset = MultiBrainDataset(
        metadata1,
        dataset_root=PROJECT_ROOT / "data" / "data_toy",
    )

    print("nombre d'exemples :", len(dataset))

    eeg_a, eeg_b, label = dataset[0]

    print("EEG A :", eeg_a.shape)
    print("EEG B :", eeg_b.shape)
    print("Label :", label)