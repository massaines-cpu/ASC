#lit csv

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from labels import prepare_classification_table

PROJECT_ROOT = Path(__file__).resolve().parents[2]

metadata_path = PROJECT_ROOT / "data" / "all_metadata.csv"

metadata = pd.read_csv(metadata_path)

class MultiBrainDataset(Dataset):
    def __init__(self, dataframe, dataset_root):
        self.dataframe = dataframe.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        file_path = (
            self.dataset_root
            / row["dyad_id"]
            / "epochs"
            / row["filename"]
        )

        data = np.load(file_path)

        eeg_a = data[0]
        eeg_b = data[1]

        eeg_a = torch.tensor(
            eeg_a,
            dtype=torch.float32,
        )

        eeg_b = torch.tensor(
            eeg_b,
            dtype=torch.float32,
        )

        label = torch.tensor(
            row["label"],
            dtype=torch.long,
        )

        return eeg_a, eeg_b, label

if __name__ == "__main__":

    #préparation des labels
    metadata = prepare_classification_table(
        metadata,
        target_column="eyes_code",
        allowed_classes=["YO", "YF"],
        label_map={
            "YO": 0,
            "YF": 1,
        },
    )

    #création du Dataset
    dataset = MultiBrainDataset(
        metadata,
        dataset_root=PROJECT_ROOT / "data" / "data_toy",
    )

    print("Nombre d'exemples :", len(dataset))

    eeg_a, eeg_b, label = dataset[0]

    print("EEG A :", eeg_a.shape)
    print("EEG B :", eeg_b.shape)
    print("Label :", label)