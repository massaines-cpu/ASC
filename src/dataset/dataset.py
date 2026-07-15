#lit csv

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def prepare_classification_table(
    dataframe,
    target_column,
    allowed_classes,
    label_map,
):
    filtered_df = dataframe[
        dataframe[target_column].isin(allowed_classes)
    ].copy()

    filtered_df["label"] = filtered_df[target_column].map(label_map)

    return filtered_df


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

        eeg_a = torch.tensor(data[0], dtype=torch.float32)
        eeg_b = torch.tensor(data[1], dtype=torch.float32)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return eeg_a, eeg_b, label