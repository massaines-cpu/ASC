from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BBC2Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.metadata = pd.read_csv(self.root_dir / "labels.csv")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        a_all = np.load(
            self.root_dir / row["participant_a_file"],
            mmap_mode="r"
        )

        b_all = np.load(
            self.root_dir / row["participant_b_file"],
            mmap_mode="r"
        )

        epoch_idx = int(row["epoch_index"])

        eeg_a = torch.from_numpy(
            np.array(a_all[epoch_idx], copy=True)
        )

        eeg_b = torch.from_numpy(
            np.array(b_all[epoch_idx], copy=True)
        )

        label = torch.tensor(
            float(row["label"]),
            dtype=torch.float32
        )

        return eeg_a, eeg_b, label


if __name__ == "__main__":
    dataset_root = PROJECT_ROOT / "data" / "bbc2" / "fake_bbc2"

    print("Racine projet :", PROJECT_ROOT)
    print("Dataset :", dataset_root)
    print("labels.csv existe :", (dataset_root / "labels.csv").exists())

    dataset = BBC2Dataset(dataset_root)

    eeg_a, eeg_b, label = dataset[0]

    print("Nombre d'exemples :", len(dataset))
    print("EEG A :", eeg_a.shape)
    print("EEG B :", eeg_b.shape)
    print("Label :", label.item())