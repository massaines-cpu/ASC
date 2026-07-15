from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from bbc2_dataset import BBC2Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "bbc2" / "fake_bbc2"


def create_dataloaders(
    batch_size: int = 16,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    dataset = BBC2Dataset(DATASET_ROOT)

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders(batch_size=16)

    eeg_a, eeg_b, labels = next(iter(train_loader))

    print("Nombre de batchs train :", len(train_loader))
    print("Nombre de batchs validation :", len(val_loader))

    print("EEG A :", eeg_a.shape)
    print("EEG B :", eeg_b.shape)
    print("Labels :", labels.shape)

    print("Labels du batch :", labels)