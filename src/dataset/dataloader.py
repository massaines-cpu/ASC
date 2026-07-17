#cree les dataloader

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from dataset import MultiBrainDataset
from labels import prepare_classification_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "data" / "all_metadata.csv"
DATASET_ROOT = PROJECT_ROOT / "data" / "data_toy"

#sépare le DataFrame selon les identifiants de dyades

def split_by_dyad(
    classification_table,
    train_dyads,
    validation_dyads,
    test_dyads,
):
    train_table = classification_table[
        classification_table["dyad_id"].isin(train_dyads)
    ].copy()

    validation_table = classification_table[
        classification_table["dyad_id"].isin(validation_dyads)
    ].copy()

    test_table = classification_table[
        classification_table["dyad_id"].isin(test_dyads)
    ].copy()

    return train_table, validation_table, test_table


def create_dataloaders(
    classification_table,
    dataset_root,
    train_dyads,
    validation_dyads,
    test_dyads,
    batch_size=8,
):
# Crée les dataLoaders d'entraînement, validation et test.

    train_table, validation_table, test_table = split_by_dyad(
        classification_table,
        train_dyads,
        validation_dyads,
        test_dyads,
    )

    train_dataset = MultiBrainDataset(
        train_table,
        dataset_root,
    )

    validation_dataset = MultiBrainDataset(
        validation_table,
        dataset_root,
    )

    test_dataset = MultiBrainDataset(
        test_table,
        dataset_root,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, validation_loader, test_loader

if __name__ == "__main__":

    all_metadata = pd.read_csv(METADATA_PATH)

    classification_table = prepare_classification_table(
        all_metadata,
        target_column="eyes_code",
        allowed_classes=["YO", "YF"],
        label_map={
            "YO": 0,
            "YF": 1,
        },
    )

    train_loader, validation_loader, test_loader = create_dataloaders(
        classification_table=classification_table,
        dataset_root=DATASET_ROOT,
        train_dyads=["J2", "J4", "J5", "J7"],
        validation_dyads=["J10"],
        test_dyads=["J15"],
        batch_size=8,
    )

    eeg_a, eeg_b, labels = next(iter(train_loader))

    print("Nombre de batchs train :", len(train_loader))
    print("Nombre de batchs validation :", len(validation_loader))
    print("Nombre de batchs test :", len(test_loader))

    print("EEG A :", eeg_a.shape)
    print("EEG B :", eeg_b.shape)
    print("Labels :", labels.shape)

    print("Labels du batch :", labels)