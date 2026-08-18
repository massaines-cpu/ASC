"""Création des DataLoaders au niveau participant.

Le découpage est toujours effectué au niveau des dyades avant de transformer
chaque fichier en deux exemples. Cela empêche les deux membres d'une même
dyade d'apparaître dans des ensembles différents.
"""

from torch.utils.data import DataLoader

from src.dataset.participant_dataset import ParticipantDataset


def split_by_dyad(
    classification_table,
    train_dyads,
    validation_dyads,
    test_dyads,
):
    """Sépare les métadonnées à partir des identifiants de dyades."""

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


def create_participant_dataloaders(
    classification_table,
    dataset_root,
    train_dyads,
    validation_dyads,
    test_dyads,
    batch_size: int = 10,
    standardize: bool = True,
    expected_number_of_channels: int = 32,
    expected_number_of_timepoints: int = 5120,
):
    """Crée les trois DataLoaders avec un prétraitement explicite."""

    train_table, validation_table, test_table = split_by_dyad(
        classification_table=classification_table,
        train_dyads=train_dyads,
        validation_dyads=validation_dyads,
        test_dyads=test_dyads,
    )

    common_dataset_arguments = {
        "dataset_root": dataset_root,
        "standardize": standardize,
        "expected_number_of_channels": expected_number_of_channels,
        "expected_number_of_timepoints": expected_number_of_timepoints,
    }

    train_dataset = ParticipantDataset(
        classification_table=train_table,
        **common_dataset_arguments,
    )
    validation_dataset = ParticipantDataset(
        classification_table=validation_table,
        **common_dataset_arguments,
    )
    test_dataset = ParticipantDataset(
        classification_table=test_table,
        **common_dataset_arguments,
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

