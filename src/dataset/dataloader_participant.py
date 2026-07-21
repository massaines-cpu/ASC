# Création des DataLoaders au niveau du participant

# Le découpage Train / Validation / Test reste effectué
# selon les dyades pour éviter les fuites de données.

# Ensuite, chaque fichier de dyade est transformé en deux
# exemples individuels par ParticipantDataset.

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from src.dataset.labels import prepare_classification_table
from src.dataset.participant_dataset import ParticipantDataset


chemin_projet = Path(__file__).resolve().parents[2]

chemin_metadata = (
    chemin_projet
    / "data"
    / "all_metadata.csv"
)

chemin_dataset = (
    chemin_projet
    / "data"
    / "data_toy"
)


def split_by_dyad(
    classification_table,
    train_dyads,
    validation_dyads,
    test_dyads,
):
    """
    Sépare la table selon les identifiants de dyades.

    Même si le modèle travaille sur un participant,
    on sépare d'abord les données par dyade.

    Cela évite par exemple que :

        participant A de J2 soit dans train

    et que :

        participant B de J2 soit dans test
    """

    train_table = classification_table[
        classification_table["dyad_id"].isin(train_dyads)
    ].copy()

    validation_table = classification_table[
        classification_table["dyad_id"].isin(
            validation_dyads
        )
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
    batch_size=10,
):
    """
    Crée les DataLoaders pour la classification au niveau
    du participant.
    """

    # Séparation des lignes de métadonnées par dyade
    train_table, validation_table, test_table = split_by_dyad(
        classification_table=classification_table,
        train_dyads=train_dyads,
        validation_dyads=validation_dyads,
        test_dyads=test_dyads,
    )

    # Création des Dataset au niveau participant
    train_dataset = ParticipantDataset(
        classification_table=train_table,
        dataset_root=dataset_root,
    )

    validation_dataset = ParticipantDataset(
        classification_table=validation_table,
        dataset_root=dataset_root,
    )

    test_dataset = ParticipantDataset(
        classification_table=test_table,
        dataset_root=dataset_root,
    )

    # DataLoader d'entraînement
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # DataLoader de validation
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # DataLoader de test
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, validation_loader, test_loader


if __name__ == "__main__":

    # Lecture du CSV
    all_metadata = pd.read_csv(chemin_metadata)

    # Préparation de la table YO / YF
    classification_table = prepare_classification_table(
        metadata=all_metadata,
        target_column="eyes_code",
        allowed_classes=["YO", "YF"],
        label_map={
            "YO": 0,
            "YF": 1,
        },
    )

    # Création des DataLoaders participant
    train_loader, validation_loader, test_loader = (
        create_participant_dataloaders(
            classification_table=classification_table,
            dataset_root=chemin_dataset,
            train_dyads=[
                "J2",
                "J4",
                "J5",
                "J7",
                "J8",
                "J1",
            ],
            validation_dyads=["J10"],
            test_dyads=["J15"],
            batch_size=5,
        )
    )

    # Récupération d'un premier batch
    eeg, labels = next(iter(train_loader))

    print(
        "Nombre de batches train :",
        len(train_loader),
    )

    print(
        "Nombre de batches validation :",
        len(validation_loader),
    )

    print(
        "Nombre de batches test :",
        len(test_loader),
    )

    print("EEG :", eeg.shape)
    print("Labels :", labels.shape)
    print("Labels du batch :", labels)