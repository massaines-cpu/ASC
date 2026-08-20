"""Pré-entraîne le backbone EEGNet sur un dataset public déjà préparé.

Contrat du manifeste CSV
------------------------
Chaque ligne décrit un exemple EEG :

- ``file_path`` : chemin vers un .npy de forme [32, temps] ;
- ``label`` : entier entre 0 et number_of_source_classes - 1 ;
- ``subject_id`` : identifiant du sujet ;
- ``split`` : ``train`` ou ``validation``.

Le découpage par sujet doit être réalisé pendant la préparation du manifeste.
Le script refuse tout sujet présent à la fois dans train et validation.

Le checkpoint final exclut le classifieur de la tâche publique. Il pourra donc
initialiser le backbone EEGNet ASC, puis recevoir une nouvelle tête YO/YF.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config.settings import (
    PROJECT_ROOT,
    PUBLIC_DATASET_MANIFEST,
    PUBLIC_PRETRAINING_OUTPUT,
)


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.eegNET_model import EEGNet  # noqa: E402


# Ces paramètres décrivent une seule expérience. Ne changer qu'une variable
# entre deux runs et conserver une copie du fichier configuration.json produit.
NUMBER_OF_SOURCE_CLASSES = 2
EXPECTED_CHANNELS = 32
EXPECTED_TIMEPOINTS = 5120
STANDARDIZE_PER_CHANNEL = True
BATCH_SIZE = 16
MAXIMUM_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
DEVICE_NAME = "mps"


class PublicEEGDataset(Dataset):
    """Charge les exemples décrits dans le manifeste public."""

    def __init__(self, table: pd.DataFrame):
        self.table = table.reset_index(drop=True).copy()

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
        row = self.table.iloc[index]
        file_path = Path(row["file_path"])
        eeg = np.asarray(np.load(file_path), dtype=np.float32)

        expected_shape = (EXPECTED_CHANNELS, EXPECTED_TIMEPOINTS)
        if eeg.shape != expected_shape:
            raise ValueError(
                f"{file_path} : forme {eeg.shape}, attendue {expected_shape}."
            )

        if STANDARDIZE_PER_CHANNEL:
            channel_means = eeg.mean(axis=1, keepdims=True)
            channel_stds = eeg.std(axis=1, keepdims=True)
            channel_stds[channel_stds < 1e-8] = 1.0
            eeg = (eeg - channel_means) / channel_stds

        label = int(row["label"])
        return torch.from_numpy(eeg), torch.tensor(label, dtype=torch.long)


@dataclass
class StopState:
    """Suit le minimum de validation loss."""

    patience: int
    best_loss: float = float("inf")
    epochs_without_improvement: int = 0

    def update(self, loss: float) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_device() -> torch.device:
    if DEVICE_NAME == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS demandé mais indisponible.")
        return torch.device("mps")
    if DEVICE_NAME == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA demandé mais indisponible.")
        return torch.device("cuda")
    return torch.device("cpu")


def load_and_validate_manifest() -> pd.DataFrame:
    """Vérifie les colonnes, labels et l'absence de fuite sujet."""

    if PUBLIC_DATASET_MANIFEST is None:
        raise ValueError(
            "Renseigne PUBLIC_DATASET_MANIFEST dans settings.py."
        )
    if not PUBLIC_DATASET_MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifeste introuvable : {PUBLIC_DATASET_MANIFEST}"
        )

    table = pd.read_csv(PUBLIC_DATASET_MANIFEST)
    required_columns = {"file_path", "label", "subject_id", "split"}
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(
            "Colonnes manquantes : " + ", ".join(sorted(missing_columns))
        )

    unknown_splits = set(table["split"]) - {"train", "validation"}
    if unknown_splits:
        raise ValueError(f"Splits inconnus : {sorted(unknown_splits)}")

    train_subjects = set(
        table.loc[table["split"] == "train", "subject_id"].astype(str)
    )
    validation_subjects = set(
        table.loc[
            table["split"] == "validation",
            "subject_id",
        ].astype(str)
    )
    overlap = train_subjects & validation_subjects
    if overlap:
        raise ValueError(
            "Fuite de sujets entre train et validation : "
            + ", ".join(sorted(overlap))
        )

    labels = set(table["label"].astype(int))
    expected_labels = set(range(NUMBER_OF_SOURCE_CLASSES))
    if labels != expected_labels:
        raise ValueError(
            f"Labels trouvés={sorted(labels)}, attendus={sorted(expected_labels)}."
        )

    return table


def create_model() -> EEGNet:
    """Crée EEGNet puis remplace uniquement la tête de la tâche source."""

    model = EEGNet(
        n_channels=EXPECTED_CHANNELS,
        n_samples=EXPECTED_TIMEPOINTS,
    )
    source_feature_count = model.classifier.in_features
    model.classifier = nn.Linear(
        in_features=source_feature_count,
        out_features=NUMBER_OF_SOURCE_CLASSES,
    )
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Entraîne ou évalue une epoch multiclasses source."""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for eeg, labels in loader:
            eeg = eeg.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(eeg)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def save_backbone_checkpoint(model: EEGNet, path: Path) -> None:
    """Exclut la tête source afin qu'elle ne contamine pas la tâche YO/YF."""

    backbone_state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if not name.startswith("classifier.")
    }
    torch.save(
        {
            "state_dict": backbone_state_dict,
            "metadata": {
                "architecture": "ASC simplified EEGNet",
                "number_of_channels": EXPECTED_CHANNELS,
                "number_of_timepoints": EXPECTED_TIMEPOINTS,
                "source_classes": NUMBER_OF_SOURCE_CLASSES,
                "classifier_included": False,
                "standardize_per_channel": STANDARDIZE_PER_CHANNEL,
            },
        },
        path,
    )


def main() -> None:
    """Pré-entraîne et sauvegarde le meilleur backbone public."""

    set_seed(RANDOM_SEED)
    device = select_device()
    table = load_and_validate_manifest()

    train_dataset = PublicEEGDataset(table[table["split"] == "train"])
    validation_dataset = PublicEEGDataset(
        table[table["split"] == "validation"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    stopping = StopState(patience=PATIENCE)

    PUBLIC_PRETRAINING_OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        PUBLIC_PRETRAINING_OUTPUT / "best_public_eegnet_backbone.pt"
    )
    history_rows = []
    best_epoch = 0

    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )
        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })

        if stopping.update(validation_loss):
            best_epoch = epoch
            save_backbone_checkpoint(model, checkpoint_path)

        print(
            f"Epoch {epoch:03d} | train {train_loss:.4f}, "
            f"{train_accuracy:.4f} | validation {validation_loss:.4f}, "
            f"{validation_accuracy:.4f}"
        )

        if stopping.should_stop:
            print(f"Early stopping — meilleure epoch : {best_epoch}")
            break

    pd.DataFrame(history_rows).to_csv(
        PUBLIC_PRETRAINING_OUTPUT / "history.csv",
        index=False,
    )
    configuration = {
        "manifest": str(PUBLIC_DATASET_MANIFEST),
        "number_of_source_classes": NUMBER_OF_SOURCE_CLASSES,
        "expected_channels": EXPECTED_CHANNELS,
        "expected_timepoints": EXPECTED_TIMEPOINTS,
        "standardize_per_channel": STANDARDIZE_PER_CHANNEL,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAXIMUM_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "random_seed": RANDOM_SEED,
        "best_epoch": best_epoch,
        "best_validation_loss": stopping.best_loss,
        "checkpoint": str(checkpoint_path),
    }
    (PUBLIC_PRETRAINING_OUTPUT / "configuration.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Backbone sauvegardé : {checkpoint_path}")
    print(
        "Avant le transfert ASC, exécuter "
        "inspect_checkpoint_compatibility.py sur ce fichier."
    )


if __name__ == "__main__":
    main()

