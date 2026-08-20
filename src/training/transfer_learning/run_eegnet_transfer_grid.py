"""LODO EEGNet pré-entraîné avec quatre niveaux de fine-tuning.

Ce script est autonome par rapport à l'ancien ``transfer_learning.py``. Il
réutilise uniquement le dataset, les labels et l'architecture EEGNet du projet.
Tous les choix sont lus dans ``settings.py`` : aucun argument terminal n'est
nécessaire.

Règle scientifique appliquée
----------------------------
Une couche ne peut être gelée que si tous ses paramètres proviennent réellement
du checkpoint. Geler une couche restée aléatoire ne constitue pas un transfert
d'apprentissage valide et déclenche donc une erreur explicite.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from eegnet_freezing import (
    apply_freeze_strategy,
    keep_frozen_modules_in_eval,
)
from inspect_checkpoint_compatibility import (
    load_compatible_weights,
)
from src.config.settings import (
    EEGNET_CHECKPOINT_NAME,
    EEGNET_CHECKPOINT_PATH,
    EEGNET_TRANSFER_STRATEGIES,
    EXPECTED_DYADS,
    PROJECT_ROOT,
    TRANSFER_BATCH_SIZE,
    TRANSFER_DATASET_VERSION,
    TRANSFER_DEVICE,
    TRANSFER_LEARNING_RATE,
    TRANSFER_MAXIMUM_EPOCHS,
    TRANSFER_PATIENCE,
    TRANSFER_RANDOM_SEED,
    TRANSFER_SELECTED_DYADS,
    TRANSFER_STANDARDIZE,
)


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.dataloader_participant import (  # noqa: E402
    create_participant_dataloaders,
)
from src.dataset.labels import prepare_classification_table  # noqa: E402
from src.models.eegNET_model import EEGNet  # noqa: E402


@dataclass
class EarlyStoppingState:
    """État minimal d'un early stopping surveillant la validation loss."""

    patience: int
    best_loss: float = float("inf")
    epochs_without_improvement: int = 0

    def update(self, validation_loss: float) -> bool:
        """Retourne True si la loss atteint un nouveau minimum."""

        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.epochs_without_improvement = 0
            return True

        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience


def set_seed(seed: int) -> None:
    """Fixe les générateurs avant chaque fold."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_device(device_name: str) -> torch.device:
    """Sélectionne l'appareil demandé sans repli silencieux."""

    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS demandé mais indisponible.")
        return torch.device("mps")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA demandé mais indisponible.")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    raise ValueError("TRANSFER_DEVICE doit valoir cpu, mps ou cuda.")


def sanitize_name(value: str) -> str:
    """Transforme un nom externe en nom de dossier portable."""

    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()


def create_classification_table() -> pd.DataFrame:
    """Charge les métadonnées et conserve les deux conditions YO/YF."""

    metadata_path = PROJECT_ROOT / "data" / "all_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Métadonnées introuvables : {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    return prepare_classification_table(
        metadata=metadata,
        target_column="eyes_code",
        allowed_classes=["YO", "YF"],
        label_map={"YO": 0, "YF": 1},
    )


def save_model_state_dict(model: nn.Module, path: Path) -> None:
    """Sauvegarde un checkpoint portable sur CPU."""

    state_dict = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    torch.save(state_dict, path)


def validate_frozen_parameters_have_pretrained_values(
    model: nn.Module,
    compatibility_report: pd.DataFrame,
    strategy: str,
) -> None:
    """Interdit de geler un paramètre qui n'a pas été transféré."""

    compatible_names = set(
        compatibility_report.loc[
            compatibility_report["status"] == "compatible",
            "mapped_target_name",
        ].astype(str)
    )
    frozen_parameter_names = {
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    frozen_but_random = frozen_parameter_names - compatible_names

    if frozen_but_random:
        raise RuntimeError(
            f"Stratégie '{strategy}' invalide pour ce checkpoint : ces poids "
            "seraient gelés alors qu'ils n'ont pas été transférés : "
            + ", ".join(sorted(frozen_but_random))
        )


def prepare_binary_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalise la sortie du modèle vers [batch_size]."""

    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits.squeeze(dim=1)
    raise ValueError(f"Sortie binaire inattendue : {tuple(logits.shape)}")


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    strategy: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Exécute une epoch et calcule loss/accuracy sur tous les exemples."""

    training = optimizer is not None
    if training:
        model.train()
        keep_frozen_modules_in_eval(model, strategy)
    else:
        model.eval()

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

            logits = prepare_binary_logits(model(eeg))
            targets = labels.to(dtype=logits.dtype)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            probabilities_yf = torch.sigmoid(logits)
            predictions = (probabilities_yf >= 0.5).long()
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("DataLoader vide.")

    return total_loss / total_examples, total_correct / total_examples


def evaluate_and_save(
    model: nn.Module,
    loader,
    device: torch.device,
    fold_directory: Path,
) -> None:
    """Sauvegarde prédictions, matrice et rapport du meilleur checkpoint."""

    model.eval()
    labels_all = []
    predictions_all = []
    probabilities_all = []

    with torch.no_grad():
        for eeg, labels in loader:
            logits = prepare_binary_logits(model(eeg.to(device)))
            probabilities_yf = torch.sigmoid(logits)
            predictions = (probabilities_yf >= 0.5).long()

            labels_all.extend(labels.numpy().tolist())
            predictions_all.extend(predictions.cpu().numpy().tolist())
            probabilities_all.extend(
                probabilities_yf.cpu().numpy().tolist()
            )

    prediction_table = pd.DataFrame({
        "true_label": labels_all,
        "predicted_label": predictions_all,
        "probability_yf": probabilities_all,
    })
    prediction_table["probability_yo"] = (
        1.0 - prediction_table["probability_yf"]
    )
    prediction_table.to_csv(
        fold_directory / "participant_predictions.csv",
        index=False,
    )

    matrix = confusion_matrix(labels_all, predictions_all, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["true_YO", "true_YF"],
        columns=["predicted_YO", "predicted_YF"],
    ).to_csv(fold_directory / "confusion_matrix.csv")

    report = classification_report(
        labels_all,
        predictions_all,
        labels=[0, 1],
        target_names=["YO", "YF"],
        digits=4,
        zero_division=0,
    )
    (fold_directory / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )


def train_one_fold(
    strategy: str,
    validation_dyad: str,
    classification_table: pd.DataFrame,
    checkpoint_path: Path,
    results_directory: Path,
    models_directory: Path,
    device: torch.device,
) -> dict[str, object]:
    """Entraîne et évalue un fold avec une stratégie de gel."""

    train_dyads = [
        dyad for dyad in EXPECTED_DYADS if dyad != validation_dyad
    ]
    dataset_root = PROJECT_ROOT / "data" / TRANSFER_DATASET_VERSION

    train_loader, validation_loader, _ = create_participant_dataloaders(
        classification_table=classification_table,
        dataset_root=dataset_root,
        train_dyads=train_dyads,
        validation_dyads=[validation_dyad],
        test_dyads=[],
        batch_size=TRANSFER_BATCH_SIZE,
        standardize=TRANSFER_STANDARDIZE,
        expected_number_of_channels=32,
        expected_number_of_timepoints=5120,
    )

    set_seed(TRANSFER_RANDOM_SEED)
    model = EEGNet(n_channels=32, n_samples=5120)
    compatibility_report = load_compatible_weights(model, checkpoint_path)
    parameter_counts = apply_freeze_strategy(model, strategy)
    validate_frozen_parameters_have_pretrained_values(
        model=model,
        compatibility_report=compatibility_report,
        strategy=strategy,
    )
    model = model.to(device)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=TRANSFER_LEARNING_RATE,
    )
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStoppingState(patience=TRANSFER_PATIENCE)

    fold_directory = results_directory / f"fold_{validation_dyad}"
    fold_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_output_path = (
        models_directory / f"best_model_fold_{validation_dyad}.pt"
    )

    history_rows = []
    best_epoch = 0
    best_validation_accuracy = 0.0

    for epoch in range(1, TRANSFER_MAXIMUM_EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            strategy=strategy,
        )
        validation_loss, validation_accuracy = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            strategy=strategy,
        )

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })

        improved = early_stopping.update(validation_loss)
        if improved:
            best_epoch = epoch
            best_validation_accuracy = validation_accuracy
            save_model_state_dict(model, checkpoint_output_path)

        print(
            f"{strategy} | Fold {validation_dyad} | Epoch {epoch:03d} | "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.4f} | "
            f"val loss={validation_loss:.4f}, acc={validation_accuracy:.4f}"
        )

        if early_stopping.should_stop:
            print(
                f"Early stopping — best epoch={best_epoch}, "
                f"patience={TRANSFER_PATIENCE}"
            )
            break

    history = pd.DataFrame(history_rows)
    history.to_csv(fold_directory / "history.csv", index=False)

    best_state_dict = torch.load(
        checkpoint_output_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_state_dict)
    evaluate_and_save(
        model=model,
        loader=validation_loader,
        device=device,
        fold_directory=fold_directory,
    )

    return {
        "validation_dyad": validation_dyad,
        "best_epoch": best_epoch,
        "best_validation_loss": early_stopping.best_loss,
        "best_validation_accuracy": best_validation_accuracy,
        "final_train_loss": history.iloc[-1]["train_loss"],
        "final_train_accuracy": history.iloc[-1]["train_accuracy"],
        "final_validation_loss": history.iloc[-1]["validation_loss"],
        "final_validation_accuracy": history.iloc[-1]["validation_accuracy"],
        **parameter_counts,
    }


def run_strategy(
    strategy: str,
    classification_table: pd.DataFrame,
    checkpoint_path: Path,
    device: torch.device,
) -> Path:
    """Exécute tous les folds sélectionnés pour une stratégie."""

    checkpoint_slug = sanitize_name(EEGNET_CHECKPOINT_NAME)
    experiment_name = (
        f"{TRANSFER_DATASET_VERSION}_eegnet_pretrained_{checkpoint_slug}_"
        f"{strategy}_standardized"
    )
    results_directory = PROJECT_ROOT / "results" / experiment_name
    models_directory = PROJECT_ROOT / "models" / experiment_name
    results_directory.mkdir(parents=True, exist_ok=True)
    models_directory.mkdir(parents=True, exist_ok=True)

    configuration = {
        "checkpoint_name": EEGNET_CHECKPOINT_NAME,
        "checkpoint_path": str(checkpoint_path),
        "strategy": strategy,
        "dataset_version": TRANSFER_DATASET_VERSION,
        "standardize": TRANSFER_STANDARDIZE,
        "selected_folds": TRANSFER_SELECTED_DYADS,
        "batch_size": TRANSFER_BATCH_SIZE,
        "maximum_epochs": TRANSFER_MAXIMUM_EPOCHS,
        "learning_rate": TRANSFER_LEARNING_RATE,
        "early_stopping_patience": TRANSFER_PATIENCE,
        "early_stopping_metric": "validation_loss",
        "checkpoint_selection": "minimum_validation_loss",
        "random_seed": TRANSFER_RANDOM_SEED,
        "device": str(device),
        "loss": "BCEWithLogitsLoss",
    }
    (results_directory / "experiment_config.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fold_rows = []
    for validation_dyad in TRANSFER_SELECTED_DYADS:
        fold_rows.append(
            train_one_fold(
                strategy=strategy,
                validation_dyad=validation_dyad,
                classification_table=classification_table,
                checkpoint_path=checkpoint_path,
                results_directory=results_directory,
                models_directory=models_directory,
                device=device,
            )
        )

    summary = pd.DataFrame(fold_rows)
    summary.to_csv(results_directory / "lodo_cv_summary.csv", index=False)
    print(
        f"\n{strategy} — accuracy moyenne : "
        f"{summary['best_validation_accuracy'].mean():.4f} ± "
        f"{summary['best_validation_accuracy'].std(ddof=1):.4f}"
    )
    return results_directory


def main() -> None:
    """Audite le checkpoint puis lance les quatre stratégies."""

    if EEGNET_CHECKPOINT_PATH is None:
        raise ValueError(
            "Renseigne EEGNET_CHECKPOINT_PATH dans settings.py. Exécute "
            "d'abord inspect_checkpoint_compatibility.py."
        )
    if set(TRANSFER_SELECTED_DYADS) - set(EXPECTED_DYADS):
        raise ValueError("TRANSFER_SELECTED_DYADS contient une dyade inconnue.")

    device = select_device(TRANSFER_DEVICE)
    classification_table = create_classification_table()

    print(f"Checkpoint : {EEGNET_CHECKPOINT_PATH}")
    print(f"Device     : {device}")
    print(f"Folds      : {TRANSFER_SELECTED_DYADS}")

    for strategy in EEGNET_TRANSFER_STRATEGIES:
        result_directory = run_strategy(
            strategy=strategy,
            classification_table=classification_table,
            checkpoint_path=EEGNET_CHECKPOINT_PATH,
            device=device,
        )
        print(f"Résultats : {result_directory}")


if __name__ == "__main__":
    main()

