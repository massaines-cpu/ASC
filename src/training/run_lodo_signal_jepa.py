"""Lance l'expérience LODO SignalJEPA Contextual, 32 canaux ASC.

Contrairement à ``run_lodo_signal_jepa_prelocal.py`` (spécialisé, fenêtres de
2 s agrégées par participant), ce script réutilise le pipeline générique du
projet ASC (``ExperimentConfig``, ``model_fabrication``, ``epoch_runs``,
``ParticipantDataset``) : un exemple = un essai de 10 s, comme pour
``experience_*``/``data_final_*``. C'est ce pipeline générique qui a produit
les résultats partiels (fold J1 seulement) de ``data_signal_jepa_128hz_uv``,
avant que ``src/models/signal_jepa_model.py`` ne disparaisse du dépôt.

Ordre conseillé des trois expériences
--------------------------------------
1. ``MODEL_NAME = "signal_jepa_scratch"``
2. ``MODEL_NAME = "signal_jepa_pretrained"``, ``FREEZE_STRATEGY =
   "classifier_only"``
3. ``MODEL_NAME = "signal_jepa_pretrained"``, ``FREEZE_STRATEGY =
   "full_finetuning"``

Pour le premier test technique, conserver ``SELECTED_FOLDS = ("J1",)``. Le
run précédent s'était arrêté là avec une collecte accuracy=0.5 constante à
cause de la mauvaise classe SignalJEPA (voir la docstring de
``signal_jepa_model.py``) : si le nouveau modèle apprend sur ce seul fold,
relancer avec les huit dyades.

``batch_size=16`` et ``learning_rate=5e-3`` reprennent volontairement les
valeurs déjà validées pour PreLocal (tutoriel officiel), plutôt que le
``batch_size=2`` de la tentative précédente : un batch aussi petit est une
source connue d'instabilité d'entraînement, indépendamment du bug de classe
diagnostiqué.
"""

from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import nn

from src.dataset.dataloader_participant import create_participant_dataloaders
from src.dataset.labels import prepare_classification_table
from src.evaluation.plots import (
    save_confusion_matrix_plot,
    save_fold_plots,
    save_global_comparison,
)
from src.tracking.mlflow_track import MLflowTracker
from src.training.config import ExperimentConfig
from src.training.early_stopping import EarlyStopping
from src.training.epoch_runs import run_epoch
from src.training.model_fabrication import create_model


# ============================================================================
# 1. PARAMÈTRES DE CETTE EXPÉRIENCE
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_VERSION = "data_signal_jepa_128hz_uv"

# Valeurs possibles : "signal_jepa_scratch" ou "signal_jepa_pretrained".
MODEL_NAME = "signal_jepa_pretrained"

# Valeurs possibles : "classifier_only" ou "full_finetuning".
# classifier_only n'est autorisé qu'avec signal_jepa_pretrained.
FREEZE_STRATEGY = "full_finetuning"

# J1 permet d'abord de vérifier rapidement si le modèle quitte le hasard.
# Pour la LODO complète :
# SELECTED_FOLDS = ("J1", "J2", "J4", "J5", "J7", "J8", "J10", "J15")
SELECTED_FOLDS = ("J1",)

# cf. docstring du module : valeurs du tutoriel officiel SignalJEPA, déjà
# validées pour PreLocal, préférées au batch_size=2 de la tentative précédente.
BATCH_SIZE = 16
NUMBER_OF_EPOCHS = 50
LEARNING_RATE = 5e-3

# Le train accuracy restait bloqué au hasard même en scratch/full_finetuning
# (voir historique J1 dans les commits précédents) alors que 3,46M de
# paramètres devraient trivialement sur-apprendre 294 exemples si les
# gradients circulaient normalement. Hypothèse retenue : un transformer
# entraîné from scratch avec le LR complet dès le premier pas reste bloqué
# près de son initialisation (AdamW n'a pas encore d'estimation fiable de la
# variance des gradients) — cf. "Attention Is All You Need", qui utilise déjà
# un warmup pour cette raison. WARMUP_EPOCHS=5 monte le LR linéairement de
# LEARNING_RATE/10 à LEARNING_RATE avant de le laisser constant.
WARMUP_EPOCHS = 5

# Patience augmentée car le warmup rend les toutes premières epochs peu
# informatives : il ne faut pas que l'early stopping se déclenche pendant
# que le LR est encore en train de monter.
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4

RANDOM_SEED = 42
DEVICE_NAME = "mps"

PRETRAINED_CHECKPOINT = "braindecode/signal-jepa"

ENABLE_MLFLOW = False
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
MLFLOW_EXPERIMENT_NAME = "ASC_YO_YF_EXPERIMENTS"

DEVELOPMENT_DYADS = ("J1", "J2", "J4", "J5", "J7", "J8", "J10", "J15")

LABEL_NAMES = {0: "YO", 1: "YF"}


def create_config() -> ExperimentConfig:
    """Transforme les constantes ci-dessus en configuration validée.

    number_of_channels=32, number_of_timepoints=1280 et
    sampling_frequency=128.0 sont figés par la validation stricte
    d'ExperimentConfig pour tout modèle SignalJEPA (cf. config.py) : ce ne
    sont pas des valeurs libres à ajuster ici.
    """

    return ExperimentConfig(
        project_root=PROJECT_ROOT,
        dataset_version=DATASET_VERSION,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        number_of_epochs=NUMBER_OF_EPOCHS,
        learning_rate=LEARNING_RATE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
        random_seed=RANDOM_SEED,
        standardize=False,
        number_of_channels=32,
        number_of_timepoints=1280,
        sampling_frequency=128.0,
        pretrained_checkpoint=PRETRAINED_CHECKPOINT,
        freeze_strategy=FREEZE_STRATEGY,
        device_name=DEVICE_NAME,
    )


def set_seed(seed: int) -> None:
    """Fixe les générateurs utilisés par Python, NumPy et PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def warmup_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch_number: int,
    target_learning_rate: float,
    warmup_epochs: int,
) -> float:
    """Monte le LR linéairement de target/10 à target sur warmup_epochs.

    Réglage volontairement simple (par epoch, pas par pas de gradient) :
    avec ~19 pas par epoch ici, la granularité par epoch reste fine, et ça
    évite de modifier epoch_runs.py (partagé par toutes les autres
    expériences ASC déjà validées).
    """

    if epoch_number > warmup_epochs:
        applied_learning_rate = target_learning_rate
    else:
        start_learning_rate = target_learning_rate / 10.0
        progress = epoch_number / warmup_epochs
        applied_learning_rate = (
            start_learning_rate
            + progress * (target_learning_rate - start_learning_rate)
        )

    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = applied_learning_rate

    return applied_learning_rate


def select_device(device_name: str) -> torch.device:
    """Sélectionne CPU, CUDA ou le GPU Apple MPS."""

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA a été demandé mais n'est pas disponible.")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS a été demandé mais n'est pas disponible.")

    return torch.device(device_name)


def validate_folds(selected_folds: tuple[str, ...]) -> None:
    """Détecte une faute de frappe dans les identifiants de dyades."""

    unknown_folds = set(selected_folds) - set(DEVELOPMENT_DYADS)
    if unknown_folds:
        raise ValueError(f"Dyades inconnues : {sorted(unknown_folds)}.")

    if len(set(selected_folds)) != len(selected_folds):
        raise ValueError("SELECTED_FOLDS contient une dyade en double.")


def create_classification_table(project_root: Path) -> pd.DataFrame:
    """Prépare YO=0 et YF=1 à partir des métadonnées existantes."""

    metadata_path = project_root / "data" / "all_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Métadonnées introuvables : {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    return prepare_classification_table(
        metadata=metadata,
        target_column="eyes_code",
        allowed_classes=["YO", "YF"],
        label_map={"YO": 0, "YF": 1},
    )


def save_checkpoint_on_cpu(model: nn.Module, path: Path) -> None:
    """Sauvegarde un state_dict portable après entraînement MPS/CUDA."""

    state_dict_cpu = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    torch.save(state_dict_cpu, path)


def evaluate_and_save_fold(
    validation_dyad: str,
    model: nn.Module,
    loader,
    results_dir: Path,
    device: torch.device,
) -> dict[str, float]:
    """Évalue le meilleur checkpoint et enregistre prédictions et rapport.

    Même format de sortie (predictions.csv, confusion_matrix.csv,
    evaluation_report.txt) que les autres expériences génériques ASC, pour
    que result_io.py et les scripts de reporting les lisent sans distinction.
    """

    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    probabilities_yf: list[float] = []

    with torch.no_grad():
        for eeg, labels in loader:
            eeg = eeg.to(device)
            logits = model(eeg)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(dim=1)

            probability_yf = torch.sigmoid(logits).cpu().numpy()
            predictions = (probability_yf >= 0.5).astype(int)

            true_labels.extend(labels.numpy().tolist())
            predicted_labels.extend(predictions.tolist())
            probabilities_yf.extend(probability_yf.tolist())

    fold_dir = results_dir / f"fold_{validation_dyad}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    predictions_table = pd.DataFrame({
        "true_label": true_labels,
        "true_class": [LABEL_NAMES[label] for label in true_labels],
        "predicted_label": predicted_labels,
        "predicted_class": [LABEL_NAMES[label] for label in predicted_labels],
        "probability_yo": [1.0 - probability for probability in probabilities_yf],
        "probability_yf": probabilities_yf,
    })
    predictions_table["correct"] = (
        predictions_table["true_label"] == predictions_table["predicted_label"]
    )
    predictions_table.to_csv(fold_dir / "predictions.csv", index=False)

    matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    matrix_table = pd.DataFrame(
        matrix,
        index=["true_YO", "true_YF"],
        columns=["predicted_YO", "predicted_YF"],
    )
    matrix_table.to_csv(fold_dir / "confusion_matrix.csv")
    save_confusion_matrix_plot(
        validation_dyad=validation_dyad,
        matrix=matrix,
        results_dir=results_dir,
    )

    accuracy = float(np.mean(np.array(true_labels) == np.array(predicted_labels)))
    report_text = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["YO", "YF"],
        digits=4,
    )
    mean_confidence = float(
        np.mean(np.maximum(probabilities_yf, 1.0 - np.array(probabilities_yf)))
    )

    report_path = fold_dir / "evaluation_report.txt"
    report_path.write_text(
        f"Dyade de validation : {validation_dyad}\n"
        f"Accuracy : {accuracy:.4f}\n"
        f"Confiance moyenne : {mean_confidence:.4f}\n\n"
        f"Matrice de confusion :\n{matrix}\n\n"
        f"Classification report :\n{report_text}\n",
        encoding="utf-8",
    )

    return {"accuracy": accuracy, "mean_confidence": mean_confidence}


def train_one_fold(
    validation_dyad: str,
    classification_table: pd.DataFrame,
    config: ExperimentConfig,
    results_dir: Path,
    models_dir: Path,
    device: torch.device,
    tracker: MLflowTracker,
) -> tuple[dict[str, list[float]], dict[str, float | int]]:
    """Entraîne un modèle neuf puis évalue son meilleur checkpoint."""

    train_dyads = [dyad for dyad in DEVELOPMENT_DYADS if dyad != validation_dyad]

    train_loader, validation_loader, _test_loader = create_participant_dataloaders(
        classification_table=classification_table,
        dataset_root=config.dataset_root,
        train_dyads=train_dyads,
        validation_dyads=[validation_dyad],
        test_dyads=[],
        batch_size=config.batch_size,
        standardize=config.standardize,
        expected_number_of_channels=config.number_of_channels,
        expected_number_of_timepoints=config.number_of_timepoints,
    )

    # Chaque fold repart de la même seed mais d'un modèle entièrement neuf.
    set_seed(config.random_seed)
    model = create_model(
        model_name=config.model_name,
        hidden_layer_size=config.hidden_layer_size,
        dropout_rate=config.dropout_rate,
        number_of_channels=config.number_of_channels,
        number_of_timepoints=config.number_of_timepoints,
        sampling_frequency=config.sampling_frequency,
        pretrained_checkpoint=config.pretrained_checkpoint,
        freeze_strategy=config.freeze_strategy,
    ).to(device)

    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameter_count == 0:
        raise RuntimeError("Le modèle ne possède aucun paramètre entraînable.")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )
    criterion = nn.BCEWithLogitsLoss()

    print("=" * 72)
    print(f"FOLD - validation : {validation_dyad}")
    print("Train             :", train_dyads)
    print("Modèle            :", config.model_name)
    print("Stratégie         :", config.freeze_strategy)
    print("Entrée            : [batch, 32, 1280]")
    print("Paramètres entraînables :", f"{trainable_parameter_count:,}")
    print("Appareil          :", device)
    print("=" * 72)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )

    best_epoch = 0
    best_model_path = models_dir / f"best_model_fold_{validation_dyad}.pt"

    for epoch_index in range(config.number_of_epochs):
        epoch_number = epoch_index + 1

        applied_learning_rate = warmup_learning_rate(
            optimizer=optimizer,
            epoch_number=epoch_number,
            target_learning_rate=config.learning_rate,
            warmup_epochs=WARMUP_EPOCHS,
        )

        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            freeze_strategy=config.freeze_strategy,
        )
        validation_loss, validation_accuracy = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            freeze_strategy=config.freeze_strategy,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        tracker.log_epoch_metrics(
            epoch=epoch_number,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
        )

        improved = early_stopping.update(validation_loss)
        if improved:
            best_epoch = epoch_number
            save_checkpoint_on_cpu(model, best_model_path)

        print(
            f"Fold {validation_dyad} | Epoch {epoch_number:03d}/{config.number_of_epochs} | "
            f"LR {applied_learning_rate:.2e} | "
            f"Train loss {train_loss:.4f}, acc {train_accuracy:.4f} | "
            f"Val loss {validation_loss:.4f}, acc {validation_accuracy:.4f}"
        )
        print(
            "Epochs sans amélioration : "
            f"{early_stopping.epochs_without_improvement}/{config.early_stopping_patience}"
        )

        if early_stopping.should_stop():
            print(f"Early stopping — fold {validation_dyad}, meilleure epoch : {best_epoch}")
            break

    if best_epoch == 0:
        raise RuntimeError("Aucun checkpoint n'a été sauvegardé.")

    save_fold_plots(
        validation_dyad=validation_dyad,
        history=history,
        results_dir=results_dir,
    )

    best_state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state_dict)

    evaluation_metrics = evaluate_and_save_fold(
        validation_dyad=validation_dyad,
        model=model,
        loader=validation_loader,
        results_dir=results_dir,
        device=device,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_validation_loss": early_stopping.best_loss,
        "best_validation_accuracy": history["validation_accuracy"][best_epoch - 1],
        "final_train_loss": history["train_loss"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "evaluation_accuracy": evaluation_metrics["accuracy"],
        "evaluation_mean_confidence": evaluation_metrics["mean_confidence"],
    }

    tracker.log_best_metrics(
        best_epoch=best_epoch,
        best_validation_loss=summary["best_validation_loss"],
        best_validation_accuracy=summary["best_validation_accuracy"],
    )
    tracker.log_evaluation_metrics(evaluation_metrics)
    tracker.log_artifact(best_model_path, artifact_path="checkpoint")
    tracker.log_artifacts(fold_dir_of(results_dir, validation_dyad), artifact_path="fold_results")

    return history, summary


def fold_dir_of(results_dir: Path, validation_dyad: str) -> Path:
    return results_dir / f"fold_{validation_dyad}"


def save_config(config: ExperimentConfig, selected_folds: tuple[str, ...], results_dir: Path) -> None:
    """Enregistre les paramètres afin que l'expérience soit reproductible."""

    config_dictionary = asdict(config)
    config_dictionary["project_root"] = str(config.project_root)
    config_dictionary["selected_folds"] = list(selected_folds)

    (results_dir / "experiment_config.json").write_text(
        json.dumps(config_dictionary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    """Exécute les folds sélectionnés et produit le résumé global."""

    config = create_config()
    validate_folds(SELECTED_FOLDS)
    device = select_device(config.device_name)

    if not config.dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {config.dataset_root}. "
            "Exécuter d'abord python -m src.dataset.prepare_signal_jepa"
        )

    results_dir = PROJECT_ROOT / "results" / config.experiment_name
    models_dir = PROJECT_ROOT / "models" / config.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, SELECTED_FOLDS, results_dir)

    classification_table = create_classification_table(PROJECT_ROOT)
    histories = {}
    summaries = []

    tracker = MLflowTracker(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        enabled=ENABLE_MLFLOW,
    )

    print("Expérience :", config.experiment_name)
    print("Dataset    :", config.dataset_root)
    print("Folds      :", list(SELECTED_FOLDS))
    print("Z-score    :", config.standardize)
    print("Unité      : microvolts")
    print("MLflow     :", "activé" if ENABLE_MLFLOW else "désactivé")

    for validation_dyad in SELECTED_FOLDS:
        train_dyads = [dyad for dyad in DEVELOPMENT_DYADS if dyad != validation_dyad]

        fold_parameters = {
            "dataset_version": config.dataset_version,
            "model_name": config.model_name,
            "freeze_strategy": config.freeze_strategy,
            "pretrained_checkpoint": (
                config.pretrained_checkpoint if config.uses_pretrained_weights else "none"
            ),
            "validation_dyad": validation_dyad,
            "training_dyads": ",".join(train_dyads),
            "protocol": "Leave-One-Dyad-Out",
            "number_of_channels": config.number_of_channels,
            "number_of_timepoints": config.number_of_timepoints,
            "sampling_frequency_hz": config.sampling_frequency,
            "data_unit": "microvolts",
            "standardized": config.standardize,
            "batch_size": config.batch_size,
            "maximum_epochs": config.number_of_epochs,
            "learning_rate": config.learning_rate,
            "early_stopping_patience": config.early_stopping_patience,
            "random_seed": config.random_seed,
            "device": str(device),
            "loss_function": "BCEWithLogitsLoss",
            "optimizer": "AdamW",
        }

        with tracker.fold_run(
            run_name=f"{config.experiment_name}__fold_{validation_dyad}",
            parameters=fold_parameters,
        ):
            history, summary = train_one_fold(
                validation_dyad=validation_dyad,
                classification_table=classification_table,
                config=config,
                results_dir=results_dir,
                models_dir=models_dir,
                device=device,
                tracker=tracker,
            )
        histories[validation_dyad] = history
        summaries.append({"validation_dyad": validation_dyad, **summary})

        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(results_dir / "lodo_cv_summary.csv", index=False)
    save_global_comparison(histories, results_dir)

    mean_accuracy = summary_table["best_validation_accuracy"].mean()
    mean_loss = summary_table["best_validation_loss"].mean()

    print("\n" + "=" * 72)
    print("RÉSUMÉ GLOBAL — SIGNALJEPA CONTEXTUAL, 32 CANAUX")
    print("=" * 72)
    print(summary_table.to_string(index=False))
    print(f"\nAccuracy de validation moyenne : {mean_accuracy:.4f}")
    print(f"Loss de validation moyenne     : {mean_loss:.4f}")
    print("Résultats                      :", results_dir)


if __name__ == "__main__":
    main()
