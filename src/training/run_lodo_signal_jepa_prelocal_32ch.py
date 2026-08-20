"""Lance l'expérience LODO SignalJEPA PreLocal, montage ASC complet 32 canaux.

Variante de ``run_lodo_signal_jepa_prelocal.py`` sans réduction de montage —
voir la docstring de ``prepare_signal_jepa_prelocal_32ch.py`` pour pourquoi
cette variante existe (Contextual à 32 canaux a un déséquilibre structurel
position/contenu documenté dans ``src/models/signal_jepa_model.py`` ;
PreLocal, lui, n'a jamais dépendu du nombre de canaux).

Ordre conseillé des trois expériences
--------------------------------------
1. ``MODEL_VARIANT = "scratch"`` et ``FREEZE_STRATEGY = "full_finetuning"``
2. ``MODEL_VARIANT = "pretrained"`` et ``FREEZE_STRATEGY = "classifier_only"``
3. ``MODEL_VARIANT = "pretrained"`` et ``FREEZE_STRATEGY = "full_finetuning"``

Pour le premier test technique, conserver seulement ``SELECTED_FOLDS =
("J1",)``. Si le modèle apprend, remplacer ensuite cette valeur par toutes
les dyades afin de lancer la LODO complète.
"""

from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.dataset.labels import prepare_classification_table
from src.dataset.signal_jepa_window_dataset import (
    create_signal_jepa_window_dataloaders,
)
from src.evaluation.metrics_signal_jepa_prelocal import (
    evaluate_and_save_fold,
    evaluate_validation_epoch,
)
from src.evaluation.plots_signal_jepa_prelocal import (
    save_confusion_matrix_plot,
    save_fold_history_and_plots,
    save_global_fold_comparison,
)
from src.models.signal_jepa_prelocal_model import (
    count_parameters,
    create_signal_jepa_prelocal,
    print_trainable_parameters,
)
from src.tracking.mlflow_prelocal_tracker import (
    PreLocalMLflowTracker,
)
from src.training.config_signal_jepa_prelocal import (
    SignalJEPAPreLocalConfig,
)
from src.training.early_stopping import EarlyStopping
from src.training.epoch_runs_signal_jepa_prelocal import (
    run_training_epoch,
)


# ============================================================================
# 1. PARAMÈTRES DE CETTE EXPÉRIENCE
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_VERSION = "data_signal_jepa_prelocal_32ch_128hz_2s_uv"
NUMBER_OF_CHANNELS = 32

# Valeurs possibles : "scratch" ou "pretrained".
MODEL_VARIANT = "scratch"

# Valeurs possibles : "classifier_only" ou "full_finetuning".
# classifier_only n'est autorisé qu'avec MODEL_VARIANT = "pretrained".
FREEZE_STRATEGY = "full_finetuning"

# J1 permet d'abord de vérifier rapidement si le modèle quitte le hasard.
# Pour la LODO complète :
SELECTED_FOLDS = ("J1", "J2", "J4", "J5", "J7", "J8", "J10", "J15")
# SELECTED_FOLDS = ("J1",)

# Le tutoriel officiel emploie AdamW, lr=0.005 et batch_size=16 : mêmes
# valeurs déjà validées côté 19 canaux.
BATCH_SIZE = 16
NUMBER_OF_EPOCHS = 10
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-2

EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4

RANDOM_SEED = 42
DEVICE_NAME = "mps"

# PreLocal ne réutilise pas de table d'embeddings de canaux pré-entraînée :
# la variante "without-chans" (checkpoint plus léger) reste donc adaptée
# même à 32 canaux.
PRETRAINED_CHECKPOINT = "braindecode/signal-jepa_without-chans"

ENABLE_MLFLOW = False
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
MLFLOW_EXPERIMENT_NAME = "ASC_YO_YF_EXPERIMENTS"

DEVELOPMENT_DYADS = ("J1", "J2", "J4", "J5", "J7", "J8", "J10", "J15")


def create_config() -> SignalJEPAPreLocalConfig:
    """Transforme les constantes ci-dessus en configuration validée."""

    return SignalJEPAPreLocalConfig(
        project_root=PROJECT_ROOT,
        dataset_version=DATASET_VERSION,
        model_variant=MODEL_VARIANT,
        freeze_strategy=FREEZE_STRATEGY,
        batch_size=BATCH_SIZE,
        number_of_epochs=NUMBER_OF_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
        random_seed=RANDOM_SEED,
        device_name=DEVICE_NAME,
        selected_folds=SELECTED_FOLDS,
        pretrained_checkpoint=PRETRAINED_CHECKPOINT,
        number_of_channels=NUMBER_OF_CHANNELS,
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
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    torch.save(state_dict_cpu, path)


def train_one_fold(
    validation_dyad: str,
    classification_table: pd.DataFrame,
    config: SignalJEPAPreLocalConfig,
    results_dir: Path,
    models_dir: Path,
    device: torch.device,
    tracker: PreLocalMLflowTracker,
) -> tuple[dict[str, list[float]], dict[str, float | int]]:
    """Entraîne un modèle neuf puis évalue son meilleur checkpoint."""

    train_dyads = [
        dyad for dyad in DEVELOPMENT_DYADS if dyad != validation_dyad
    ]

    train_loader, validation_loader = create_signal_jepa_window_dataloaders(
        classification_table=classification_table,
        dataset_root=config.dataset_root,
        train_dyads=train_dyads,
        validation_dyads=[validation_dyad],
        batch_size=config.batch_size,
        random_seed=config.random_seed,
        expected_number_of_channels=config.number_of_channels,
    )

    set_seed(config.random_seed)
    model = create_signal_jepa_prelocal(
        model_variant=config.model_variant,
        freeze_strategy=config.freeze_strategy,
        checkpoint_name=config.pretrained_checkpoint,
        number_of_channels=config.number_of_channels,
    ).to(device)

    total_parameters, trainable_parameters = count_parameters(model)
    if trainable_parameters == 0:
        raise RuntimeError("Le modèle ne possède aucun paramètre entraînable.")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    print("=" * 72)
    print(f"FOLD - validation : {validation_dyad}")
    print("Train             :", train_dyads)
    print("Modèle            : SignalJEPA_PreLocal (32 canaux)")
    print("Initialisation    :", config.model_variant)
    print("Stratégie         :", config.freeze_strategy)
    print(f"Entrée            : [batch, {config.number_of_channels}, 256]")
    print("Paramètres totaux :", f"{total_parameters:,}")
    print("Entraînables      :", f"{trainable_parameters:,}")
    print("Appareil          :", device)
    print("=" * 72)
    print_trainable_parameters(model)

    history = {
        "train_window_loss": [],
        "train_window_accuracy": [],
        "validation_window_loss": [],
        "validation_window_accuracy": [],
        "validation_participant_loss": [],
        "validation_participant_accuracy": [],
    }

    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )

    best_epoch = 0
    best_validation_accuracy = 0.0
    best_model_path = models_dir / f"best_model_fold_{validation_dyad}.pt"

    for epoch_index in range(config.number_of_epochs):
        epoch_number = epoch_index + 1

        train_loss, train_accuracy = run_training_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        validation_metrics = evaluate_validation_epoch(
            model=model,
            loader=validation_loader,
            device=device,
        )

        history["train_window_loss"].append(train_loss)
        history["train_window_accuracy"].append(train_accuracy)
        history["validation_window_loss"].append(
            validation_metrics["window_loss"]
        )
        history["validation_window_accuracy"].append(
            validation_metrics["window_accuracy"]
        )
        history["validation_participant_loss"].append(
            validation_metrics["participant_loss"]
        )
        history["validation_participant_accuracy"].append(
            validation_metrics["participant_accuracy"]
        )

        tracker.log_epoch_metrics(
            epoch=epoch_number,
            metrics={
                "train_window_loss": train_loss,
                "train_window_accuracy": train_accuracy,
                "validation_window_loss": validation_metrics["window_loss"],
                "validation_window_accuracy": validation_metrics[
                    "window_accuracy"
                ],
                "validation_participant_loss": validation_metrics[
                    "participant_loss"
                ],
                "validation_participant_accuracy": validation_metrics[
                    "participant_accuracy"
                ],
            },
        )

        improved = early_stopping.update(
            validation_metrics["participant_loss"]
        )
        if improved:
            best_epoch = epoch_number
            best_validation_accuracy = validation_metrics[
                "participant_accuracy"
            ]
            save_checkpoint_on_cpu(model, best_model_path)

        print(
            f"Fold {validation_dyad} | "
            f"Epoch {epoch_number:03d}/{config.number_of_epochs} | "
            f"Train fenêtre loss {train_loss:.4f}, acc {train_accuracy:.4f} | "
            "Val participant loss "
            f"{validation_metrics['participant_loss']:.4f}, "
            f"acc {validation_metrics['participant_accuracy']:.4f}"
        )
        print(
            "Epochs sans amélioration : "
            f"{early_stopping.epochs_without_improvement}/"
            f"{config.early_stopping_patience}"
        )

        if early_stopping.should_stop():
            print(
                f"Early stopping — fold {validation_dyad}, "
                f"meilleure epoch : {best_epoch}"
            )
            break

    if best_epoch == 0:
        raise RuntimeError("Aucun checkpoint n'a été sauvegardé.")

    save_fold_history_and_plots(
        validation_dyad=validation_dyad,
        history=history,
        results_dir=results_dir,
    )

    best_state_dict = torch.load(
        best_model_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_state_dict)

    final_metrics, matrix = evaluate_and_save_fold(
        validation_dyad=validation_dyad,
        model=model,
        loader=validation_loader,
        results_dir=results_dir,
        device=device,
    )
    save_confusion_matrix_plot(
        validation_dyad=validation_dyad,
        matrix=matrix,
        results_dir=results_dir,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_validation_loss": early_stopping.best_loss,
        "best_validation_accuracy": best_validation_accuracy,
        "final_train_window_loss": history["train_window_loss"][-1],
        "final_train_window_accuracy": history["train_window_accuracy"][-1],
        "best_checkpoint_participant_loss": final_metrics["participant_loss"],
        "best_checkpoint_participant_accuracy": final_metrics[
            "participant_accuracy"
        ],
        "best_checkpoint_window_loss": final_metrics["window_loss"],
        "best_checkpoint_window_accuracy": final_metrics["window_accuracy"],
    }

    tracker.log_metrics(summary)
    tracker.log_metrics({
        f"evaluation_{metric_name}": metric_value
        for metric_name, metric_value in final_metrics.items()
    })
    tracker.log_artifact(best_model_path, artifact_path="checkpoint")
    tracker.log_artifacts(
        results_dir / f"fold_{validation_dyad}",
        artifact_path="fold_results",
    )

    return history, summary


def save_config(
    config: SignalJEPAPreLocalConfig,
    results_dir: Path,
) -> None:
    """Enregistre les paramètres afin que l'expérience soit reproductible."""

    config_dictionary = asdict(config)
    config_dictionary["project_root"] = str(config.project_root)
    config_dictionary["selected_folds"] = list(config.selected_folds)

    (results_dir / "experiment_config.json").write_text(
        json.dumps(config_dictionary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    """Exécute les folds sélectionnés et produit le résumé global."""

    config = create_config()
    validate_folds(config.selected_folds)
    device = select_device(config.device_name)

    if not config.dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {config.dataset_root}. "
            "Exécuter d'abord "
            "python -m src.dataset.prepare_signal_jepa_prelocal_32ch"
        )

    results_dir = PROJECT_ROOT / "results" / config.experiment_name
    models_dir = PROJECT_ROOT / "models" / config.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, results_dir)

    classification_table = create_classification_table(PROJECT_ROOT)
    histories = {}
    summaries = []

    tracker = PreLocalMLflowTracker(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        enabled=ENABLE_MLFLOW,
    )

    print("Expérience :", config.experiment_name)
    print("Dataset    :", config.dataset_root)
    print("Canaux     :", config.number_of_channels)
    print("Folds      :", list(config.selected_folds))
    print("Z-score    : False")
    print("Unité      : microvolts")
    print("MLflow     :", "activé" if ENABLE_MLFLOW else "désactivé")

    for validation_dyad in config.selected_folds:
        train_dyads = [
            dyad for dyad in DEVELOPMENT_DYADS if dyad != validation_dyad
        ]
        fold_parameters = {
            "dataset_version": config.dataset_version,
            "model_family": "signal_jepa_prelocal",
            "model_variant": config.model_variant,
            "freeze_strategy": config.freeze_strategy,
            "pretrained_checkpoint": (
                config.pretrained_checkpoint
                if config.uses_pretrained_weights
                else "none"
            ),
            "validation_dyad": validation_dyad,
            "training_dyads": ",".join(train_dyads),
            "protocol": "Leave-One-Dyad-Out",
            "number_of_channels": config.number_of_channels,
            "number_of_timepoints": config.number_of_timepoints,
            "sampling_frequency_hz": config.sampling_frequency,
            "windows_per_participant": config.windows_per_participant,
            "window_duration_seconds": 2.0,
            "data_unit": "microvolts",
            "standardized": False,
            "batch_size": config.batch_size,
            "maximum_epochs": config.number_of_epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_min_delta": config.early_stopping_min_delta,
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
    save_global_fold_comparison(histories, results_dir)

    mean_accuracy = summary_table["best_checkpoint_participant_accuracy"].mean()
    mean_loss = summary_table["best_checkpoint_participant_loss"].mean()
    mean_window_accuracy = summary_table["best_checkpoint_window_accuracy"].mean()
    mean_window_loss = summary_table["best_checkpoint_window_loss"].mean()

    if len(summary_table) > 1:
        std_accuracy = summary_table["best_checkpoint_participant_accuracy"].std()
        std_loss = summary_table["best_checkpoint_participant_loss"].std()
        std_window_accuracy = summary_table["best_checkpoint_window_accuracy"].std()
        std_window_loss = summary_table["best_checkpoint_window_loss"].std()
    else:
        std_accuracy = 0.0
        std_loss = 0.0
        std_window_accuracy = 0.0
        std_window_loss = 0.0

    tracker.log_summary_run(
        run_name=f"{config.experiment_name}__summary",
        parameters={
            "dataset_version": config.dataset_version,
            "model_family": "signal_jepa_prelocal",
            "model_variant": config.model_variant,
            "freeze_strategy": config.freeze_strategy,
            "protocol": "Leave-One-Dyad-Out",
            "selected_folds": ",".join(config.selected_folds),
            "number_of_folds": len(config.selected_folds),
            "number_of_channels": config.number_of_channels,
            "number_of_timepoints": config.number_of_timepoints,
            "sampling_frequency_hz": config.sampling_frequency,
            "windows_per_participant": config.windows_per_participant,
            "standardized": False,
            "batch_size": config.batch_size,
            "maximum_epochs": config.number_of_epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "random_seed": config.random_seed,
        },
        metrics={
            "participant_accuracy_mean": float(mean_accuracy),
            "participant_accuracy_std": float(std_accuracy),
            "participant_loss_mean": float(mean_loss),
            "participant_loss_std": float(std_loss),
            "window_accuracy_mean": float(mean_window_accuracy),
            "window_accuracy_std": float(std_window_accuracy),
            "window_loss_mean": float(mean_window_loss),
            "window_loss_std": float(std_window_loss),
        },
        results_dir=results_dir,
    )

    print("\n" + "=" * 72)
    print("RÉSUMÉ GLOBAL — SIGNALJEPA PRELOCAL, 32 CANAUX")
    print("=" * 72)
    print(summary_table.to_string(index=False))
    print(
        "\nAccuracy participant moyenne : "
        f"{mean_accuracy:.4f} ± {std_accuracy:.4f}"
    )
    print(
        "Loss participant moyenne     : "
        f"{mean_loss:.4f} ± {std_loss:.4f}"
    )
    print("Résultats                    :", results_dir)


if __name__ == "__main__":
    main()
