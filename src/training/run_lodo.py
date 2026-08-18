"""Point d'entrée du protocole Leave-One-Dyad-Out ASC.

Le script entraîne toujours un modèle neuf pour chaque dyade de validation.
Il prend en charge les architectures historiques et les deux variantes
comparables de SignalJEPA.

Test SignalJEPA conseillé
-------------------------
Un seul fold et trois epochs, sans MLflow :

    python -m src.training.run_lodo \
        --model-name signal_jepa_pretrained \
        --folds J1 \
        --epochs 3 \
        --patience 3 \
        --batch-size 2 \
        --disable-mlflow
"""

from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

from src.dataset.dataloader_participant import create_participant_dataloaders
from src.dataset.labels import prepare_classification_table
from src.evaluation.metrics import collect_predictions, evaluate_fold
from src.evaluation.plots import (
    save_confusion_matrix_plot,
    save_fold_plots,
    save_global_comparison,
)
from src.tracking.mlflow_track import MLflowTracker
from src.training.config import ExperimentConfig, SIGNAL_JEPA_MODEL_NAMES
from src.training.early_stopping import EarlyStopping
from src.training.epoch_runs import run_epoch
from src.training.model_fabrication import create_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_DYADS = ["J1", "J2", "J4", "J5", "J7", "J8", "J10", "J15"]
TEST_DYADS: list[str] = []

MODEL_NAMES = [
    "linear",
    "non_linear",
    "small_cnn",
    "eegnet",
    "signal_jepa_scratch",
    "signal_jepa_pretrained",
]

# Le port 5000 est utilisé par ControlCenter/AirPlay sur certains Mac.
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
MLFLOW_EXPERIMENT_NAME = "ASC_YO_YF_EXPERIMENTS"


def set_seed(seed: int) -> None:
    """Fixe les générateurs utilisés pour les poids et le Dropout."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_device(device_name: str) -> torch.device:
    """Sélectionne explicitement CPU, CUDA ou le GPU MPS du Mac."""

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


def create_classification_table(project_root: Path) -> pd.DataFrame:
    """Charge les métadonnées et conserve seulement les classes YO/YF."""

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


def save_checkpoint_on_cpu(model: nn.Module, checkpoint_path: Path) -> None:
    """Sauvegarde un state_dict portable après entraînement sur GPU."""

    cpu_state_dict = {
        parameter_name: parameter.detach().cpu()
        for parameter_name, parameter in model.state_dict().items()
    }
    torch.save(cpu_state_dict, checkpoint_path)


def train_one_fold(
    validation_dyad: str,
    train_dyads: list[str],
    classification_table: pd.DataFrame,
    config: ExperimentConfig,
    results_dir: Path,
    models_dir: Path,
    tracker: MLflowTracker,
    device: torch.device,
) -> tuple[dict[str, list[float]], dict[str, float | int]]:
    """Entraîne puis évalue un modèle sur un fold LODO."""

    if validation_dyad in train_dyads:
        raise ValueError(
            f"La dyade {validation_dyad} est présente dans train et validation."
        )

    train_loader, validation_loader, _ = create_participant_dataloaders(
        classification_table=classification_table,
        dataset_root=config.dataset_root,
        train_dyads=train_dyads,
        validation_dyads=[validation_dyad],
        test_dyads=TEST_DYADS,
        batch_size=config.batch_size,
        standardize=config.standardize,
        expected_number_of_channels=config.number_of_channels,
        expected_number_of_timepoints=config.number_of_timepoints,
    )

    # La seed est réinitialisée avant chaque fold afin de rendre
    # l'initialisation de chaque expérience reproductible.
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
    )
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    # Les paramètres gelés sont exclus de l'optimiseur. Cela rend la
    # stratégie classifier_only explicite et évite des calculs inutiles.
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("Le modèle ne contient aucun paramètre entraînable.")

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=config.learning_rate,
    )

    total_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    print(
        f"Modèle : {config.model_name} | "
        f"Paramètres totaux : {total_parameter_count:,} | "
        f"Paramètres entraînables : {trainable_parameter_count:,} | "
        f"Appareil : {device}"
    )

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
    best_validation_accuracy = 0.0
    best_model_path = models_dir / f"best_model_fold_{validation_dyad}.pt"

    run_name = f"{config.experiment_name}_fold_{validation_dyad}"
    run_parameters = {
        "dataset": config.dataset_version,
        "model": config.model_name,
        "hidden_layer_size": config.hidden_layer_size,
        "dropout_rate": config.dropout_rate,
        "validation_dyad": validation_dyad,
        "training_dyads": ",".join(train_dyads),
        "protocol": "Leave-One-Dyad-Out",
        "standardized": config.standardize,
        "number_of_channels": config.number_of_channels,
        "number_of_timepoints": config.number_of_timepoints,
        "sampling_frequency_hz": config.sampling_frequency,
        "preprocessing": config.preprocessing_name,
        "data_unit": (
            "microvolts"
            if config.is_signal_jepa
            else ("unitless_z_score" if config.standardize else "volts")
        ),
        "training_type": (
            "pretrained"
            if config.uses_pretrained_weights
            else "from_scratch"
        ),
        "pretrained_checkpoint": (
            config.pretrained_checkpoint
            if config.uses_pretrained_weights
            else "none"
        ),
        "freeze_strategy": config.freeze_strategy,
        "device": str(device),
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "maximum_epochs": config.number_of_epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "random_seed": config.random_seed,
        "total_parameters": total_parameter_count,
        "trainable_parameters": trainable_parameter_count,
        "loss_function": criterion.__class__.__name__,
    }

    with tracker.fold_run(run_name, run_parameters):
        for epoch_index in range(config.number_of_epochs):
            epoch_number = epoch_index + 1

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
                best_validation_accuracy = validation_accuracy
                save_checkpoint_on_cpu(model, best_model_path)

            print(
                f"Fold {validation_dyad} | "
                f"Epoch {epoch_number:03d}/{config.number_of_epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Train acc: {train_accuracy:.4f} | "
                f"Val loss: {validation_loss:.4f} | "
                f"Val acc: {validation_accuracy:.4f}"
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
            raise RuntimeError("Aucun meilleur checkpoint n'a été sauvegardé.")

        save_fold_plots(validation_dyad, history, results_dir)

        best_state_dict = torch.load(
            best_model_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(best_state_dict)

        evaluation_metrics = evaluate_fold(
            validation_dyad=validation_dyad,
            model=model,
            loader=validation_loader,
            results_dir=results_dir,
            device=device,
        )

        labels, predictions, _ = collect_predictions(
            model,
            validation_loader,
            device,
        )
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        save_confusion_matrix_plot(
            validation_dyad,
            matrix,
            results_dir,
        )

        tracker.log_best_metrics(
            best_epoch=best_epoch,
            best_validation_loss=early_stopping.best_loss,
            best_validation_accuracy=best_validation_accuracy,
        )
        tracker.log_evaluation_metrics(evaluation_metrics)
        tracker.log_artifact(best_model_path, artifact_path="checkpoint")
        tracker.log_artifacts(
            results_dir / f"fold_{validation_dyad}",
            artifact_path=f"fold_{validation_dyad}",
        )

    summary = {
        "best_epoch": best_epoch,
        "best_validation_loss": early_stopping.best_loss,
        "best_validation_accuracy": best_validation_accuracy,
        "final_train_loss": history["train_loss"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_validation_loss": history["validation_loss"][-1],
        "final_validation_accuracy": history["validation_accuracy"][-1],
    }
    return history, summary


def parse_arguments():
    """Permet de lancer une expérience sans modifier le code source."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        choices=MODEL_NAMES,
        default="non_linear",
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help=(
            "Par défaut : data_final pour les anciens modèles et "
            "data_signal_jepa_128hz_uv pour SignalJEPA."
        ),
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=None,
        help="Dyades à exécuter, par exemple --folds J1 ou --folds J1 J2.",
    )
    parser.add_argument("--hidden-layer-size", type=int, default=32)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Par défaut : 5, ou 2 pour SignalJEPA afin de limiter la mémoire.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--standardize",
        action=BooleanOptionalAction,
        default=None,
        help=(
            "Par défaut : activé pour les anciens modèles et désactivé "
            "pour SignalJEPA. Options : --standardize / --no-standardize."
        ),
    )
    parser.add_argument("--number-of-timepoints", type=int, default=None)
    parser.add_argument("--sampling-frequency", type=float, default=None)
    parser.add_argument(
        "--pretrained-checkpoint",
        default="braindecode/signal-jepa",
    )
    parser.add_argument(
        "--freeze-strategy",
        choices=["full_finetuning", "classifier_only"],
        default="full_finetuning",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Exécute l'expérience sans contacter le serveur MLflow.",
    )
    return parser.parse_args()


def create_config_from_arguments(arguments) -> ExperimentConfig:
    """Résout les valeurs SignalJEPA sans changer les anciens modèles."""

    is_signal_jepa = arguments.model_name in SIGNAL_JEPA_MODEL_NAMES

    dataset_version = arguments.dataset_version
    if dataset_version is None:
        dataset_version = (
            "data_signal_jepa_128hz_uv"
            if is_signal_jepa
            else "data_final"
        )

    standardize = arguments.standardize
    if standardize is None:
        standardize = not is_signal_jepa

    number_of_timepoints = arguments.number_of_timepoints
    if number_of_timepoints is None:
        number_of_timepoints = 1280 if is_signal_jepa else 5120

    sampling_frequency = arguments.sampling_frequency
    if sampling_frequency is None:
        sampling_frequency = 128.0 if is_signal_jepa else 512.0

    batch_size = arguments.batch_size
    if batch_size is None:
        batch_size = 2 if is_signal_jepa else 5

    return ExperimentConfig(
        project_root=PROJECT_ROOT,
        dataset_version=dataset_version,
        model_name=arguments.model_name,
        hidden_layer_size=arguments.hidden_layer_size,
        dropout_rate=arguments.dropout_rate,
        batch_size=batch_size,
        number_of_epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        early_stopping_patience=arguments.patience,
        early_stopping_min_delta=arguments.min_delta,
        random_seed=arguments.seed,
        standardize=standardize,
        number_of_channels=32,
        number_of_timepoints=number_of_timepoints,
        sampling_frequency=sampling_frequency,
        pretrained_checkpoint=arguments.pretrained_checkpoint,
        freeze_strategy=arguments.freeze_strategy,
        device_name=arguments.device,
    )


def validate_selected_folds(requested_folds: list[str] | None) -> list[str]:
    """Valide un sous-ensemble de folds pour les tests courts."""

    if requested_folds is None:
        return DEVELOPMENT_DYADS.copy()

    unknown_folds = set(requested_folds) - set(DEVELOPMENT_DYADS)
    if unknown_folds:
        raise ValueError(
            f"Dyades inconnues : {sorted(unknown_folds)}. "
            f"Valeurs possibles : {DEVELOPMENT_DYADS}."
        )

    return list(dict.fromkeys(requested_folds))


def save_experiment_config(
    config: ExperimentConfig,
    selected_folds: list[str],
    results_dir: Path,
) -> None:
    """Enregistre la configuration exacte utilisée."""

    config_dictionary = asdict(config)
    config_dictionary["project_root"] = str(config.project_root)
    config_dictionary["selected_folds"] = selected_folds

    config_path = results_dir / "experiment_config.json"
    config_path.write_text(
        json.dumps(config_dictionary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    """Exécute les folds demandés puis produit le résumé global."""

    arguments = parse_arguments()
    config = create_config_from_arguments(arguments)
    selected_folds = validate_selected_folds(arguments.folds)
    device = select_device(config.device_name)

    if not config.dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {config.dataset_root}. "
            "Pour SignalJEPA, exécute d'abord "
            "python -m src.dataset.prepare_signal_jepa_dataset."
        )

    if config.is_signal_jepa and config.standardize:
        print(
            "ATTENTION : SignalJEPA est lancé avec un Z-score. Cette option "
            "constitue une ablation distincte du protocole principal en "
            "microvolts sans Z-score."
        )

    results_dir = PROJECT_ROOT / "results" / config.experiment_name
    models_dir = PROJECT_ROOT / "models" / config.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(config, selected_folds, results_dir)

    tracker = MLflowTracker(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        enabled=not arguments.disable_mlflow,
    )
    classification_table = create_classification_table(PROJECT_ROOT)
    set_seed(config.random_seed)

    print("Appareil sélectionné :", device)
    print("Dataset              :", config.dataset_root)
    print("Standardisation      :", config.standardize)
    print("Folds demandés       :", selected_folds)

    all_histories = {}
    fold_summaries = []

    for validation_dyad in selected_folds:
        # Même lors d'un test limité à J1, l'entraînement utilise toutes les
        # autres dyades, conformément au protocole LODO.
        train_dyads = [
            dyad
            for dyad in DEVELOPMENT_DYADS
            if dyad != validation_dyad
        ]

        print("=" * 70)
        print(f"FOLD - dyade de validation : {validation_dyad}")
        print(f"Train dyads      : {train_dyads}")
        print(f"Validation dyad  : {validation_dyad}")
        print("Test dyads isolés: []")
        print("=" * 70)

        history, summary = train_one_fold(
            validation_dyad=validation_dyad,
            train_dyads=train_dyads,
            classification_table=classification_table,
            config=config,
            results_dir=results_dir,
            models_dir=models_dir,
            tracker=tracker,
            device=device,
        )
        all_histories[validation_dyad] = history
        fold_summaries.append({
            "validation_dyad": validation_dyad,
            **summary,
        })

        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    summary_table = pd.DataFrame(fold_summaries)
    summary_table.to_csv(results_dir / "lodo_cv_summary.csv", index=False)
    save_global_comparison(all_histories, results_dir)

    mean_accuracy = summary_table["best_validation_accuracy"].mean()
    mean_loss = summary_table["best_validation_loss"].mean()

    if len(summary_table) > 1:
        std_accuracy = summary_table["best_validation_accuracy"].std()
        std_loss = summary_table["best_validation_loss"].std()
    else:
        # Un seul fold est un test technique et ne mesure pas la variabilité.
        std_accuracy = 0.0
        std_loss = 0.0

    print("\n" + "=" * 70)
    print("RÉSUMÉ GLOBAL DE LA CROSS-VALIDATION")
    print("=" * 70)
    print(f"Expérience : {config.experiment_name}")
    print(summary_table.to_string(index=False))
    print(f"\nAccuracy moyenne : {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Loss moyenne     : {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"Résultats        : {results_dir}")


if __name__ == "__main__":
    main()
