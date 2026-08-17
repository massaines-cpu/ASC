"""Point d'entrée du protocole Leave-One-Dyad-Out ASC.

Exemples
--------
Tester les tailles de la couche cachée sans Dropout :

    python -m src.training.run_lodo --hidden-layer-size 128 --dropout-rate 0
    python -m src.training.run_lodo --hidden-layer-size 64  --dropout-rate 0
    python -m src.training.run_lodo --hidden-layer-size 32  --dropout-rate 0

Tester le Dropout à taille constante :

    python -m src.training.run_lodo --hidden-layer-size 128 --dropout-rate 0.2
    python -m src.training.run_lodo --hidden-layer-size 128 --dropout-rate 0.5
"""

from argparse import ArgumentParser
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
from src.training.config import ExperimentConfig
from src.training.early_stopping import EarlyStopping
from src.training.epoch_runs import run_epoch
from src.training.model_fabrication import create_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_DYADS = ["J1", "J2", "J4", "J5", "J7", "J8", "J10", "J15"]
TEST_DYADS: list[str] = []

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "ASC_YO_YF_EXPERIMENTS"


def set_seed(seed: int) -> None:
    """Fixe les générateurs utilisés pour l'initialisation et le Dropout."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_classification_table(project_root: Path) -> pd.DataFrame:
    """Charge les métadonnées et conserve uniquement YO/YF."""

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


def train_one_fold(
    validation_dyad: str,
    train_dyads: list[str],
    classification_table: pd.DataFrame,
    config: ExperimentConfig,
    results_dir: Path,
    models_dir: Path,
    tracker: MLflowTracker,
) -> tuple[dict[str, list[float]], dict[str, float | int]]:
    """Entraîne un modèle neuf et évalue son meilleur checkpoint."""

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
    )

    # Une nouvelle initialisation aléatoire est créée dans chaque fold.
    set_seed(config.random_seed)
    model = create_model(
        model_name=config.model_name,
        hidden_layer_size=config.hidden_layer_size,
        dropout_rate=config.dropout_rate,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Modèle : {config.model_name} | "
        f"Paramètres entraînables : {number_of_parameters:,}"
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
        "standardized": True,
        "training_type": "from_scratch",
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "maximum_epochs": config.number_of_epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "random_seed": config.random_seed,
        "trainable_parameters": number_of_parameters,
    }

    with tracker.fold_run(run_name, run_parameters):
        for epoch_index in range(config.number_of_epochs):
            epoch_number = epoch_index + 1

            train_loss, train_accuracy = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
            )
            validation_loss, validation_accuracy = run_epoch(
                model=model,
                loader=validation_loader,
                criterion=criterion,
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
                torch.save(model.state_dict(), best_model_path)

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

        model.load_state_dict(
            torch.load(best_model_path, weights_only=True)
        )
        evaluation_metrics = evaluate_fold(
            validation_dyad=validation_dyad,
            model=model,
            loader=validation_loader,
            results_dir=results_dir,
        )

        labels, predictions, _ = collect_predictions(model, validation_loader)
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
    """Permet de lancer plusieurs expériences sans modifier le code."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", default="data_final")
    parser.add_argument(
        "--model-name",
        choices=["linear", "non_linear", "small_cnn", "eegnet"],
        default="non_linear",
    )
    parser.add_argument("--hidden-layer-size", type=int, default=32)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Exécute l'expérience sans contacter le serveur MLflow.",
    )
    return parser.parse_args()


def main() -> None:
    """Exécute les huit folds puis produit le résumé global."""

    arguments = parse_arguments()
    config = ExperimentConfig(
        project_root=PROJECT_ROOT,
        dataset_version=arguments.dataset_version,
        model_name=arguments.model_name,
        hidden_layer_size=arguments.hidden_layer_size,
        dropout_rate=arguments.dropout_rate,
        batch_size=arguments.batch_size,
        number_of_epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        early_stopping_patience=arguments.patience,
        early_stopping_min_delta=arguments.min_delta,
        random_seed=arguments.seed,
    )

    if not config.dataset_root.exists():
        raise FileNotFoundError(f"Dataset introuvable : {config.dataset_root}")

    results_dir = PROJECT_ROOT / "results" / config.experiment_name
    models_dir = PROJECT_ROOT / "models" / config.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    tracker = MLflowTracker(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        enabled=not arguments.disable_mlflow,
    )
    classification_table = create_classification_table(PROJECT_ROOT)
    set_seed(config.random_seed)

    all_histories = {}
    fold_summaries = []

    for validation_dyad in DEVELOPMENT_DYADS:
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
        )
        all_histories[validation_dyad] = history
        fold_summaries.append({
            "validation_dyad": validation_dyad,
            **summary,
        })

    summary_table = pd.DataFrame(fold_summaries)
    summary_table.to_csv(results_dir / "lodo_cv_summary.csv", index=False)
    save_global_comparison(all_histories, results_dir)

    mean_accuracy = summary_table["best_validation_accuracy"].mean()
    std_accuracy = summary_table["best_validation_accuracy"].std()
    mean_loss = summary_table["best_validation_loss"].mean()
    std_loss = summary_table["best_validation_loss"].std()

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
