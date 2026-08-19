"""Importe dans MLflow les résultats PreLocal déjà calculés.

Ce script lit les CSV, JSON, figures et rapports présents dans ``results/``.
Il ne relance aucun entraînement et ne modifie aucun résultat existant.

Les dossiers importés sont détectés automatiquement à partir du motif
``*signal_jepa_prelocal*/lodo_cv_summary.csv``. Cela inclut normalement :

* scratch + full_finetuning ;
* pretrained + classifier_only ;
* pretrained + full_finetuning.

Le script crée un run récapitulatif par expérience, puis un run enfant par
fold. Un tag unique empêche un second lancement de créer des doublons.
"""

import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd


# ============================================================================
# 1. PARAMÈTRES À VÉRIFIER AVANT LE LANCEMENT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
MODELS_ROOT = PROJECT_ROOT / "models"

# Le projet ASC utilise le port 5001 afin d'éviter le service AirPlay qui peut
# répondre sur le port 5000 des Mac.
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
MLFLOW_EXPERIMENT_NAME = "ASC_YO_YF_EXPERIMENTS"

# False évite les doublons. Passer volontairement à True crée une nouvelle
# copie de chaque expérience dans MLflow.
FORCE_REIMPORT = False

RESULT_DIRECTORY_PATTERN = "*signal_jepa_prelocal*"


def normalize_parameter_value(value):
    """Transforme listes et dictionnaires en chaînes acceptées par MLflow."""

    if value is None:
        return "none"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Path):
        return str(value)
    return value


def load_experiment_config(experiment_directory: Path) -> dict:
    """Charge le JSON produit au lancement de la LODO."""

    config_path = experiment_directory / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration absente : {config_path}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))

    # Un chemin absolu local n'aide pas à comparer les expériences et peut
    # varier d'une machine à l'autre.
    config.pop("project_root", None)

    return {
        key: normalize_parameter_value(value)
        for key, value in config.items()
    }


def find_experiment_directories() -> list[Path]:
    """Détecte uniquement les dossiers contenant un résumé exploitable."""

    directories = []
    for candidate in sorted(RESULTS_ROOT.glob(RESULT_DIRECTORY_PATTERN)):
        if (candidate / "lodo_cv_summary.csv").exists():
            directories.append(candidate)
    return directories


def import_key_for(experiment_directory: Path) -> str:
    """Produit un identifiant stable utilisé pour détecter les doublons."""

    return f"asc_prelocal_import::{experiment_directory.name}"


def already_imported(
    client: MlflowClient,
    experiment_id: str,
    import_key: str,
) -> bool:
    """Vérifie si un run récapitulatif terminé possède déjà ce tag."""

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        max_results=5000,
    )

    return any(
        run.info.status == "FINISHED"
        and run.data.tags.get("asc_import_key") == import_key
        and run.data.tags.get("run_role") == "lodo_summary"
        for run in runs
    )


def calculate_summary_metrics(summary_table: pd.DataFrame) -> dict[str, float]:
    """Calcule les métriques globales à partir des huit folds disponibles."""

    metric_columns = {
        "participant_accuracy": "best_checkpoint_participant_accuracy",
        "participant_loss": "best_checkpoint_participant_loss",
        "window_accuracy": "best_checkpoint_window_accuracy",
        "window_loss": "best_checkpoint_window_loss",
        "best_epoch": "best_epoch",
    }

    metrics = {"number_of_folds": float(len(summary_table))}

    for metric_prefix, csv_column in metric_columns.items():
        if csv_column not in summary_table.columns:
            continue

        values = summary_table[csv_column].astype(float)
        metrics[f"{metric_prefix}_mean"] = float(values.mean())
        metrics[f"{metric_prefix}_std"] = (
            float(values.std(ddof=1))
            if len(values) > 1
            else 0.0
        )

    return metrics


def log_history(history_path: Path) -> None:
    """Importe toutes les courbes enregistrées à chaque epoch."""

    if not history_path.exists():
        return

    history = pd.read_csv(history_path)
    metric_columns = [
        "train_window_loss",
        "train_window_accuracy",
        "validation_window_loss",
        "validation_window_accuracy",
        "validation_participant_loss",
        "validation_participant_accuracy",
    ]

    for row_index, row in history.iterrows():
        epoch = (
            int(row["epoch"])
            if "epoch" in history.columns
            else row_index + 1
        )
        metrics = {
            metric_name: float(row[metric_name])
            for metric_name in metric_columns
            if metric_name in history.columns
            and np.isfinite(float(row[metric_name]))
        }
        if metrics:
            mlflow.log_metrics(metrics, step=epoch)


def log_fold(
    experiment_directory: Path,
    result_row: pd.Series,
    common_parameters: dict,
) -> None:
    """Crée un run enfant contenant métriques, courbes et artefacts du fold."""

    validation_dyad = str(result_row["validation_dyad"])
    fold_directory = experiment_directory / f"fold_{validation_dyad}"

    with mlflow.start_run(
        run_name=f"{experiment_directory.name}__fold_{validation_dyad}",
        nested=True,
        tags={
            "run_role": "fold",
            "model_family": "signal_jepa_prelocal",
        },
    ):
        fold_parameters = {
            **common_parameters,
            "validation_dyad": validation_dyad,
            "protocol": "Leave-One-Dyad-Out",
            "imported_from_existing_results": True,
        }
        mlflow.log_params(fold_parameters)

        numeric_metrics = {}
        for column_name, value in result_row.items():
            if column_name == "validation_dyad":
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value):
                numeric_metrics[column_name] = numeric_value

        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)

        log_history(fold_directory / "history.csv")

        if fold_directory.exists():
            mlflow.log_artifacts(
                str(fold_directory),
                artifact_path="fold_results",
            )

        checkpoint_path = (
            MODELS_ROOT
            / experiment_directory.name
            / f"best_model_fold_{validation_dyad}.pt"
        )
        if checkpoint_path.exists():
            mlflow.log_artifact(
                str(checkpoint_path),
                artifact_path="checkpoint",
            )


def import_experiment(
    experiment_directory: Path,
    experiment_id: str,
    client: MlflowClient,
) -> bool:
    """Importe une expérience complète et retourne True si elle a été créée."""

    summary_path = experiment_directory / "lodo_cv_summary.csv"
    summary_table = pd.read_csv(summary_path)
    if summary_table.empty:
        raise ValueError(f"Résumé vide : {summary_path}")

    common_parameters = load_experiment_config(experiment_directory)
    import_key = import_key_for(experiment_directory)

    if not FORCE_REIMPORT and already_imported(
        client,
        experiment_id,
        import_key,
    ):
        print(f"Déjà importée, ignorée : {experiment_directory.name}")
        return False

    print(
        f"Import de {experiment_directory.name} "
        f"({len(summary_table)} folds)"
    )

    with mlflow.start_run(
        run_name=f"{experiment_directory.name}__summary",
        tags={
            "asc_import_key": import_key,
            "run_role": "lodo_summary",
            "model_family": "signal_jepa_prelocal",
            "imported_from_existing_results": "true",
        },
    ):
        mlflow.log_params({
            **common_parameters,
            "protocol": "Leave-One-Dyad-Out",
            "imported_from_existing_results": True,
        })
        mlflow.log_metrics(calculate_summary_metrics(summary_table))

        mlflow.log_artifact(str(summary_path), artifact_path="summary")

        config_path = experiment_directory / "experiment_config.json"
        mlflow.log_artifact(str(config_path), artifact_path="summary")

        global_figure = (
            experiment_directory
            / "all_folds_participant_comparison.png"
        )
        if global_figure.exists():
            mlflow.log_artifact(
                str(global_figure),
                artifact_path="summary",
            )

        for _, result_row in summary_table.iterrows():
            log_fold(
                experiment_directory=experiment_directory,
                result_row=result_row,
                common_parameters=common_parameters,
            )

    return True


def main() -> None:
    """Connecte MLflow, détecte les résultats et effectue l'import."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        client = MlflowClient()
        # Cette lecture simple produit immédiatement une erreur claire si le
        # serveur local n'est pas lancé ou si le mauvais port est utilisé.
        client.get_experiment(experiment.experiment_id)
    except Exception as error:
        raise RuntimeError(
            "Connexion impossible à MLflow sur "
            f"{MLFLOW_TRACKING_URI}. Vérifier que le serveur utilise bien "
            "le port 5001 avant de relancer l'import."
        ) from error

    experiment_directories = find_experiment_directories()
    if not experiment_directories:
        raise FileNotFoundError(
            "Aucun dossier PreLocal contenant lodo_cv_summary.csv n'a été "
            f"trouvé dans {RESULTS_ROOT}."
        )

    imported_count = 0
    for experiment_directory in experiment_directories:
        imported = import_experiment(
            experiment_directory=experiment_directory,
            experiment_id=experiment.experiment_id,
            client=client,
        )
        imported_count += int(imported)

    print("\nImport terminé.")
    print("Expériences détectées :", len(experiment_directories))
    print("Expériences importées :", imported_count)
    print("Expériences ignorées  :", len(experiment_directories) - imported_count)
    print("Expérience MLflow      :", MLFLOW_EXPERIMENT_NAME)
    print("Serveur MLflow         :", MLFLOW_TRACKING_URI)


if __name__ == "__main__":
    main()
