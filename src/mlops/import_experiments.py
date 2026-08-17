from pathlib import Path

import mlflow

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "ASC_YO_YF_EXPERIMENTS"
)

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"

MLFLOW_EXPERIMENT_NAME = "ASC_YO_YF_EXPERIMENTS"


def extract_experiment_information(
    experiment_name: str,
) -> dict:
    """
    Extrait les informations contenues dans un nom comme :

    experience_A_small_cnn_standardized
    experience_C_eegnet_standardized
    """

    cleaned_name = experiment_name.removeprefix("experience_")

    parts = cleaned_name.split("_")

    dataset = parts[0]

    if "small_cnn" in cleaned_name:
        model = "small_cnn"

    elif "non_linear" in cleaned_name:
        model = "non_linear_mlp"

    elif "linear" in cleaned_name:
        model = "linear_mlp"

    elif "eegnet" in cleaned_name:
        model = "eegnet"

    else:
        model = "unknown"

    standardized = "standardized" in cleaned_name

    return {
        "dataset": dataset,
        "model": model,
        "standardized": standardized,
        "training_type": "from_scratch",
    }


def log_history(
    history_path: Path,
) -> None:
    """Ajoute dans MLflow les métriques enregistrées à chaque epoch."""

    history = pd.read_csv(history_path)

    for row_index, row in history.iterrows():

        if "epoch" in history.columns:
            epoch = int(row["epoch"])
        else:
            epoch = row_index + 1

        metric_names = [
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
        ]

        for metric_name in metric_names:

            if metric_name in history.columns:

                mlflow.log_metric(
                    metric_name,
                    float(row[metric_name]),
                    step=epoch,
                )


def log_fold_artifacts(
    fold_directory: Path,
) -> None:
    """Ajoute les CSV, figures et rapports du fold."""

    artifact_files = [
        "history.csv",
        "training_curves.png",
        "accuracy_curve.png",
        "loss_curve.png",
        "evaluation_report.txt",
    ]

    for filename in artifact_files:

        artifact_path = fold_directory / filename

        if artifact_path.exists():

            mlflow.log_artifact(
                str(artifact_path),
                artifact_path="fold_results",
            )


def import_standard_experiment(
    experiment_directory: Path,
) -> None:

    experiment_information = (
        extract_experiment_information(
            experiment_directory.name
        )
    )

    summary_path = (
        experiment_directory
        / "lodo_cv_summary.csv"
    )

    if not summary_path.exists():
        print(
            f"Résumé absent : "
            f"{experiment_directory.name}"
        )
        return

    summary = pd.read_csv(summary_path)

    for _, result_row in summary.iterrows():

        validation_dyad = str(
            result_row["validation_dyad"]
        )

        fold_directory = (
            experiment_directory
            / f"fold_{validation_dyad}"
        )

        run_name = (
            f"{experiment_information['dataset']}_"
            f"{experiment_information['model']}_"
            f"{validation_dyad}"
        )

        print(f"Import : {run_name}")

        with mlflow.start_run(
            run_name=run_name
        ):

            mlflow.log_params(
                {
                    **experiment_information,
                    "validation_dyad": validation_dyad,
                    "protocol": "LODO",
                }
            )

            metric_mapping = {
                "best_epoch": "best_epoch",
                "best_validation_loss":
                    "best_validation_loss",
                "best_validation_accuracy":
                    "best_validation_accuracy",
            }

            for (
                csv_column,
                mlflow_metric,
            ) in metric_mapping.items():

                if csv_column in summary.columns:

                    mlflow.log_metric(
                        mlflow_metric,
                        float(result_row[csv_column]),
                    )

            history_path = (
                fold_directory
                / "history.csv"
            )

            if history_path.exists():
                log_history(history_path)

            if fold_directory.exists():
                log_fold_artifacts(
                    fold_directory
                )


def main() -> None:

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    experiment_directories = sorted(
        RESULTS_ROOT.glob(
            "experience_*_standardized"
        )
    )

    print(
        f"{len(experiment_directories)} "
        "expériences trouvées."
    )

    for experiment_directory in experiment_directories:

        import_standard_experiment(
            experiment_directory
        )


if __name__ == "__main__":
    main()