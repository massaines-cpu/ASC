"""Suivi MLflow dédié aux expériences SignalJEPA PreLocal.

Cette petite couche garde toute la logique MLflow hors du code scientifique.
Lorsque ``enabled=False``, les méthodes ne font rien et l'entraînement reste
strictement identique.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import mlflow


class PreLocalMLflowTracker:
    """Enregistre les métriques fenêtre et participant d'une expérience."""

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        enabled: bool,
    ) -> None:
        self.enabled = enabled

        if self.enabled:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

    @contextmanager
    def fold_run(
        self,
        run_name: str,
        parameters: dict,
    ) -> Iterator[None]:
        """Ouvre un run MLflow indépendant pour un fold LODO."""

        if not self.enabled:
            yield
            return

        with mlflow.start_run(
            run_name=run_name,
            tags={
                "run_role": "fold",
                "model_family": "signal_jepa_prelocal",
            },
        ):
            mlflow.log_params(parameters)
            yield

    def log_epoch_metrics(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Enregistre les six courbes train/validation à une epoch donnée."""

        if self.enabled:
            mlflow.log_metrics(metrics, step=epoch)

    def log_metrics(self, metrics: dict[str, float | int]) -> None:
        """Enregistre uniquement les valeurs numériques finies."""

        if not self.enabled:
            return

        finite_metrics = {
            metric_name: float(metric_value)
            for metric_name, metric_value in metrics.items()
            if float(metric_value) == float(metric_value)
        }
        mlflow.log_metrics(finite_metrics)

    def log_artifact(
        self,
        path: Path,
        artifact_path: str | None = None,
    ) -> None:
        """Ajoute un fichier s'il existe."""

        if self.enabled and path.exists():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(
        self,
        directory: Path,
        artifact_path: str | None = None,
    ) -> None:
        """Ajoute récursivement le contenu d'un dossier s'il existe."""

        if self.enabled and directory.exists():
            mlflow.log_artifacts(
                str(directory),
                artifact_path=artifact_path,
            )

    def log_summary_run(
        self,
        run_name: str,
        parameters: dict,
        metrics: dict[str, float],
        results_dir: Path,
    ) -> None:
        """Crée un run global contenant moyenne, écart-type et tableaux."""

        if not self.enabled:
            return

        with mlflow.start_run(
            run_name=run_name,
            tags={
                "run_role": "lodo_summary",
                "model_family": "signal_jepa_prelocal",
            },
        ):
            mlflow.log_params(parameters)
            self.log_metrics(metrics)
            self.log_artifact(
                results_dir / "lodo_cv_summary.csv",
                artifact_path="summary",
            )
            self.log_artifact(
                results_dir / "experiment_config.json",
                artifact_path="summary",
            )
            self.log_artifact(
                results_dir / "all_folds_participant_comparison.png",
                artifact_path="summary",
            )
