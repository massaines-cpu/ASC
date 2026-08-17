"""Couche isolant MLflow du pipeline scientifique d'entraînement."""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import mlflow


class MLflowTracker:
    """Enregistre paramètres, métriques et artefacts d'un fold.

    Lorsque ``enabled=False``, toutes les méthodes deviennent inactives.
    Cela permet de lancer une expérience locale sans modifier le code
    scientifique lorsque le serveur MLflow n'est pas disponible.
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        enabled: bool = True,
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
        """Maintient un run ouvert pendant l'intégralité d'un fold."""

        if not self.enabled:
            yield
            return

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(parameters)
            yield

    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        validation_loss: float,
        validation_accuracy: float,
    ) -> None:
        """Enregistre les métriques variables à chaque epoch."""

        if not self.enabled:
            return

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            },
            step=epoch,
        )

    def log_best_metrics(
        self,
        best_epoch: int,
        best_validation_loss: float,
        best_validation_accuracy: float,
    ) -> None:
        """Enregistre les valeurs correspondant au meilleur checkpoint."""

        if not self.enabled:
            return

        mlflow.log_metrics({
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_validation_accuracy": best_validation_accuracy,
        })

    def log_evaluation_metrics(self, metrics: dict[str, float]) -> None:
        """Enregistre les métriques calculées après rechargement du modèle."""

        if not self.enabled:
            return

        # MLflow ne doit pas recevoir de valeurs NaN pour des groupes vides.
        finite_metrics = {
            name: value
            for name, value in metrics.items()
            if value == value
        }
        mlflow.log_metrics(finite_metrics)

    def log_artifact(
        self,
        file_path: Path,
        artifact_path: str | None = None,
    ) -> None:
        """Enregistre un fichier produit par l'expérience."""

        if self.enabled:
            mlflow.log_artifact(str(file_path), artifact_path=artifact_path)

    def log_artifacts(
        self,
        directory_path: Path,
        artifact_path: str | None = None,
    ) -> None:
        """Enregistre récursivement le contenu d'un dossier."""

        if self.enabled:
            mlflow.log_artifacts(
                str(directory_path),
                artifact_path=artifact_path,
            )
