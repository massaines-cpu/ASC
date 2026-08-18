"""Configuration unique des expériences LODO du projet ASC."""

from dataclasses import dataclass
from pathlib import Path


SIGNAL_JEPA_MODEL_NAMES = {
    "signal_jepa_scratch",
    "signal_jepa_pretrained",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Regroupe les variables qui doivent être enregistrées avec un run."""

    project_root: Path
    dataset_version: str = "data_final"
    model_name: str = "non_linear"
    hidden_layer_size: int = 32
    dropout_rate: float = 0.0
    batch_size: int = 5
    number_of_epochs: int = 100
    learning_rate: float = 1e-3
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    random_seed: int = 42
    standardize: bool = True
    number_of_channels: int = 32
    number_of_timepoints: int = 5120
    sampling_frequency: float = 512.0
    pretrained_checkpoint: str = "braindecode/signal-jepa"
    freeze_strategy: str = "full_finetuning"
    device_name: str = "auto"

    @property
    def is_signal_jepa(self) -> bool:
        """Indique si le modèle appartient à l'expérience SignalJEPA."""

        return self.model_name in SIGNAL_JEPA_MODEL_NAMES

    @property
    def uses_pretrained_weights(self) -> bool:
        """Indique si un checkpoint externe initialise le modèle."""

        return self.model_name == "signal_jepa_pretrained"

    @property
    def dataset_root(self) -> Path:
        return self.project_root / "data" / self.dataset_version

    @property
    def preprocessing_name(self) -> str:
        """Produit un nom court mais explicite pour les résultats."""

        if self.standardize:
            return "zscore"
        if self.is_signal_jepa:
            return "microvolts_no_zscore"
        return "raw_no_zscore"

    @property
    def experiment_name(self) -> str:
        """Évite l'écrasement entre configurations expérimentales."""

        if self.is_signal_jepa:
            return (
                f"{self.dataset_version}_{self.model_name}"
                f"_{self.freeze_strategy}_{self.preprocessing_name}"
            )

        return (
            f"{self.dataset_version}_{self.model_name}"
            f"_hidden_{self.hidden_layer_size}"
            f"_dropout_{self.dropout_rate}"
            f"_{self.preprocessing_name}"
        )
