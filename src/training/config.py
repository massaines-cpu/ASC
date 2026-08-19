"""Configuration unique des expériences LODO du projet ASC."""

from dataclasses import dataclass
from pathlib import Path


SIGNAL_JEPA_MODEL_NAMES = {
    "signal_jepa_scratch",
    "signal_jepa_pretrained",
}

SUPPORTED_MODEL_NAMES = {
    "linear",
    "non_linear",
    "small_cnn",
    "eegnet",
    *SIGNAL_JEPA_MODEL_NAMES,
}

SUPPORTED_DEVICE_NAMES = {
    "auto",
    "cpu",
    "mps",
    "cuda",
}

SIGNAL_JEPA_FREEZE_STRATEGIES = {
    "full_finetuning",
    "classifier_only",
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

    def __post_init__(self) -> None:
        if self.model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError(
                f"Modèle inconnu : {self.model_name}. "
                f"Valeurs possibles : {sorted(SUPPORTED_MODEL_NAMES)}."
            )
        if self.device_name not in SUPPORTED_DEVICE_NAMES:
            raise ValueError(
                f"Appareil inconnu : {self.device_name}. "
                f"Valeurs possibles : {sorted(SUPPORTED_DEVICE_NAMES)}."
            )
        if not self.dataset_version:
            raise ValueError("dataset_version ne doit pas être vide.")
        if self.batch_size <= 0:
            raise ValueError("batch_size doit être strictement positif.")
        if self.number_of_epochs <= 0:
            raise ValueError("number_of_epochs doit être strictement positif.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate doit être strictement positif.")
        if self.early_stopping_patience <= 0:
            raise ValueError(
                "early_stopping_patience doit être strictement positif."
            )
        if self.early_stopping_min_delta < 0:
            raise ValueError(
                "early_stopping_min_delta doit être positif ou nul."
            )
        if self.hidden_layer_size <= 0:
            raise ValueError("hidden_layer_size doit être strictement positif.")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate doit appartenir à [0, 1[.")

        if self.is_signal_jepa:
            if self.freeze_strategy not in SIGNAL_JEPA_FREEZE_STRATEGIES:
                raise ValueError(
                    "Stratégie SignalJEPA inconnue : "
                    f"{self.freeze_strategy}."
                )
            if self.number_of_channels != 32:
                raise ValueError("SignalJEPA ASC attend exactement 32 canaux.")
            if self.number_of_timepoints != 1280:
                raise ValueError(
                    "SignalJEPA ASC attend 1280 points après "
                    "rééchantillonnage à 128 Hz."
                )
            if self.sampling_frequency != 128.0:
                raise ValueError("SignalJEPA ASC attend une fréquence de 128 Hz.")
            if (
                self.model_name == "signal_jepa_scratch"
                and self.freeze_strategy != "full_finetuning"
            ):
                raise ValueError(
                    "Geler un encodeur initialisé aléatoirement ne constitue "
                    "pas une expérience pertinente. signal_jepa_scratch doit "
                    "utiliser full_finetuning."
                )
        elif self.freeze_strategy != "full_finetuning":
            raise ValueError(
                "Le gel de couches n'est défini ici que pour SignalJEPA."
            )

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

        optimization_name = (
            f"lr_{self.learning_rate:g}"
            f"_batch_{self.batch_size}"
            f"_epochs_{self.number_of_epochs}"
            f"_patience_{self.early_stopping_patience}"
            f"_seed_{self.random_seed}"
        )

        if self.is_signal_jepa:
            return (
                f"{self.dataset_version}_{self.model_name}"
                f"_{self.freeze_strategy}_{self.preprocessing_name}"
                f"_{optimization_name}"
            )

        return (
            f"{self.dataset_version}_{self.model_name}"
            f"_hidden_{self.hidden_layer_size}"
            f"_dropout_{self.dropout_rate}"
            f"_{self.preprocessing_name}"
            f"_{optimization_name}"
        )
