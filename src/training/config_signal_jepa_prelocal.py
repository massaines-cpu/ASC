"""Configuration de l'expérience SignalJEPA PreLocal.

Les paramètres sont écrits directement en haut du fichier, conformément au
mode de lancement souhaité : il suffit de dupliquer ce fichier de lancement
ou de changer une constante avant d'exécuter l'expérience depuis PyCharm.
"""

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_MODEL_VARIANTS = {"scratch", "pretrained"}
SUPPORTED_FREEZE_STRATEGIES = {
    "full_finetuning",
    "classifier_only",
}
SUPPORTED_DEVICE_NAMES = {"auto", "cpu", "mps", "cuda"}


@dataclass(frozen=True)
class SignalJEPAPreLocalConfig:
    """Regroupe et valide tous les choix enregistrés avec un run."""

    project_root: Path
    dataset_version: str
    model_variant: str
    freeze_strategy: str
    batch_size: int
    number_of_epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    early_stopping_min_delta: float
    random_seed: int
    device_name: str
    selected_folds: tuple[str, ...]
    pretrained_checkpoint: str = "braindecode/signal-jepa_without-chans"
    number_of_channels: int = 19
    number_of_timepoints: int = 256
    sampling_frequency: float = 128.0
    windows_per_participant: int = 5

    def __post_init__(self) -> None:
        if self.model_variant not in SUPPORTED_MODEL_VARIANTS:
            raise ValueError(
                f"Variante inconnue : {self.model_variant}. "
                f"Valeurs possibles : {sorted(SUPPORTED_MODEL_VARIANTS)}."
            )

        if self.freeze_strategy not in SUPPORTED_FREEZE_STRATEGIES:
            raise ValueError(
                f"Stratégie inconnue : {self.freeze_strategy}. "
                "Valeurs possibles : "
                f"{sorted(SUPPORTED_FREEZE_STRATEGIES)}."
            )

        if self.device_name not in SUPPORTED_DEVICE_NAMES:
            raise ValueError(
                f"Appareil inconnu : {self.device_name}. "
                f"Valeurs possibles : {sorted(SUPPORTED_DEVICE_NAMES)}."
            )

        if self.model_variant == "scratch" and self.freeze_strategy != (
            "full_finetuning"
        ):
            raise ValueError(
                "Un backbone aléatoire ne doit pas être gelé. La variante "
                "scratch doit utiliser full_finetuning."
            )

        if self.number_of_channels != 19:
            raise ValueError("Cette expérience PreLocal attend 19 canaux.")
        if self.number_of_timepoints != 256:
            raise ValueError("Cette expérience PreLocal attend 256 points.")
        if self.sampling_frequency != 128.0:
            raise ValueError("Cette expérience PreLocal attend 128 Hz.")
        if self.windows_per_participant != 5:
            raise ValueError(
                "Un signal ASC de 10 s doit produire cinq fenêtres de 2 s."
            )

        if self.batch_size <= 0:
            raise ValueError("batch_size doit être strictement positif.")
        if self.number_of_epochs <= 0:
            raise ValueError("number_of_epochs doit être strictement positif.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate doit être strictement positif.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay doit être positif ou nul.")
        if self.early_stopping_patience <= 0:
            raise ValueError(
                "early_stopping_patience doit être strictement positif."
            )
        if self.early_stopping_min_delta < 0:
            raise ValueError(
                "early_stopping_min_delta doit être positif ou nul."
            )
        if not self.selected_folds:
            raise ValueError("selected_folds ne doit pas être vide.")

    @property
    def uses_pretrained_weights(self) -> bool:
        return self.model_variant == "pretrained"

    @property
    def dataset_root(self) -> Path:
        return self.project_root / "data" / self.dataset_version

    @property
    def experiment_name(self) -> str:
        """Construit un nom unique pour éviter l'écrasement des résultats."""

        return (
            f"{self.dataset_version}"
            f"_signal_jepa_prelocal_{self.model_variant}"
            f"_{self.freeze_strategy}"
            "_19ch_128hz_2s_microvolts_no_zscore"
            f"_lr_{self.learning_rate:g}"
            f"_batch_{self.batch_size}"
            f"_epochs_{self.number_of_epochs}"
            f"_patience_{self.early_stopping_patience}"
            f"_seed_{self.random_seed}"
        )
