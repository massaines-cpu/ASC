from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    project_root: Path
    dataset_version: str = "data_final"
    model_name: str = "non_linear"
    hidden_layer_size: int = 64
    dropout_rate: float = 0.0
    batch_size: int = 5
    number_of_epochs: int = 100
    learning_rate: float = 1e-3
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    random_seed: int = 42

    @property
    def dataset_root(self) -> Path:
        return (
            self.project_root
            / "data"
            / self.dataset_version
        )

    @property
    def experiment_name(self) -> str:
        return (
            f"{self.dataset_version}_{self.model_name}"
            f"_hidden_{self.hidden_layer_size}"
            f"_dropout_{self.dropout_rate}"
            "_standardized"
        )