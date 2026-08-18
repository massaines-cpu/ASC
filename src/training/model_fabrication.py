"""Création des architectures entraînées from scratch dans ASC."""

from torch import nn

from src.models.participant_linear_model import (
    SimpleParticipantClassifier,
)
from src.models.participant_non_linear_MLP_dropout import (
    NonLinearParticipantMLP,
)
from src.models.petit_eeg_cnn import Small_CNN_EEG
from src.models.eegNET_model import EEGNet


def create_model(
    model_name: str,
    hidden_layer_size: int,
    dropout_rate: float,
) -> nn.Module:
    """Crée un modèle neuf avec des poids initialisés aléatoirement.

    Toutes les architectures retournent un seul logit associé à YF.
    La fonction de loss commune est donc ``BCEWithLogitsLoss``.
    """

    if model_name == "linear":
        return SimpleParticipantClassifier()

    if model_name == "non_linear":
        return NonLinearParticipantMLP(
            number_of_channels=32,
            number_of_timepoints=5120,
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
        )

    if model_name == "small_cnn":
        return Small_CNN_EEG(
            number_of_eeg_channels=32,
        )

    if model_name == "eegnet":
        return EEGNet(
            n_channels=32,
            n_samples=5120,
        )

    raise ValueError(f"Modèle inconnu : {model_name}")
