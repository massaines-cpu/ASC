"""Création centralisée des architectures comparées dans ASC."""

from torch import nn

from src.models.participant_linear_model import (
    SimpleParticipantClassifier,
)
from src.models.participant_non_linear_MLP_dropout import (
    NonLinearParticipantMLP,
)
from src.models.petit_eeg_cnn import Small_CNN_EEG
from src.models.eegNET_model import EEGNet
from src.models.signal_jepa_model import create_signal_jepa_model


def create_model(
    model_name: str,
    hidden_layer_size: int,
    dropout_rate: float,
    number_of_channels: int = 32,
    number_of_timepoints: int = 5120,
    sampling_frequency: float = 512.0,
    pretrained_checkpoint: str = "braindecode/signal-jepa",
    freeze_strategy: str = "full_finetuning",
) -> nn.Module:
    """Crée l'architecture et l'initialisation demandées.

    Toutes les architectures retournent un seul logit YF, éventuellement de
    forme ``[batch, 1]``. ``epoch_runs.py`` normalise ensuite la forme avant
    de transmettre ce logit à ``BCEWithLogitsLoss``.
    """

    if model_name == "linear":
        return SimpleParticipantClassifier(
            number_of_channels=number_of_channels,
            number_of_timepoints=number_of_timepoints,
        )

    if model_name == "non_linear":
        return NonLinearParticipantMLP(
            number_of_channels=number_of_channels,
            number_of_timepoints=number_of_timepoints,
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
        )

    if model_name == "small_cnn":
        return Small_CNN_EEG(
            number_of_eeg_channels=number_of_channels,
        )

    if model_name == "eegnet":
        return EEGNet(
            n_channels=number_of_channels,
            n_samples=number_of_timepoints,
        )

    if model_name in {
        "signal_jepa_scratch",
        "signal_jepa_pretrained",
    }:
        return create_signal_jepa_model(
            pretrained=(model_name == "signal_jepa_pretrained"),
            number_of_timepoints=number_of_timepoints,
            sampling_frequency=sampling_frequency,
            checkpoint_name=pretrained_checkpoint,
            freeze_strategy=freeze_strategy,
        )

    raise ValueError(f"Modèle inconnu : {model_name}")
