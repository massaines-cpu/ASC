import torch
from torch import nn


class NonLinearParticipantMLP(nn.Module):
    """
    MLP non linéaire configurable pour la classification EEG YO/YF.

    Architecture
    ------------
    EEG [batch, 32, 5120]
            ↓
    Flatten
            ↓
    Linear [163840 → hidden_layer_size]
            ↓
    ReLU
            ↓
    Dropout
            ↓
    Linear [hidden_layer_size → 2]
            ↓
    Deux logits : YO et YF

    Important
    ---------
    Le réseau retourne des logits bruts afin de rester compatible
    avec CrossEntropyLoss.

    Les probabilités sont calculées pendant l'évaluation avec :

        torch.softmax(logits, dim=1)
    """

    def __init__(
        self,
        number_of_channels: int = 32,
        number_of_timepoints: int = 5120,
        hidden_layer_size: int = 128,
        dropout_rate: float = 0.0,
        number_of_classes: int = 2,
    ):
        super().__init__()

        if hidden_layer_size <= 0:
            raise ValueError(
                "hidden_layer_size doit être strictement positif."
            )

        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                "dropout_rate doit être compris entre 0 inclus "
                "et 1 exclu."
            )

        self.number_of_channels = number_of_channels
        self.number_of_timepoints = number_of_timepoints
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.number_of_classes = number_of_classes

        flattened_input_size = (
            number_of_channels * number_of_timepoints
        )

        self.flatten = nn.Flatten()

        # Cette couche compresse les 163840 valeurs EEG dans une
        # représentation de taille 128, 64 ou 32 selon l'expérience.
        self.hidden_layer = nn.Linear(
            in_features=flattened_input_size,
            out_features=hidden_layer_size,
        )

        # ReLU permet d'apprendre des relations non linéaires.
        self.activation = nn.ReLU()

        # Le Dropout désactive aléatoirement une fraction des neurones
        # pendant l'entraînement. L'hypothèse testée est qu'il peut
        # limiter le surapprentissage du MLP, qui contient plusieurs
        # millions de paramètres.
        #
        # Avec dropout_rate=0.0, aucune activation n'est supprimée.
        self.dropout = nn.Dropout(p=dropout_rate)

        # La sortie contient deux logits bruts.
        # CrossEntropyLoss attend précisément ce format.
        self.classifier = nn.Linear(
            in_features=hidden_layer_size,
            out_features=number_of_classes,
        )

    def forward(self, eeg):
        """
        Calcule les logits YO/YF.

        Paramètre
        ---------
        eeg :
            Tenseur [batch_size, 32, 5120].

        Retour
        ------
        logits :
            Tenseur [batch_size, 2].
        """

        eeg = self.flatten(eeg)

        hidden_representation = self.hidden_layer(eeg)
        hidden_representation = self.activation(
            hidden_representation
        )
        hidden_representation = self.dropout(
            hidden_representation
        )

        logits = self.classifier(hidden_representation)

        return logits


if __name__ == "__main__":
    # Test indépendant du dataset.
    fake_eeg = torch.randn(5, 32, 5120)

    model = NonLinearParticipantMLP(
        hidden_layer_size=128,
        dropout_rate=0.0,
    )

    logits = model(fake_eeg)
    probabilities = torch.softmax(logits, dim=1)

    print("Entrée :", fake_eeg.shape)
    print("Logits :", logits.shape)
    print("Probabilités :", probabilities.shape)
    print(
        "Somme des probabilités :",
        probabilities.sum(dim=1),
    )

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Paramètres entraînables :",
        f"{number_of_parameters:,}",
    )