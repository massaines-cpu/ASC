"""Adaptation de l'EEGNet de Braindecode au pipeline ASC.

Le modèle reçoit les mêmes données que les autres architectures du projet :

    [batch_size, 32, 5120]

Il retourne deux logits :

    [batch_size, 2]

Classes :
    0 -> YO
    1 -> YF
"""

import torch
from torch import nn
from braindecode.models import EEGNet as BraindecodeEEGNetModel


class BraindecodeEEGNet(nn.Module):
    """Adapte EEGNet de Braindecode aux dimensions du dataset ASC.

    Paramètres
    ----------
    number_of_channels : int
        Nombre d'électrodes EEG.

    number_of_timepoints : int
        Nombre de points temporels par participant.

    number_of_classes : int
        Nombre de classes à prédire.

    dropout_rate : float
        Proportion d'activations désactivées pendant l'entraînement.
    """

    def __init__(
        self,
        number_of_channels: int = 32,
        number_of_timepoints: int = 5120,
        number_of_classes: int = 2,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        self.number_of_channels = number_of_channels
        self.number_of_timepoints = number_of_timepoints
        self.number_of_classes = number_of_classes

        # Les paramètres F1, D, F2 et les tailles de noyaux sont choisis
        # pour rester aussi proches que possible de l'EEGNet déjà utilisé
        # dans le projet. La principale variable expérimentale devient
        # ainsi l'implémentation Braindecode de l'architecture.
        self.model = BraindecodeEEGNetModel(
            n_chans=number_of_channels,
            n_outputs=number_of_classes,
            n_times=number_of_timepoints,

            # Nombre de filtres temporels du premier bloc.
            F1=8,

            # Chaque filtre temporel apprend deux filtres spatiaux.
            D=2,

            # Nombre de filtres après la convolution séparable.
            # Dans EEGNet, F2 est généralement égal à F1 × D.
            F2=16,

            # Taille de la première convolution temporelle.
            kernel_length=64,

            # Taille de la convolution temporelle depthwise du
            # bloc séparable.
            depthwise_kernel_length=16,

            # Même réduction temporelle que dans le modèle actuel.
            pool1_kernel_size=4,
            pool2_kernel_size=8,

            # Même dropout que dans l'EEGNet simplifié actuel.
            drop_prob=dropout_rate,

            # La moyenne est utilisée pendant les opérations de pooling.
            pool_mode="mean",

            # Braindecode détermine automatiquement la dimension
            # précédant la couche finale.
            final_conv_length="auto",
        )

    def forward(
        self,
        eeg: torch.Tensor,
    ) -> torch.Tensor:
        """Calcule les logits YO/YF pour un batch EEG.

        Paramètres
        ----------
        eeg : torch.Tensor
            Batch de forme [batch_size, 32, 5120].

        Retour
        ------
        torch.Tensor
            Deux logits par participant, de forme [batch_size, 2].
        """

        if eeg.ndim != 3:
            raise ValueError(
                "BraindecodeEEGNet attend un tenseur de forme "
                "[batch_size, channels, time], mais a reçu "
                f"{tuple(eeg.shape)}."
            )

        if eeg.shape[1] != self.number_of_channels:
            raise ValueError(
                f"Nombre de canaux incorrect : {eeg.shape[1]}. "
                f"Nombre attendu : {self.number_of_channels}."
            )

        if eeg.shape[2] != self.number_of_timepoints:
            raise ValueError(
                f"Nombre de points temporels incorrect : {eeg.shape[2]}. "
                f"Nombre attendu : {self.number_of_timepoints}."
            )

        logits = self.model(eeg)

        return logits


def count_trainable_parameters(model: nn.Module) -> int:
    """Compte uniquement les paramètres qui seront entraînés."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    # Ce test vérifie seulement que le modèle accepte les dimensions
    # du projet. Il n'est pas exécuté lorsque la classe est importée
    # par le pipeline d'entraînement.
    model = BraindecodeEEGNet(
        number_of_channels=32,
        number_of_timepoints=5120,
        number_of_classes=2,
        dropout_rate=0.5,
    )

    fake_eeg = torch.randn(
        5,
        32,
        5120,
    )

    with torch.no_grad():
        predictions = model(fake_eeg)

    print("Forme de l'entrée :", fake_eeg.shape)
    print("Forme de la sortie :", predictions.shape)
    print(
        "Paramètres entraînables :",
        count_trainable_parameters(model),
    )