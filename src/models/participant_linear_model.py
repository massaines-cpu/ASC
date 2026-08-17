"""Modèle linéaire de référence pour la classification EEG YO/YF.

Ce module contient uniquement l'architecture du modèle. La lecture des
métadonnées, la création des DataLoaders et la validation LODO restent
dans les modules ``dataset`` et ``training`` afin de ne pas exécuter de
logique d'entraînement lors d'un simple import.
"""

import torch
from torch import nn


class SimpleParticipantClassifier(nn.Module):
    """Classifieur linéaire recevant un EEG par participant.

    Entrée
    ------
    Tenseur de forme ``[batch_size, 32, 5120]``.

    Sortie
    ------
    Deux logits de forme ``[batch_size, 2]`` :

    - indice 0 : score de la classe YO ;
    - indice 1 : score de la classe YF.

    Le modèle ne contient volontairement aucune activation de sortie.
    Les logits sont transmis à ``CrossEntropyLoss`` pendant
    l'entraînement. Les probabilités sont calculées avec ``softmax``
    uniquement pendant l'évaluation.
    """

    def __init__(
        self,
        number_of_channels: int = 32,
        number_of_timepoints: int = 5120,
        number_of_classes: int = 2,
    ) -> None:
        super().__init__()

        self.number_of_channels = number_of_channels
        self.number_of_timepoints = number_of_timepoints
        self.number_of_classes = number_of_classes

        # La mise à plat transforme chaque EEG de forme (32, 5120)
        # en un vecteur de 163840 valeurs. Aucune valeur n'est modifiée :
        # seule l'organisation du tenseur change.
        self.flatten = nn.Flatten()

        # Cette unique transformation apprend directement deux scores
        # depuis le signal complet. Elle constitue la baseline linéaire
        # utilisée pour mesurer l'apport des architectures non linéaires.
        self.classifier = nn.Linear(
            in_features=number_of_channels * number_of_timepoints,
            out_features=number_of_classes,
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """Calcule les deux logits YO/YF d'un batch EEG."""

        if eeg.ndim != 3:
            raise ValueError(
                "Le MLP linéaire attend une entrée de forme "
                "[batch, channels, time], mais a reçu "
                f"{tuple(eeg.shape)}."
            )

        if eeg.shape[1] != self.number_of_channels:
            raise ValueError(
                f"Le modèle attend {self.number_of_channels} canaux, "
                f"mais en a reçu {eeg.shape[1]}."
            )

        if eeg.shape[2] != self.number_of_timepoints:
            raise ValueError(
                f"Le modèle attend {self.number_of_timepoints} points, "
                f"mais en a reçu {eeg.shape[2]}."
            )

        flattened_eeg = self.flatten(eeg)
        logits = self.classifier(flattened_eeg)
        return logits


if __name__ == "__main__":
    # Ce test autonome ne dépend d'aucun DataLoader. Il est exécuté
    # uniquement lorsque ce fichier est lancé directement, jamais lors
    # de son import par CROSS_VALIDATION_CONFIG.py.
    fake_eeg = torch.randn(5, 32, 5120)
    model = SimpleParticipantClassifier()

    logits = model(fake_eeg)
    probabilities = torch.softmax(logits, dim=1)

    print("Forme EEG :", fake_eeg.shape)
    print("Forme logits :", logits.shape)
    print("Forme probabilités :", probabilities.shape)
    print("Somme des probabilités :", probabilities.sum(dim=1))

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print("Paramètres entraînables :", number_of_parameters)
