"""Modèle linéaire binaire pour la classification EEG YO/YF.

Convention :
- YO -> label 0
- YF -> label 1

Le modèle produit un seul logit correspondant à la classe YF.
"""

import torch
from torch import nn


class SimpleParticipantClassifier(nn.Module):
    """Classifieur linéaire binaire recevant un EEG par participant.

    Entrée
    ------
    Tenseur de forme [batch_size, 32, 5120].

    Sortie
    ------
    Un logit par participant, de forme [batch_size].

    Le logit est transformé en probabilité avec une fonction Sigmoid :

        P(YF) = sigmoid(logit)
        P(YO) = 1 - P(YF)
    """

    def __init__(
        self,
        number_of_channels: int = 32,
        number_of_timepoints: int = 5120,
    ) -> None:
        super().__init__()

        self.number_of_channels = number_of_channels
        self.number_of_timepoints = number_of_timepoints

        # Transforme chaque EEG [32, 5120] en un vecteur de
        # 32 × 5120 = 163840 valeurs.
        self.flatten = nn.Flatten()

        # Une seule sortie est nécessaire pour une classification
        # binaire formulée avec Sigmoid :
        #
        # logit négatif  -> probabilité YF inférieure à 0.5 -> YO
        # logit positif  -> probabilité YF supérieure à 0.5 -> YF
        self.classifier = nn.Linear(
            in_features=(
                number_of_channels * number_of_timepoints
            ),
            out_features=1,
        )

        # La Sigmoid convertit le logit en une probabilité
        # comprise entre 0 et 1.
        #
        # Elle sera utilisée uniquement pour l'évaluation.
        self.output_activation = nn.Sigmoid()

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """Calcule le logit associé à la classe YF."""

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

        # La couche retourne initialement [batch_size, 1].
        logits = self.classifier(flattened_eeg)

        # Suppression de la dernière dimension :
        # [batch_size, 1] devient [batch_size].
        logits = logits.squeeze(dim=1)

        return logits

    def predict_yf_probability(
        self,
        eeg: torch.Tensor,
    ) -> torch.Tensor:
        """Calcule la probabilité d'appartenir à la classe YF."""

        logits = self.forward(eeg)
        probability_yf = self.output_activation(logits)

        return probability_yf


if __name__ == "__main__":
    fake_eeg = torch.randn(5, 32, 5120)
    model = SimpleParticipantClassifier()

    logits = model(fake_eeg)
    probability_yf = model.predict_yf_probability(fake_eeg)
    probability_yo = 1.0 - probability_yf

    predictions = (probability_yf >= 0.5).long()

    print("Forme EEG :", fake_eeg.shape)
    print("Forme logits :", logits.shape)
    print("Probabilité YO :", probability_yo)
    print("Probabilité YF :", probability_yf)
    print("Classes prédites :", predictions)

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Paramètres entraînables :", number_of_parameters)