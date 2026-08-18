import torch
from torch import nn


class NonLinearParticipantMLP(nn.Module):
    """
    MLP non linéaire binaire pour la classification EEG YO/YF.

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
    Linear [hidden_layer_size → 1]
            ↓
    Un logit pour la classe YF

    Convention
    ----------
    YO -> label 0
    YF -> label 1

    Pendant l'entraînement :
        le modèle retourne un logit brut ;
        la loss utilisée est BCEWithLogitsLoss.

    Pendant l'évaluation :
        Sigmoid transforme ce logit en probabilité YF.
    """

    def __init__(
        self,
        number_of_channels: int = 32,
        number_of_timepoints: int = 5120,
        hidden_layer_size: int = 128,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        if hidden_layer_size <= 0:
            raise ValueError(
                "hidden_layer_size doit être strictement positif."
            )

        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                "dropout_rate doit être compris entre "
                "0 inclus et 1 exclu."
            )

        self.number_of_channels = number_of_channels
        self.number_of_timepoints = number_of_timepoints
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate

        flattened_input_size = (
            number_of_channels * number_of_timepoints
        )

        self.flatten = nn.Flatten()

        self.hidden_layer = nn.Linear(
            in_features=flattened_input_size,
            out_features=hidden_layer_size,
        )

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(
            p=dropout_rate,
        )

        # Une seule sortie binaire :
        # logit positif  -> plutôt YF
        # logit négatif  -> plutôt YO
        self.classifier = nn.Linear(
            in_features=hidden_layer_size,
            out_features=1,
        )

        # Cette activation est utilisée pour convertir le logit
        # en probabilité pendant l'évaluation.
        self.sigmoid = nn.Sigmoid()

    def forward(self, eeg):
        """
        Retourne un logit brut par exemple.

        Entrée :
            [batch_size, 32, 5120]

        Sortie :
            [batch_size]
        """

        eeg = self.flatten(eeg)

        hidden_representation = self.hidden_layer(eeg)
        hidden_representation = self.activation(
            hidden_representation
        )
        hidden_representation = self.dropout(
            hidden_representation
        )

        logits = self.classifier(
            hidden_representation
        )

        # [batch_size, 1] devient [batch_size].
        return logits.squeeze(dim=1)

    def predict_yf_probability(self, eeg):
        """Retourne la probabilité d'appartenir à YF."""

        logits = self.forward(eeg)
        probability_yf = self.sigmoid(logits)

        return probability_yf


if __name__ == "__main__":
    fake_eeg = torch.randn(5, 32, 5120)

    model = NonLinearParticipantMLP(
        hidden_layer_size=32,
        dropout_rate=0.0,
    )

    logits = model(fake_eeg)
    probability_yf = model.predict_yf_probability(fake_eeg)
    probability_yo = 1.0 - probability_yf

    predictions = (
        probability_yf >= 0.5
    ).long()

    print("Entrée :", fake_eeg.shape)
    print("Logits :", logits.shape)
    print("Probabilité YO :", probability_yo)
    print("Probabilité YF :", probability_yf)
    print("Prédictions :", predictions)