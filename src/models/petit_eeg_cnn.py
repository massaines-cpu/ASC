import torch
from torch import nn


class Small_CNN_EEG(nn.Module):
    """Petit réseau convolutionnel 1D pour classifier les EEG YO/YF.

    Entrée
    ------
    eeg :
        Tenseur de forme :

            [batch_size, 32, 5120]

    Sortie
    ------
    logits :
        Un logit brut par exemple, de forme :

            [batch_size]

    Convention
    ----------
    YO -> label 0
    YF -> label 1

    Pendant l'entraînement, le logit est transmis à :

        nn.BCEWithLogitsLoss()

    Pendant l'évaluation, la probabilité de YF est obtenue avec :

        torch.sigmoid(logits)
    """

    def __init__(
        self,
        number_of_eeg_channels: int = 32,
    ):
        super().__init__()

        # ----------------------------------------------------------
        # Partie convolutionnelle
        # ----------------------------------------------------------
        self.features = nn.Sequential(

            # La convolution se déplace dans le temps et transforme
            # les 32 canaux EEG en 16 cartes de caractéristiques.
            nn.Conv1d(
                in_channels=number_of_eeg_channels,
                out_channels=16,
                kernel_size=25,
                stride=2,
                padding=12,
            ),

            nn.BatchNorm1d(16),
            nn.ELU(),

            # Réduction de la dimension temporelle.
            nn.MaxPool1d(
                kernel_size=4,
                stride=4,
            ),

            # Deuxième niveau d'extraction temporelle :
            # 16 cartes de caractéristiques deviennent 32 cartes.
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=15,
                stride=1,
                padding=7,
            ),

            nn.BatchNorm1d(32),
            nn.ELU(),

            nn.MaxPool1d(
                kernel_size=4,
                stride=4,
            ),

            # Troisième niveau d'extraction temporelle :
            # 32 cartes deviennent 64 cartes.
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
            ),

            nn.BatchNorm1d(64),
            nn.ELU(),

            # Chaque carte de caractéristiques est résumée par
            # une seule valeur.
            #
            # [batch, 64, longueur_temporelle]
            # devient :
            # [batch, 64, 1]
            nn.AdaptiveAvgPool1d(1),
        )

        # ----------------------------------------------------------
        # Partie classification
        # ----------------------------------------------------------
        self.classifier = nn.Sequential(

            # [batch, 64, 1] devient [batch, 64].
            nn.Flatten(),

            # Désactive aléatoirement 50 % des caractéristiques
            # pendant l'entraînement pour limiter le surapprentissage.
            nn.Dropout(p=0.5),

            # Une seule sortie brute est nécessaire :
            # elle représente le logit associé à la classe YF.
            nn.Linear(
                in_features=64,
                out_features=1,
            ),
        )

    def forward(
        self,
        eeg: torch.Tensor,
    ) -> torch.Tensor:
        """Calcule le logit brut associé à la classe YF."""

        if eeg.ndim != 3:
            raise ValueError(
                "Small_CNN_EEG attend une entrée de forme "
                "[batch, channels, time], mais a reçu "
                f"{tuple(eeg.shape)}."
            )

        features = self.features(eeg)

        # Forme initiale : [batch_size, 1]
        logits = self.classifier(features)

        # Forme finale : [batch_size]
        #
        # squeeze(dim=1) est utilisé plutôt que squeeze() afin de
        # conserver la dimension du batch lorsque batch_size vaut 1.
        logits = logits.squeeze(dim=1)

        return logits

    def predict_yf_probability(
        self,
        eeg: torch.Tensor,
    ) -> torch.Tensor:
        """Calcule la probabilité d'appartenir à la classe YF."""

        logits = self.forward(eeg)
        probability_yf = torch.sigmoid(logits)

        return probability_yf


if __name__ == "__main__":
    model = Small_CNN_EEG()

    fake_eeg = torch.randn(
        5,
        32,
        5120,
    )

    # Sortie brute utilisée par BCEWithLogitsLoss.
    logits = model(fake_eeg)

    # Sigmoid utilisée uniquement pour obtenir les probabilités.
    probability_yf = torch.sigmoid(logits)
    probability_yo = 1.0 - probability_yf

    # Seuil de décision binaire.
    predictions = (
        probability_yf >= 0.5
    ).long()

    print("EEG :", fake_eeg.shape)
    print("Logits :", logits.shape)
    print("Probabilité YO :", probability_yo)
    print("Probabilité YF :", probability_yf)
    print("Prédictions :", predictions)