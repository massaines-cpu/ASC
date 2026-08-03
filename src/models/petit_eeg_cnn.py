import torch
from torch import nn


class Small_CNN_EEG(nn.Module):
    """Petit réseau convolutionnel 1D pour classifier des segments EEG.

    Entrée attendue
    ----------------
    eeg : torch.Tensor
        Tensor de forme :

            [batch_size, number_of_channels, number_of_time_points]

        Dans notre cas :

            [batch_size, 32, 5120]

    Sortie
    ------
    logits : torch.Tensor
        Tensor de forme :

            [batch_size, 2]

        Une valeur par classe :
        - classe 0 : YO
        - classe 1 : YF
    """

    def __init__(
        self,
        number_of_eeg_channels: int = 32,
        number_of_classes: int = 2,
    ):
        super().__init__()

        # ----------------------------------------------------------
        # Partie convolutionnelle
        # ----------------------------------------------------------
        #
        # Conv1d attend une entrée de forme :
        #
        #     [batch, channels, time]
        #
        # Ici les 32 électrodes sont considérées comme les canaux
        # d'entrée, et la convolution se déplace dans le temps.
        self.features = nn.Sequential(

            # Première convolution :
            # 32 canaux EEG -> 16 cartes de caractéristiques.
            nn.Conv1d(
                in_channels=number_of_eeg_channels,
                out_channels=16,
                kernel_size=25,
                stride=2,
                padding=12,
            ),

            # Stabilise l'apprentissage des activations.
            nn.BatchNorm1d(16),

            # Activation souvent utilisée en EEG.
            nn.ELU(),

            # Réduit la longueur temporelle.
            nn.MaxPool1d(
                kernel_size=4,
                stride=4,
            ),

            # Deuxième convolution :
            # 16 cartes -> 32 cartes.
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

            # Troisième convolution :
            # 32 cartes -> 64 cartes.
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
            ),

            nn.BatchNorm1d(64),
            nn.ELU(),

            # Réduit toute la dimension temporelle à une seule valeur
            # par carte de caractéristiques.

            # Avant :
            #     [batch, 64, longueur_temporelle]

            # Après :
            #     [batch, 64, 1]
            nn.AdaptiveAvgPool1d(1),
        )

        # ----------------------------------------------------------
        # Partie classification
        # ----------------------------------------------------------
        self.classifier = nn.Sequential(

            # [batch, 64, 1] -> [batch, 64]
            nn.Flatten(),

            # Régularisation pour limiter le surapprentissage.
            nn.Dropout(p=0.5),

            # 64 caractéristiques -> 2 logits.
            nn.Linear(
                in_features=64,
                out_features=number_of_classes,
            ),
        )

    def forward(
        self,
        eeg: torch.Tensor,
    ) -> torch.Tensor:
        """Effectue une passe avant dans le réseau."""

        features = self.features(eeg)
        logits = self.classifier(features)

        return logits

if __name__ == "__main__":

    model = Small_CNN_EEG()

    test = torch.randn(
        5,
        32,
        5120,
    )

    predictions = model(test)

    print("EEG :", test.shape)
    print("Prédictions :", predictions.shape)