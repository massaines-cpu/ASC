import torch
from torch import nn


class EEGNet(nn.Module):
    """EEGNet simplifié pour une classification EEG binaire YO/YF.

    Entrée
    ------
    x :
        Tenseur de forme :

            [batch_size, 32, 5120]

    Sortie
    ------
    logits :
        Un logit brut par exemple :

            [batch_size]

    Convention
    ----------
    YO -> label 0
    YF -> label 1

    Pendant l'entraînement :

        BCEWithLogitsLoss(logits, labels.float())

    Pendant l'évaluation :

        probability_yf = sigmoid(logits)
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_samples: int = 5120,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.temporal_filters = temporal_filters
        self.depth_multiplier = depth_multiplier
        self.dropout_rate = dropout_rate

        # Après la convolution spatiale, chaque filtre temporel
        # produit depth_multiplier filtres spatiaux.
        spatial_filters = (
            temporal_filters * depth_multiplier
        )

        # ==========================================================
        # Bloc 1 : convolution temporelle
        # ==========================================================

        # L'entrée sera transformée de :
        #
        # [batch, electrodes, time]
        #
        # vers :
        #
        # [batch, 1, electrodes, time]
        #
        # Le noyau (1, 64) se déplace uniquement dans le temps.
        self.temporal_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=temporal_filters,
                kernel_size=(1, 64),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
        )

        # ==========================================================
        # Bloc 2 : convolution spatiale depthwise
        # ==========================================================

        # Le noyau couvre simultanément les 32 électrodes.
        # Ce bloc apprend des combinaisons spatiales entre les
        # différents canaux EEG.
        self.spatial_block = nn.Sequential(
            nn.Conv2d(
                in_channels=temporal_filters,
                out_channels=spatial_filters,
                kernel_size=(n_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 4),
            ),
            nn.Dropout(
                p=dropout_rate,
            ),
        )

        # ==========================================================
        # Bloc 3 : convolution séparable
        # ==========================================================

        self.separable_block = nn.Sequential(

            # Convolution depthwise :
            # chaque filtre est traité séparément dans le temps.
            nn.Conv2d(
                in_channels=spatial_filters,
                out_channels=spatial_filters,
                kernel_size=(1, 16),
                padding="same",
                groups=spatial_filters,
                bias=False,
            ),

            # Convolution pointwise :
            # les caractéristiques extraites sont combinées.
            nn.Conv2d(
                in_channels=spatial_filters,
                out_channels=spatial_filters,
                kernel_size=(1, 1),
                bias=False,
            ),

            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1, 8),
            ),

            nn.Dropout(
                p=dropout_rate,
            ),
        )

        # Réduit automatiquement la dimension spatiale et temporelle
        # à une valeur par filtre.
        #
        # Sortie :
        # [batch, spatial_filters, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1),
        )

        # Une seule sortie brute est nécessaire.
        #
        # Cette sortie représente le logit associé à la classe YF.
        self.classifier = nn.Linear(
            in_features=spatial_filters,
            out_features=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Calcule un logit YF pour chaque EEG du batch."""

        if x.ndim != 3:
            raise ValueError(
                "EEGNet attend une entrée de forme "
                "[batch, channels, time], mais a reçu "
                f"{tuple(x.shape)}."
            )

        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"EEGNet attend {self.n_channels} canaux, "
                f"mais en a reçu {x.shape[1]}."
            )

        if x.shape[2] != self.n_samples:
            raise ValueError(
                f"EEGNet attend {self.n_samples} points temporels, "
                f"mais en a reçu {x.shape[2]}."
            )

        # [batch, 32, 5120]
        # devient :
        # [batch, 1, 32, 5120]
        x = x.unsqueeze(dim=1)

        x = self.temporal_block(x)

        # Après le bloc spatial, la dimension des électrodes
        # est réduite à 1.
        x = self.spatial_block(x)

        x = self.separable_block(x)

        # [batch, spatial_filters, 1, 1]
        x = self.global_pool(x)

        # [batch, spatial_filters]
        x = torch.flatten(
            x,
            start_dim=1,
        )

        # [batch, 1]
        logits = self.classifier(x)

        # [batch, 1] devient [batch].
        #
        # squeeze(dim=1) conserve correctement la dimension du batch
        # même lorsque batch_size vaut 1.
        logits = logits.squeeze(dim=1)

        return logits

    def predict_yf_probability(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Calcule la probabilité d'appartenir à la classe YF."""

        logits = self.forward(x)
        probability_yf = torch.sigmoid(logits)

        return probability_yf


if __name__ == "__main__":
    model = EEGNet(
        n_channels=32,
        n_samples=5120,
    )

    fake_eeg = torch.randn(
        5,
        32,
        5120,
    )

    # Logits bruts utilisés par BCEWithLogitsLoss.
    logits = model(fake_eeg)

    # Probabilités utilisées pendant l'évaluation.
    probability_yf = torch.sigmoid(logits)
    probability_yo = 1.0 - probability_yf

    predictions = (
        probability_yf >= 0.5
    ).long()

    print("Forme entrée :", fake_eeg.shape)
    print("Forme logits :", logits.shape)
    print("Probabilité YO :", probability_yo)
    print("Probabilité YF :", probability_yf)
    print("Prédictions :", predictions)

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Paramètres entraînables :",
        number_of_parameters,
    )