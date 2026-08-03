import torch
from torch import nn


class EEGNet(nn.Module):
    """
    EEGNet simplifié pour une classification EEG binaire.

    Entrée :
        x : [batch_size, 32, 5120]

    Sortie :
        logits : [batch_size, 2]
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_samples: int = 5120,
        n_classes: int = 2,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        # Nombre de filtres après la convolution spatiale depthwise.
        spatial_filters = temporal_filters * depth_multiplier

        # Bloc 1 : convolution temporelle

        # L'entrée aura la forme :
        # [batch, 1, electrodes, time]
        #
        # Le noyau (1, 64) se déplace uniquement dans le temps.
        # Il apprend donc des motifs temporels communs aux électrodes.
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

        # Bloc 2 : convolution spatiale depthwise

        # Le noyau couvre les 32 électrodes en une seule fois.
        # Il apprend des combinaisons spatiales entre les canaux EEG.
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
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Bloc 3 : convolution séparable

        # Première convolution : profondeur par profondeur.
        # Deuxième convolution : combinaison des filtres.
        self.separable_block = nn.Sequential(
            nn.Conv2d(
                in_channels=spatial_filters,
                out_channels=spatial_filters,
                kernel_size=(1, 16),
                padding="same",
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=spatial_filters,
                out_channels=spatial_filters,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Réduit automatiquement la dimension temporelle à 1.
        # Cela évite de calculer manuellement la taille après les poolings.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Couche finale de classification.
        self.classifier = nn.Linear(
            in_features=spatial_filters,
            out_features=n_classes,
        )

    def forward(self, x):
        """
        Paramètre
        ---------
        x : tenseur [batch, channels, time]

        Retour
        ------
        logits : tenseur [batch, n_classes]
        """

        if x.ndim != 3:
            raise ValueError(
                "EEGNet attend une entrée de forme "
                f"[batch, channels, time], mais a reçu {tuple(x.shape)}."
            )

        # [batch, 32, 5120]
        # devient :
        # [batch, 1, 32, 5120]
        x = x.unsqueeze(1)

        x = self.temporal_block(x)

        # Après ce bloc, la dimension des électrodes devient 1.
        x = self.spatial_block(x)

        x = self.separable_block(x)

        # [batch, spatial_filters, 1, 1]
        x = self.global_pool(x)

        # [batch, spatial_filters]
        x = torch.flatten(x, start_dim=1)

        # [batch, 2]
        logits = self.classifier(x)

        return logits
#------------------------------------------- teest
model = EEGNet(
    n_channels=32,
    n_samples=5120,
    n_classes=2,
)

fake_eeg = torch.randn(5, 32, 5120)

output = model(fake_eeg)

print("Forme entrée :", fake_eeg.shape)
print("Forme sortie :", output.shape)

number_of_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Paramètres entraînables :", number_of_parameters)