"""SignalJEPA PreLocal adapté à la classification binaire ASC YO/YF.

Trois expériences sont possibles avec exactement la même architecture :

1. ``scratch`` + ``full_finetuning`` : tous les poids sont aléatoires ;
2. ``pretrained`` + ``classifier_only`` : le feature encoder pré-entraîné
   est gelé, tandis que ``spatial_conv`` et ``final_layer`` apprennent ;
3. ``pretrained`` + ``full_finetuning`` : tous les blocs s'adaptent à ASC.

La sortie contient un seul logit associé à YF. L'entraînement utilise donc
``BCEWithLogitsLoss`` et la probabilité est calculée avec Sigmoid.
"""

from typing import Literal

import torch
from braindecode.models import SignalJEPA_PreLocal
from torch import nn


ModelVariant = Literal["scratch", "pretrained"]
FreezeStrategy = Literal["full_finetuning", "classifier_only"]

DEFAULT_CHECKPOINT = "braindecode/signal-jepa_without-chans"
# Valeur par défaut historique : le montage réduit à 19 électrodes. Passer
# number_of_channels=32 à create_signal_jepa_prelocal pour le montage ASC
# complet — PreLocal ne réutilise pas de table d'embeddings pré-entraînée
# par canal (contrairement à Contextual), donc rien dans cette architecture
# ne dépend d'un nombre de canaux précis.
NUMBER_OF_CHANNELS = 19
NUMBER_OF_TIMEPOINTS = 256
SAMPLING_FREQUENCY = 128.0
NUMBER_OF_OUTPUTS = 1


def configure_freezing(
    model: SignalJEPA_PreLocal,
    freeze_strategy: FreezeStrategy,
) -> None:
    """Définit précisément les paramètres mis à jour par l'optimiseur."""

    if freeze_strategy == "full_finetuning":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    if freeze_strategy == "classifier_only":
        # Tout est d'abord gelé, puis les couches nouvelles sont réactivées.
        for parameter in model.parameters():
            parameter.requires_grad = False

        # Dans PreLocal, spatial_conv dépend du montage cible de 19 canaux :
        # ce bloc n'est pas fourni par le checkpoint auto-supervisé.
        for parameter in model.spatial_conv.parameters():
            parameter.requires_grad = True

        # La tête binaire YO/YF est également propre à la tâche ASC.
        for parameter in model.final_layer.parameters():
            parameter.requires_grad = True
        return

    raise ValueError(
        f"Stratégie de gel inconnue : {freeze_strategy}."
    )


def create_signal_jepa_prelocal(
    model_variant: ModelVariant,
    freeze_strategy: FreezeStrategy,
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    number_of_channels: int = NUMBER_OF_CHANNELS,
) -> nn.Module:
    """Crée un PreLocal neuf pour chaque fold LODO."""

    common_arguments = {
        "n_chans": number_of_channels,
        "n_times": NUMBER_OF_TIMEPOINTS,
        "n_outputs": NUMBER_OF_OUTPUTS,
    }

    if model_variant == "pretrained":
        model = SignalJEPA_PreLocal.from_pretrained(
            checkpoint_name,
            **common_arguments,
            # Le checkpoint contient le feature encoder, mais pas les couches
            # spécifiques au montage ASC ni la tête binaire YO/YF.
            strict=False,
        )
    elif model_variant == "scratch":
        model = SignalJEPA_PreLocal(
            **common_arguments,
            sfreq=SAMPLING_FREQUENCY,
            # Valeur identique au checkpoint et au tutoriel officiel.
            drop_prob=0.0,
        )
    else:
        raise ValueError(f"Variante de modèle inconnue : {model_variant}.")

    configure_freezing(model, freeze_strategy)

    # Ces attributs permettent aux boucles d'entraînement de distinguer le
    # backbone gelé sans dépendre du nom choisi dans le script de lancement.
    model.asc_model_family = "signal_jepa_prelocal"
    model.asc_model_variant = model_variant
    model.asc_freeze_strategy = freeze_strategy

    return model


def prepare_binary_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalise la sortie PreLocal de ``[batch, 1]`` vers ``[batch]``."""

    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits.squeeze(dim=1)

    raise ValueError(
        "SignalJEPA PreLocal doit produire un logit YF par fenêtre, "
        f"mais a produit {tuple(logits.shape)}."
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Retourne le nombre total puis le nombre de paramètres entraînables."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def print_trainable_parameters(model: nn.Module) -> None:
    """Affiche les poids réellement transmis à l'optimiseur."""

    print("Paramètres entraînables :")
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(f"  {parameter_name:<55} {tuple(parameter.shape)}")
