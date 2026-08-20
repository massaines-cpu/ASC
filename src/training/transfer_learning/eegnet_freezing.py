"""Niveaux de gel progressif pour l'EEGNet simplifié ASC."""

from __future__ import annotations

from torch import nn


FREEZE_STRATEGIES = (
    "full_finetuning",
    "freeze_temporal",
    "freeze_temporal_spatial",
    "classifier_only",
)


def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    """Active ou désactive les gradients de tous les poids d'un module."""

    for parameter in module.parameters():
        parameter.requires_grad = trainable


def require_eegnet_blocks(model: nn.Module) -> None:
    """Vérifie que les noms de blocs correspondent au modèle ASC."""

    required_names = {
        "temporal_block",
        "spatial_block",
        "separable_block",
        "global_pool",
        "classifier",
    }
    missing_names = {
        name for name in required_names if not hasattr(model, name)
    }
    if missing_names:
        raise AttributeError(
            "Le modèle ne possède pas les blocs EEGNet attendus : "
            + ", ".join(sorted(missing_names))
        )


def apply_freeze_strategy(model: nn.Module, strategy: str) -> dict[str, int]:
    """Applique une stratégie et retourne le nombre de paramètres.

    Les stratégies sont cumulatives :

    - full_finetuning : tous les poids transférés peuvent s'adapter ;
    - freeze_temporal : le premier bloc reste identique au checkpoint ;
    - freeze_temporal_spatial : les deux premiers blocs restent fixes ;
    - classifier_only : seul le nouveau classifieur YO/YF apprend.
    """

    require_eegnet_blocks(model)

    if strategy not in FREEZE_STRATEGIES:
        raise ValueError(
            f"Stratégie inconnue : {strategy}. "
            f"Valeurs possibles : {FREEZE_STRATEGIES}."
        )

    # Point de départ identique : tout est entraînable.
    set_module_trainable(model, True)

    if strategy == "freeze_temporal":
        set_module_trainable(model.temporal_block, False)
    elif strategy == "freeze_temporal_spatial":
        set_module_trainable(model.temporal_block, False)
        set_module_trainable(model.spatial_block, False)
    elif strategy == "classifier_only":
        set_module_trainable(model, False)
        set_module_trainable(model.classifier, True)

    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": total_parameters - trainable_parameters,
    }


def keep_frozen_modules_in_eval(model: nn.Module, strategy: str) -> None:
    """Empêche les BatchNorm et Dropout gelés de changer pendant le train.

    ``requires_grad=False`` ne suffit pas : après ``model.train()``, les
    statistiques de BatchNorm continueraient sinon à être mises à jour.
    """

    if strategy == "freeze_temporal":
        model.temporal_block.eval()
    elif strategy == "freeze_temporal_spatial":
        model.temporal_block.eval()
        model.spatial_block.eval()
    elif strategy == "classifier_only":
        model.eval()
        model.classifier.train()

