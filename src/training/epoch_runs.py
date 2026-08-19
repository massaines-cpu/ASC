"""Exécution d'une epoch binaire d'entraînement ou de validation."""

import torch
from torch import nn


def _prepare_binary_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalise une sortie binaire vers la forme ``[batch_size]``."""

    if logits.ndim == 1:
        return logits

    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits.squeeze(dim=1)

    raise ValueError(
        "BCEWithLogitsLoss attend un seul logit par exemple, "
        f"mais le modèle a produit {tuple(logits.shape)}."
    )


def _configure_model_mode(
    model: nn.Module,
    training: bool,
    freeze_strategy: str | None,
    device: torch.device,
) -> None:
    """Configure correctement les modes train/eval du modèle.

    ``requires_grad=False`` gèle les poids, mais ``model.train()`` réactive
    tout de même le Dropout des modules gelés. Pour le linear probe
    SignalJEPA, le backbone doit donc rester en mode ``eval`` tandis que la
    nouvelle tête YO/YF reste en mode ``train``.

    Cette distinction garantit aussi que les caractéristiques produites par
    l'encodeur gelé restent déterministes pendant l'apprentissage de la tête.
    """

    if not training:
        model.eval()
        return

    model.train()

    # Les autres architectures ASC suivent simplement model.train().
    if getattr(model, "asc_model_family", None) != "signal_jepa":
        return

    effective_freeze_strategy = (
        freeze_strategy
        if freeze_strategy is not None
        else getattr(model, "asc_freeze_strategy", None)
    )

    if effective_freeze_strategy == "full_finetuning":
        if device.type == "mps":
            # PyTorch/MPS ne prend pas en charge le Dropout de
            # scaled_dot_product_attention. Le mode eval le désactive,
            # mais ne bloque pas les gradients : les poids de l'encodeur
            # restent donc entraînables pendant le fine-tuning.
            model.transformer.eval()
        return

    if effective_freeze_strategy != "classifier_only":
        raise ValueError(
            "Stratégie SignalJEPA inconnue : "
            f"{effective_freeze_strategy}."
        )

    frozen_backbone_names = (
        "feature_encoder",
        "pos_encoder",
        "transformer",
    )

    for module_name in frozen_backbone_names:
        frozen_module = getattr(model, module_name, None)
        if frozen_module is not None:
            frozen_module.eval()

    # La tête est la seule partie qui doit rester en mode entraînement.
    final_layer = getattr(model, "final_layer", None)
    if final_layer is None:
        raise AttributeError(
            "La stratégie classifier_only attend une couche final_layer."
        )
    final_layer.train()


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
    freeze_strategy: str | None = None,
) -> tuple[float, float]:
    """Exécute une epoch sur CPU, CUDA ou MPS.

    Le déplacement des données est centralisé ici afin que tous les modèles
    utilisent réellement le même appareil que le modèle. Sans ces lignes,
    PyTorch entraîne silencieusement sur CPU même lorsque MPS est disponible.
    """

    if not isinstance(criterion, nn.BCEWithLogitsLoss):
        raise TypeError(
            "Les modèles ASC à une sortie doivent utiliser "
            "nn.BCEWithLogitsLoss."
        )

    if device is None:
        # Cette valeur par défaut préserve la compatibilité avec les anciens
        # appels à run_epoch qui ne transmettaient pas encore ``device``.
        # Le modèle indique alors lui-même où doivent être placés les batches.
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    training = optimizer is not None
    _configure_model_mode(
        model=model,
        training=training,
        freeze_strategy=freeze_strategy,
        device=device,
    )

    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    gradient_context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with gradient_context:
        for eeg, labels in loader:
            eeg = eeg.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(eeg)
            binary_logits = _prepare_binary_logits(logits)

            # BCEWithLogitsLoss attend des cibles réelles 0.0/1.0 de
            # même forme que les logits, contrairement à CrossEntropyLoss
            # qui recevait auparavant des labels entiers.
            binary_targets = labels.to(dtype=binary_logits.dtype)
            loss = criterion(binary_logits, binary_targets)

            # Sigmoid transforme le logit associé à YF en probabilité.
            # Une probabilité d'au moins 0,5 donne la classe YF (1).
            probability_yf = torch.sigmoid(binary_logits)
            predictions = (probability_yf >= 0.5).long()

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct_predictions += (
                predictions == labels
            ).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Le DataLoader est vide.")

    return (
        total_loss / total_examples,
        correct_predictions / total_examples,
    )
