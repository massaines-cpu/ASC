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


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Exécute une epoch avec un logit YF et une décision à 0,5."""

    if not isinstance(criterion, nn.BCEWithLogitsLoss):
        raise TypeError(
            "Les modèles ASC à une sortie doivent utiliser "
            "nn.BCEWithLogitsLoss."
        )

    training = optimizer is not None
    model.train() if training else model.eval()

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
            if training:
                optimizer.zero_grad()

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
