import torch
from torch import nn


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Exécute une epoch de train ou de validation."""

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
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)

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