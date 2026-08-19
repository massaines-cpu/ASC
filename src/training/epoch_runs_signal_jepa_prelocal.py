"""Boucles d'epoch dédiées à SignalJEPA PreLocal."""

import torch
from torch import nn

from src.models.signal_jepa_prelocal_model import prepare_binary_logits


def configure_training_mode(model: nn.Module) -> None:
    """Active uniquement les modules prévus par la stratégie de gel."""

    model.train()
    freeze_strategy = getattr(model, "asc_freeze_strategy", None)

    if freeze_strategy == "classifier_only":
        # Le backbone gelé reste en mode eval. C'est cohérent avec un linear
        # probe et évite qu'un éventuel Dropout modifie ses représentations.
        model.feature_encoder.eval()

        # Les deux blocs nouveaux doivent réellement apprendre.
        model.spatial_conv.train()
        model.final_layer.train()
        return

    if freeze_strategy != "full_finetuning":
        raise ValueError(
            f"Stratégie de gel inconnue : {freeze_strategy}."
        )


def run_training_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Entraîne sur les fenêtres et retourne loss/accuracy fenêtre."""

    if not isinstance(criterion, nn.BCEWithLogitsLoss):
        raise TypeError(
            "La sortie binaire PreLocal exige nn.BCEWithLogitsLoss."
        )

    configure_training_mode(model)

    total_loss = 0.0
    correct_predictions = 0
    total_windows = 0

    for eeg, labels, _sample_ids, _window_indices in loader:
        eeg = eeg.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = prepare_binary_logits(model(eeg))
        targets = labels.to(dtype=logits.dtype)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        probabilities_yf = torch.sigmoid(logits)
        predictions = (probabilities_yf >= 0.5).long()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct_predictions += (predictions == labels).sum().item()
        total_windows += batch_size

    if total_windows == 0:
        raise ValueError("Le DataLoader d'entraînement est vide.")

    return (
        total_loss / total_windows,
        correct_predictions / total_windows,
    )
