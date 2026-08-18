"""Exécution d'une epoch binaire d'entraînement ou de validation."""

import torch
from torch import nn


# ======================================================================
# 1. Normalisation de la sortie binaire
# ======================================================================

def _prepare_binary_logits(
    logits: torch.Tensor,
) -> torch.Tensor:
    """Normalise une sortie binaire vers la forme [batch_size].

    SignalJEPA retourne normalement :

        [batch_size, 1]

    tandis que BCEWithLogitsLoss peut travailler ici plus simplement avec :

        [batch_size]
    """

    if logits.ndim == 1:
        return logits

    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits.squeeze(dim=1)

    raise ValueError(
        "BCEWithLogitsLoss attend un seul logit par exemple, "
        f"mais le modèle a produit {tuple(logits.shape)}."
    )


# ======================================================================
# 2. Gestion correcte de train() / eval() avec le freezing
# ======================================================================

def _configure_model_mode(
    model: nn.Module,
    training: bool,
    freeze_strategy: str | None,
) -> None:
    """Configure train/eval selon la stratégie de transfert learning.

    Important
    ---------
    requires_grad=False :
        empêche la mise à jour des poids.

    eval() :
        désactive notamment le Dropout.

    Ce sont donc deux choses différentes.

    Pour les blocs gelés de SignalJEPA, on souhaite ici conserver
    leurs représentations pré-entraînées de manière déterministe.
    """

    # --------------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------------
    if not training:
        model.eval()
        return

    # --------------------------------------------------------------
    # ENTRAÎNEMENT
    # --------------------------------------------------------------
    # Par défaut, tout le modèle passe en train().
    model.train()

    # Si aucune stratégie particulière n'est demandée,
    # on laisse tout le modèle en mode train.
    if freeze_strategy is None:
        return

    if freeze_strategy == "full_finetuning":
        # Tout est entraînable.
        return

    # --------------------------------------------------------------
    # 1. Feature encoder gelé
    # --------------------------------------------------------------
    if freeze_strategy == "freeze_feature_encoder":

        model.feature_encoder.eval()

        return

    # --------------------------------------------------------------
    # 2. Feature encoder + position encoder gelés
    # --------------------------------------------------------------
    if freeze_strategy == "freeze_feature_and_position":

        model.feature_encoder.eval()
        model.pos_encoder.eval()

        return

    # --------------------------------------------------------------
    # 3. Feature encoder + position encoder
    #    + 2 premières couches Transformer gelées
    # --------------------------------------------------------------
    if (
        freeze_strategy
        == "freeze_feature_position_first_2_transformer"
    ):

        model.feature_encoder.eval()
        model.pos_encoder.eval()

        transformer_layers = (
            model.transformer.encoder.layers
        )

        for layer in transformer_layers[:2]:
            layer.eval()

        return

    # --------------------------------------------------------------
    # 4. Seule la tête YO/YF est entraînée
    # --------------------------------------------------------------
    if freeze_strategy == "classifier_only":

        # Backbone complètement en évaluation.
        model.feature_encoder.eval()
        model.pos_encoder.eval()
        model.transformer.eval()

        # Nouvelle tête de classification en entraînement.
        model.final_layer.train()

        return

    raise ValueError(
        f"Stratégie de freezing inconnue : {freeze_strategy}"
    )


# ======================================================================
# 3. Une epoch
# ======================================================================

def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
    freeze_strategy: str | None = None,
) -> tuple[float, float]:
    """Exécute une epoch binaire d'entraînement ou de validation.

    Parameters
    ----------
    model
        Modèle PyTorch.

    loader
        DataLoader train ou validation.

    criterion
        Doit être BCEWithLogitsLoss.

    optimizer
        Si None :
            validation.

        Sinon :
            entraînement.

    device
        cpu, cuda ou mps.

    freeze_strategy
        Stratégie utilisée pour le transfert learning SignalJEPA.

        Exemples :
        - full_finetuning
        - freeze_feature_encoder
        - freeze_feature_and_position
        - freeze_feature_position_first_2_transformer
        - classifier_only
    """

#verif loss
    if not isinstance(
        criterion,
        nn.BCEWithLogitsLoss,
    ):
        raise TypeError(
            "Les modèles ASC à une sortie doivent utiliser "
            "nn.BCEWithLogitsLoss."
        )

#device
    if device is None:

        try:
            device = next(
                model.parameters()
            ).device

        except StopIteration:
            device = torch.device("cpu")

#train ou validation?
    training = optimizer is not None

    _configure_model_mode(
        model=model,
        training=training,
        freeze_strategy=freeze_strategy,
    )

#accumulateur?

    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

#gradien pdt entrainement
    gradient_context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

#boucle sur batch
    with gradient_context:

        for eeg, labels in loader:

#device
            eeg = eeg.to(device)
            labels = labels.to(device)
#reset gradien
            if training:

                optimizer.zero_grad(
                    set_to_none=True
                )

#foward
            logits = model(eeg)

            binary_logits = (
                _prepare_binary_logits(
                    logits
                )
            )

            # ------------------------------------------------------
            # Targets BCE
            #
            # Labels :
            # YO = 0
            # YF = 1
            #
            # BCEWithLogitsLoss attend des float.
            # ------------------------------------------------------

            binary_targets = labels.to(
                dtype=binary_logits.dtype
            )

#loss
            loss = criterion(
                binary_logits,
                binary_targets,
            )
#proba
            probability_yf = torch.sigmoid(
                binary_logits
            )

            predictions = (
                probability_yf >= 0.5
            ).long()

#backpro
            if training:

                loss.backward()

                optimizer.step()

            # ------------------------------------------------------
            # Statistiques
            # ------------------------------------------------------

            batch_size = labels.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_examples += batch_size

    # ==============================================================
    # FIN DE L'EPOCH
    # ==============================================================

    if total_examples == 0:

        raise ValueError(
            "Le DataLoader est vide."
        )

    average_loss = (
        total_loss
        / total_examples
    )

    accuracy = (
        correct_predictions
        / total_examples
    )

    return (
        average_loss,
        accuracy,
    )