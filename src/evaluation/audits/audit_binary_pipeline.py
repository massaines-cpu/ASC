"""Vérifie la cohérence Logit / Sigmoid / BCEWithLogitsLoss.

Le but n'est pas d'entraîner les modèles. Ce test répond précisément à la
question : « la classification binaire possède-t-elle bien une Sigmoid ? »

Réponse attendue :
- ``forward`` produit un logit brut par EEG ;
- ``BCEWithLogitsLoss`` applique la formulation stable Logit + Sigmoid ;
- ``torch.sigmoid`` convertit le logit en probabilité pendant l'évaluation.
"""

from dataclasses import dataclass
import sys

import pandas as pd
import torch
from torch import nn

from src.config.settings import PROJECT_ROOT, REPORT_OUTPUT_ROOT


# Les imports des modèles existants deviennent possibles même si ce fichier est
# encore exécuté depuis le pack séparé plutôt que depuis le dépôt ASC.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.eegNET_model import EEGNet  # noqa: E402
from src.models.participant_linear_model import (  # noqa: E402
    SimpleParticipantClassifier,
)
from src.models.participant_non_linear_MLP_dropout import (  # noqa: E402
    NonLinearParticipantMLP,
)
from src.models.petit_eeg_cnn import Small_CNN_EEG  # noqa: E402


@dataclass(frozen=True)
class ModelAuditSpec:
    """Associe un nom lisible à une instance de modèle."""

    name: str
    model: nn.Module


def count_sigmoid_calls_during_forward(
    model: nn.Module,
    fake_eeg: torch.Tensor,
) -> int:
    """Compte les Sigmoid réellement exécutées par ``model.forward``.

    Un module ``nn.Sigmoid`` peut exister comme aide pour la prédiction sans
    être utilisé dans ``forward``. Les hooks permettent de distinguer ces deux
    situations.
    """

    call_counter = {"value": 0}
    hooks = []

    def record_call(_module, _inputs, _output) -> None:
        call_counter["value"] += 1

    for module in model.modules():
        if isinstance(module, nn.Sigmoid):
            hooks.append(module.register_forward_hook(record_call))

    try:
        model(fake_eeg)
    finally:
        for hook in hooks:
            hook.remove()

    return call_counter["value"]


def audit_one_model(specification: ModelAuditSpec) -> dict[str, object]:
    """Teste formes, loss, probabilités et gradients d'un modèle."""

    torch.manual_seed(42)
    model = specification.model
    model.train()

    fake_eeg = torch.randn(2, 32, 5120)
    binary_targets = torch.tensor([0.0, 1.0])

    sigmoid_calls = count_sigmoid_calls_during_forward(model, fake_eeg)
    logits = model(fake_eeg)

    if logits.shape != binary_targets.shape:
        raise ValueError(
            f"{specification.name} produit {tuple(logits.shape)} au lieu "
            f"de {tuple(binary_targets.shape)}."
        )

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, binary_targets)
    loss.backward()

    probabilities_yf = torch.sigmoid(logits.detach())
    probabilities_are_valid = bool(
        ((probabilities_yf >= 0.0) & (probabilities_yf <= 1.0)).all()
    )
    at_least_one_gradient = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    # Avec BCEWithLogitsLoss, une Sigmoid appelée dans forward serait une erreur
    # car la loss appliquerait ensuite la transformation une deuxième fois.
    audit_passed = (
        sigmoid_calls == 0
        and probabilities_are_valid
        and torch.isfinite(loss).item()
        and at_least_one_gradient
    )

    return {
        "model": specification.name,
        "forward_output": "one_raw_logit_per_eeg",
        "output_shape": str(tuple(logits.shape)),
        "sigmoid_calls_inside_forward": sigmoid_calls,
        "training_loss": "BCEWithLogitsLoss",
        "evaluation_probability": "torch.sigmoid(logits)",
        "loss_is_finite": bool(torch.isfinite(loss).item()),
        "probabilities_in_0_1": probabilities_are_valid,
        "gradient_present": at_least_one_gradient,
        "audit_passed": audit_passed,
    }


def main() -> None:
    """Audite les quatre architectures from scratch du projet ASC."""

    model_specs = (
        ModelAuditSpec("MLP linéaire", SimpleParticipantClassifier()),
        ModelAuditSpec(
            "MLP non linéaire",
            NonLinearParticipantMLP(
                hidden_layer_size=32,
                dropout_rate=0.0,
            ),
        ),
        ModelAuditSpec("Petit CNN", Small_CNN_EEG()),
        ModelAuditSpec(
            "EEGNet",
            EEGNet(n_channels=32, n_samples=5120),
        ),
    )

    rows = [audit_one_model(specification) for specification in model_specs]
    audit_table = pd.DataFrame(rows)

    REPORT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_OUTPUT_ROOT / "binary_pipeline_audit.csv"
    audit_table.to_csv(output_path, index=False)

    print(audit_table.to_string(index=False))
    print(f"\nRapport sauvegardé : {output_path}")

    if not audit_table["audit_passed"].all():
        raise RuntimeError(
            "Au moins un modèle ne respecte pas le contrat binaire ASC."
        )


if __name__ == "__main__":
    main()

