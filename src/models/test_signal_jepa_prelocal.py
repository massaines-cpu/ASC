"""Test technique court avant de lancer une LODO SignalJEPA PreLocal.

Ce fichier ne mesure aucune performance scientifique. Il vérifie seulement :

* la forme ``[batch, 19, 256]`` ;
* la sortie binaire ``[batch]`` après normalisation ;
* la compatibilité avec ``BCEWithLogitsLoss`` ;
* les paramètres entraînables de ``classifier_only`` ;
* la modification effective d'un poids après un pas d'optimisation.
"""

import torch
from torch import nn

from src.models.signal_jepa_prelocal_model import (
    configure_freezing,
    count_parameters,
    create_signal_jepa_prelocal,
    prepare_binary_logits,
)


# False teste l'architecture sans téléchargement.
# True vérifie également le chargement réel du checkpoint Hugging Face.
TEST_PRETRAINED_CHECKPOINT = False


def test_one_model(model, model_name: str) -> None:
    """Effectue un forward, un backward et un pas d'AdamW."""

    fake_eeg = torch.randn(4, 19, 256)
    fake_labels = torch.tensor([0, 1, 0, 1], dtype=torch.float32)

    logits = prepare_binary_logits(model(fake_eeg))
    if logits.shape != (4,):
        raise AssertionError(
            f"Sortie incorrecte pour {model_name} : {logits.shape}."
        )

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, fake_labels)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=5e-3)

    first_parameter = trainable_parameters[0]
    value_before = first_parameter.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    maximum_change = float(
        (first_parameter.detach() - value_before).abs().max()
    )
    if maximum_change == 0.0:
        raise AssertionError(
            f"Aucun poids n'a changé pour {model_name}."
        )

    total, trainable = count_parameters(model)
    print("-" * 64)
    print("Modèle                 :", model_name)
    print("Forme entrée           :", tuple(fake_eeg.shape))
    print("Forme logits           :", tuple(logits.shape))
    print("Loss                   :", float(loss.detach()))
    print("Paramètres totaux      :", f"{total:,}")
    print("Paramètres entraînables:", f"{trainable:,}")
    print("Modification poids max :", maximum_change)


def main() -> None:
    """Teste d'abord scratch, puis éventuellement le checkpoint."""

    scratch_model = create_signal_jepa_prelocal(
        model_variant="scratch",
        freeze_strategy="full_finetuning",
    )
    test_one_model(scratch_model, "scratch / full_finetuning")

    # Ce second test structurel vérifie que les nouvelles couches PreLocal
    # sont les seules entraînables en classifier_only.
    classifier_only_model = create_signal_jepa_prelocal(
        model_variant="scratch",
        freeze_strategy="full_finetuning",
    )
    configure_freezing(classifier_only_model, "classifier_only")

    trainable_names = {
        name
        for name, parameter in classifier_only_model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable_names:
        raise AssertionError("classifier_only ne contient aucun poids.")
    if not all(
        name.startswith(("spatial_conv.", "final_layer."))
        for name in trainable_names
    ):
        raise AssertionError(
            "classifier_only entraîne un bloc qui devrait être gelé : "
            f"{sorted(trainable_names)}"
        )
    if not any(name.startswith("spatial_conv.") for name in trainable_names):
        raise AssertionError("spatial_conv devrait être entraînable.")
    if not any(name.startswith("final_layer.") for name in trainable_names):
        raise AssertionError("final_layer devrait être entraînable.")

    test_one_model(
        classifier_only_model,
        "structure classifier_only",
    )

    if TEST_PRETRAINED_CHECKPOINT:
        pretrained_model = create_signal_jepa_prelocal(
            model_variant="pretrained",
            freeze_strategy="classifier_only",
        )
        test_one_model(
            pretrained_model,
            "pretrained / classifier_only",
        )

    print("\nTous les tests techniques PreLocal ont réussi.")


if __name__ == "__main__":
    main()
