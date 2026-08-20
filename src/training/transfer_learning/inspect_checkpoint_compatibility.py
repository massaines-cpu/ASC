"""Compare un checkpoint local au modèle EEGNet binaire ASC.

La compatibilité d'un poids exige simultanément :
1. une correspondance de nom explicite ;
2. une forme strictement identique.

Le classifieur final est toujours ignoré : la tâche source n'est pas YO/YF et
sa tête de sortie ne doit pas être transférée.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys

import pandas as pd
import torch

from src.config.settings import (
    CHECKPOINT_KEY_RENAMES,
    EEGNET_CHECKPOINT_PATH,
    PROJECT_ROOT,
    REPORT_OUTPUT_ROOT,
)


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.eegNET_model import EEGNet  # noqa: E402


STATE_DICT_CONTAINER_KEYS = (
    "state_dict",
    "model_state_dict",
    "model",
    "weights",
)

IGNORED_TARGET_PREFIXES = ("classifier.",)


def load_checkpoint_object(checkpoint_path: Path):
    """Charge le checkpoint sur CPU afin d'éviter une dépendance au GPU."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

    return torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


def extract_state_dict(checkpoint_object) -> dict[str, torch.Tensor]:
    """Extrait le dictionnaire de poids des formats PyTorch habituels."""

    if not isinstance(checkpoint_object, Mapping):
        raise TypeError(
            "Le checkpoint doit contenir un dictionnaire de poids PyTorch."
        )

    for container_key in STATE_DICT_CONTAINER_KEYS:
        nested_object = checkpoint_object.get(container_key)
        if isinstance(nested_object, Mapping):
            tensor_values = {
                str(key): value
                for key, value in nested_object.items()
                if isinstance(value, torch.Tensor)
            }
            if tensor_values:
                return tensor_values

    direct_tensor_values = {
        str(key): value
        for key, value in checkpoint_object.items()
        if isinstance(value, torch.Tensor)
    }
    if direct_tensor_values:
        return direct_tensor_values

    raise ValueError("Aucun state_dict tensoriel n'a été trouvé.")


def remove_common_prefixes(parameter_name: str) -> str:
    """Retire uniquement des préfixes techniques connus."""

    cleaned_name = parameter_name
    for prefix in ("module.", "model.", "network."):
        if cleaned_name.startswith(prefix):
            cleaned_name = cleaned_name[len(prefix):]
    return cleaned_name


def map_source_name(source_name: str) -> str:
    """Applique une correspondance documentée, jamais une supposition."""

    cleaned_name = remove_common_prefixes(source_name)
    return CHECKPOINT_KEY_RENAMES.get(cleaned_name, cleaned_name)


def inspect_state_dict(
    source_state_dict: dict[str, torch.Tensor],
    target_model: EEGNet,
) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    """Construit le rapport et les seuls poids autorisés au transfert."""

    target_state_dict = target_model.state_dict()
    report_rows = []
    compatible_state_dict = {}

    for source_name, source_tensor in source_state_dict.items():
        target_name = map_source_name(source_name)

        if target_name.startswith(IGNORED_TARGET_PREFIXES):
            status = "ignored_classifier"
            target_shape = ""
        elif target_name not in target_state_dict:
            status = "target_name_absent"
            target_shape = ""
        else:
            target_tensor = target_state_dict[target_name]
            target_shape = str(tuple(target_tensor.shape))

            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                status = "shape_mismatch"
            else:
                status = "compatible"
                compatible_state_dict[target_name] = source_tensor.detach().cpu()

        report_rows.append({
            "source_name": source_name,
            "mapped_target_name": target_name,
            "source_shape": str(tuple(source_tensor.shape)),
            "target_shape": target_shape,
            "number_of_source_values": source_tensor.numel(),
            "status": status,
        })

    transferred_target_names = set(compatible_state_dict)
    for target_name, target_tensor in target_state_dict.items():
        if target_name in transferred_target_names:
            continue
        if target_name.startswith(IGNORED_TARGET_PREFIXES):
            target_status = "target_classifier_reinitialized"
        else:
            target_status = "target_random_initialization"

        report_rows.append({
            "source_name": "",
            "mapped_target_name": target_name,
            "source_shape": "",
            "target_shape": str(tuple(target_tensor.shape)),
            "number_of_source_values": 0,
            "status": target_status,
        })

    return pd.DataFrame(report_rows), compatible_state_dict


def load_compatible_weights(
    model: EEGNet,
    checkpoint_path: Path,
) -> pd.DataFrame:
    """Charge uniquement les poids prouvés compatibles et renvoie le rapport."""

    checkpoint_object = load_checkpoint_object(checkpoint_path)
    source_state_dict = extract_state_dict(checkpoint_object)
    report, compatible_state_dict = inspect_state_dict(
        source_state_dict=source_state_dict,
        target_model=model,
    )

    if not compatible_state_dict:
        raise RuntimeError(
            "Aucun poids compatible : le fine-tuning ne doit pas être lancé."
        )

    model.load_state_dict(compatible_state_dict, strict=False)
    return report


def main() -> None:
    """Génère un rapport CSV sans lancer d'entraînement."""

    if EEGNET_CHECKPOINT_PATH is None:
        raise ValueError(
            "Renseigne EEGNET_CHECKPOINT_PATH dans settings.py après avoir "
            "téléchargé un checkpoint et documenté sa source."
        )

    target_model = EEGNet(n_channels=32, n_samples=5120)
    checkpoint_object = load_checkpoint_object(EEGNET_CHECKPOINT_PATH)
    source_state_dict = extract_state_dict(checkpoint_object)
    report, compatible_state_dict = inspect_state_dict(
        source_state_dict=source_state_dict,
        target_model=target_model,
    )

    output_directory = REPORT_OUTPUT_ROOT / "transfer_learning"
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "checkpoint_compatibility.csv"
    report.to_csv(report_path, index=False)

    source_value_count = sum(
        tensor.numel() for tensor in source_state_dict.values()
    )
    compatible_value_count = sum(
        tensor.numel() for tensor in compatible_state_dict.values()
    )
    compatible_ratio = (
        compatible_value_count / source_value_count
        if source_value_count > 0
        else 0.0
    )

    print(report["status"].value_counts().to_string())
    print(f"\nPoids source       : {source_value_count:,} valeurs")
    print(f"Poids transférables: {compatible_value_count:,} valeurs")
    print(f"Ratio transférable : {compatible_ratio:.2%}")
    print(f"Rapport             : {report_path}")


if __name__ == "__main__":
    main()

