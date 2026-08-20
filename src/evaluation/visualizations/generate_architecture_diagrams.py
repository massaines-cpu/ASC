"""Génère un schéma vérifiable pour chaque architecture from scratch.

Les dimensions ne sont pas écrites à la main : elles sont capturées pendant un
vrai passage avant avec une entrée factice [1, 32, 5120]. Cela réduit le risque
de présenter une dimension erronée dans le diaporama.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import torch
from torch import nn

from src.config.settings import PROJECT_ROOT, REPORT_OUTPUT_ROOT


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
class ArchitectureSpec:
    """Décrit une architecture et le nom de sa figure."""

    name: str
    file_stem: str
    model: nn.Module


def is_leaf_module(module: nn.Module) -> bool:
    """Un module feuille réalise une opération, sans sous-module interne."""

    return not any(True for _ in module.children())


def shape_text(value) -> str:
    """Convertit proprement la forme d'un tenseur ou d'une collection."""

    if isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    if isinstance(value, (list, tuple)):
        return " | ".join(shape_text(item) for item in value)
    return type(value).__name__


def collect_architecture_rows(
    model: nn.Module,
    fake_eeg: torch.Tensor,
) -> tuple[list[dict[str, object]], tuple[int, ...]]:
    """Capture les formes et paramètres des couches pendant ``forward``."""

    rows = []
    hooks = []

    def create_hook(module_name: str):
        def record_layer(module, inputs, output) -> None:
            rows.append({
                "layer_name": module_name,
                "layer_type": module.__class__.__name__,
                "input_shape": shape_text(inputs),
                "output_shape": shape_text(output),
                "parameters": sum(
                    parameter.numel()
                    for parameter in module.parameters(recurse=False)
                ),
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in module.parameters(recurse=False)
                    if parameter.requires_grad
                ),
            })
        return record_layer

    for module_name, module in model.named_modules():
        if module_name and is_leaf_module(module):
            hooks.append(module.register_forward_hook(create_hook(module_name)))

    model.eval()
    try:
        with torch.no_grad():
            output = model(fake_eeg)
    finally:
        for hook in hooks:
            hook.remove()

    return rows, tuple(output.shape)


def layer_color(layer_type: str) -> str:
    """Associe une couleur stable à la fonction principale d'une couche."""

    if "Conv" in layer_type:
        return "#77B6EA"
    if "Linear" in layer_type:
        return "#F4A261"
    if "Pool" in layer_type:
        return "#74C69D"
    if layer_type in {"ReLU", "ELU", "Sigmoid"}:
        return "#C9A0DC"
    if "Dropout" in layer_type:
        return "#F4CCCC"
    return "#D9E2EC"


def save_architecture_figure(
    specification: ArchitectureSpec,
    rows: list[dict[str, object]],
    output_shape: tuple[int, ...],
) -> None:
    """Enregistre un diagramme vertical et une table CSV vérifiable."""

    output_directory = REPORT_OUTPUT_ROOT / "architectures"
    output_directory.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(rows)
    table_path = output_directory / f"{specification.file_stem}_layers.csv"
    table.to_csv(table_path, index=False)

    number_of_rows = len(rows) + 2
    figure_height = max(7.0, number_of_rows * 0.8)
    figure, axis = plt.subplots(figsize=(12, figure_height))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, number_of_rows + 1)
    axis.axis("off")

    total_parameters = sum(
        parameter.numel()
        for parameter in specification.model.parameters()
    )
    axis.set_title(
        (
            f"{specification.name} — architecture mesurée\n"
            f"{total_parameters:,} paramètres au total"
        ),
        fontsize=16,
        weight="bold",
        pad=20,
    )

    blocks = [{
        "layer_name": "Input EEG",
        "layer_type": "Input",
        "input_shape": "—",
        "output_shape": "(1, 32, 5120)",
        "parameters": 0,
    }, *rows]

    for row_index, row in enumerate(blocks):
        y_position = number_of_rows - row_index
        box = FancyBboxPatch(
            (0.8, y_position - 0.35),
            10.4,
            0.7,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=layer_color(str(row["layer_type"])),
            edgecolor="#132238",
            linewidth=1.2,
        )
        axis.add_patch(box)

        axis.text(
            1.05,
            y_position,
            f"{row['layer_name']} ({row['layer_type']})",
            va="center",
            ha="left",
            fontsize=9,
            weight="bold",
        )
        axis.text(
            6.0,
            y_position,
            f"sortie {row['output_shape']}",
            va="center",
            ha="center",
            fontsize=9,
        )
        axis.text(
            10.9,
            y_position,
            f"{int(row['parameters']):,} p.",
            va="center",
            ha="right",
            fontsize=9,
        )

        if row_index < len(blocks) - 1:
            axis.annotate(
                "",
                xy=(6, y_position - 0.72),
                xytext=(6, y_position - 0.38),
                arrowprops={"arrowstyle": "->", "color": "#132238"},
            )

    axis.text(
        6,
        0.35,
        (
            f"Sortie finale : {output_shape} — un logit brut YF par EEG. "
            "La probabilité est calculée ensuite avec Sigmoid."
        ),
        ha="center",
        fontsize=10,
        color="#374151",
    )

    figure.tight_layout()
    figure_path = output_directory / f"{specification.file_stem}.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Architecture sauvegardée : {figure_path}")
    print(f"Table des couches        : {table_path}")


def main() -> None:
    """Génère les quatre architectures comparées dans ASC."""

    torch.manual_seed(42)
    fake_eeg = torch.randn(1, 32, 5120)

    specifications = (
        ArchitectureSpec(
            "MLP linéaire",
            "architecture_mlp_lineaire",
            SimpleParticipantClassifier(),
        ),
        ArchitectureSpec(
            "MLP non linéaire — 32 neurones",
            "architecture_mlp_non_lineaire",
            NonLinearParticipantMLP(
                hidden_layer_size=32,
                dropout_rate=0.0,
            ),
        ),
        ArchitectureSpec(
            "Petit CNN",
            "architecture_petit_cnn",
            Small_CNN_EEG(),
        ),
        ArchitectureSpec(
            "EEGNet simplifié",
            "architecture_eegnet",
            EEGNet(n_channels=32, n_samples=5120),
        ),
    )

    for specification in specifications:
        rows, output_shape = collect_architecture_rows(
            model=specification.model,
            fake_eeg=fake_eeg,
        )
        save_architecture_figure(
            specification=specification,
            rows=rows,
            output_shape=output_shape,
        )


if __name__ == "__main__":
    main()

