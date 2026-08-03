from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


cm = np.array([
    [32, 0],
    [11, 21]
])

output_dir = Path("results_j8_biais/confusion_matrices")
output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 1. Matrice avec les effectifs
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["YO", "YF"]
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("Matrice de confusion — EEGNet — J8")
ax.set_xlabel("Classe prédite")
ax.set_ylabel("Classe réelle")

fig.tight_layout()
fig.savefig(
    output_dir / "confusion_matrix_EEGNet_J8_counts.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)


# ---------------------------------------------------------
# 2. Matrice normalisée par classe réelle
# ---------------------------------------------------------
cm_normalized = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 5))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=["YO", "YF"]
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format=".2%",
    colorbar=False
)

ax.set_title("Matrice de confusion normalisée — EEGNet — J8")
ax.set_xlabel("Classe prédite")
ax.set_ylabel("Classe réelle")

fig.tight_layout()
fig.savefig(
    output_dir / "confusion_matrix_EEGNet_J8_normalized.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)