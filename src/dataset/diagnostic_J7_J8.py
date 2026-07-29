"""
Diagnostic ciblé — pourquoi J7 et J8 restent les folds les plus faibles.

PARTIE 1 — J7 : l'artefact d'amplitude a-t-il été corrigé ?
-------------------------------------------------------------
On compare les statistiques du signal EEG de J7 entre l'ancien dataset
(data_toy) et le dataset réparé (data_toy_repaired), en réutilisant les
fonctions déjà écrites dans inspect_dataset.py.

PARTIE 2 — J8 : les erreurs sont-elles concentrées sur certaines
conditions, ou réparties uniformément ?
-------------------------------------------------------------
On recharge le meilleur modèle EEGNet entraîné pour le fold J8, on
prédit chaque participant individuellement, et on relie chaque
prédiction à sa condition (condition_name) pour voir si les erreurs
se regroupent sur une condition en particulier (ex: Collaboration,
sous-représentée dans J8 par rapport aux autres dyades).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset.inspect_dataset import (
    inspect_signal_statistics,
    find_extreme_values,
)
from src.dataset.labels import prepare_classification_table
from src.models.eegNET_model import EEGNet

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ==================================================================
# PARTIE 1 — J7 : ancien dataset vs dataset réparé
# ==================================================================

def compare_j7_before_after():
    """Compare les statistiques de J7 entre data_toy et data_toy_repaired."""

    for dataset_name in ["data_toy", "data_toy_repaired"]:
        dataset_root = PROJECT_ROOT / "data" / dataset_name
        j7_epochs_dir = dataset_root / "J7" / "epochs"

        if not j7_epochs_dir.exists():
            print(f"\n[{dataset_name}] Dossier introuvable : {j7_epochs_dir}")
            continue

        npy_files = sorted(j7_epochs_dir.glob("*.npy"))

        print("\n" + "=" * 70)
        print(f"J7 — {dataset_name} ({len(npy_files)} fichiers)")
        print("=" * 70)

        stats = inspect_signal_statistics(npy_files)
        print(f"Moyenne globale : {stats['mean']:.6e}")
        print(f"Écart-type global : {stats['std']:.6e}")
        print(f"Minimum : {stats['min']:.6e}")
        print(f"Maximum : {stats['max']:.6e}")
        print(f"NaN : {stats['n_nan']} | Inf : {stats['n_inf']}")

        extreme = find_extreme_values(npy_files, threshold=0.01)
        if extreme.empty:
            print("Aucune valeur extrême (|x| > 0.01) détectée.")
        else:
            print(f"\n{len(extreme)} fichier(s) avec des valeurs extrêmes :")
            print(
                extreme.head(10).to_string(
                    index=False, float_format=lambda v: f"{v:.6e}"
                )
            )


# ==================================================================
# PARTIE 2 — J8 : erreurs par condition
# ==================================================================

def analyze_j8_errors_by_condition():
    """Charge le meilleur modèle EEGNet du fold J8 et regarde si les
    erreurs de classification se concentrent sur certaines conditions."""

    # --- Métadonnées et labels ---
    metadata_path = PROJECT_ROOT / "data" / "all_metadata.csv"
    metadata = pd.read_csv(metadata_path)

    classification_table = prepare_classification_table(
        metadata=metadata,
        target_column="eyes_code",
        allowed_classes=["YO", "YF"],
        label_map={"YO": 0, "YF": 1},
    )

    j8_rows = (
        classification_table[classification_table["dyad_id"] == "J8"]
        .reset_index(drop=True)
    )

    if j8_rows.empty:
        print("Aucune ligne trouvée pour J8 dans classification_table.")
        print("Vérifie le nom de la colonne d'identifiant de dyade "
              "(dyad_id attendu).")
        return

    # --- Modèle : à adapter si le chemin du checkpoint diffère ---
    model_path = (
        PROJECT_ROOT
        / "models"
        / "protocol_B_eegnet_standardized"
        / "best_model_fold_J8.pt"
    )

    if not model_path.exists():
        print(f"Checkpoint introuvable : {model_path}")
        print("Adapte MODEL_PATH ou relance le fold J8 pour le régénérer.")
        return

    model = EEGNet(n_channels=32, n_classes=2, n_samples=5120)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dataset_root = PROJECT_ROOT / "data" / "data_toy_repaired"

    records = []

    with torch.no_grad():
        for _, row in j8_rows.iterrows():
            file_path = dataset_root / row["dyad_id"] / "epochs" / row["filename"]
            data = np.load(file_path)  # (2, 32, 5120)

            for participant_index in range(2):
                eeg = data[participant_index].astype(np.float32)

                # Même standardisation canal par canal que ParticipantDataset.
                channel_mean = eeg.mean(axis=1, keepdims=True)
                channel_std = eeg.std(axis=1, keepdims=True)
                eeg = (eeg - channel_mean) / (channel_std + 1e-8)

                eeg_tensor = torch.from_numpy(eeg).unsqueeze(0)  # (1, 32, 5120)

                logits = model(eeg_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = probabilities.argmax(dim=1).item()

                records.append(
                    {
                        "filename": row["filename"],
                        "participant_index": participant_index,
                        "condition_name": row.get("condition_name", "inconnue"),
                        "true_label": row["label"],
                        "predicted_label": predicted_class,
                        "correct": predicted_class == row["label"],
                        "confidence": probabilities.max().item(),
                    }
                )

    results = pd.DataFrame(records)

    print("\n" + "=" * 70)
    print("J8 — Répartition des erreurs par condition")
    print("=" * 70)

    summary = (
        results.groupby("condition_name")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_examples"})
        .sort_values("accuracy")
    )
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nAccuracy globale J8 :", f"{results['correct'].mean():.3f}")

    print("\nExemples mal classés (confiance la plus haute en premier) :")
    misclassified = results[~results["correct"]].sort_values(
        "confidence", ascending=False
    )
    print(misclassified.to_string(index=False))


# ==================================================================
# PROGRAMME PRINCIPAL
# ==================================================================

if __name__ == "__main__":
    compare_j7_before_after()
    analyze_j8_errors_by_condition()