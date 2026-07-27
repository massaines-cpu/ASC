from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def repair_bad_channel(
    input_path,
    output_path,
    participant_index=0,
    channel_index=5,
):
    """Remplace un canal EEG contaminé par la médiane des autres canaux.

    Le fichier original n'est pas modifié. Une copie corrigée est enregistrée
    dans output_path.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    data = np.load(input_path).copy()

    if data.ndim != 3:
        raise ValueError(
            f"Shape inattendue : {data.shape}. "
            "Shape attendue : (participants, canaux, temps)."
        )

    if participant_index >= data.shape[0]:
        raise IndexError("participant_index hors limites.")

    if channel_index >= data.shape[1]:
        raise IndexError("channel_index hors limites.")

    # On récupère tous les canaux sauf le canal contaminé.
    other_channels = np.delete(
        data[participant_index],
        channel_index,
        axis=0,
    )

    # Pour chaque instant, on calcule la médiane des autres canaux.
    replacement_signal = np.median(
        other_channels,
        axis=0,
    )

    # Remplacement du canal contaminé.
    data[participant_index, channel_index, :] = replacement_signal

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, data)

    print("Fichier original :", input_path)
    print("Fichier corrigé :", output_path)
    print(
        f"Canal remplacé : participant {participant_index}, "
        f"canal {channel_index}"
    )

input_file = (
    PROJECT_ROOT
    / "data"
    / "data_toy"
    / "J7"
    / "epochs"
    / "J7_E018_C5_YF_S1_5A15.npy"
)

output_file = (
    PROJECT_ROOT
    / "data"
    / "data_toy_repaired"
    / "J7"
    / "epochs"
    / "J7_E018_C5_YF_S1_5A15.npy"
)

repair_bad_channel(
    input_path=input_file,
    output_path=output_file,
    participant_index=0,
    channel_index=5,
)