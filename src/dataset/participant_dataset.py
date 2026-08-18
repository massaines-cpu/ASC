"""Dataset PyTorch au niveau du participant pour le projet ASC.

Un fichier contient deux participants de forme ``[2, canaux, temps]``.
Le Dataset transforme donc chaque fichier en deux exemples indépendants,
tout en laissant le découpage train/validation s'effectuer par dyade.

La standardisation est configurable. Elle reste activée pour les modèles
historiques ASC, mais doit être désactivée pour l'expérience SignalJEPA qui
utilise les amplitudes en microvolts attendues par le checkpoint.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ParticipantDataset(Dataset):
    """Charge un participant et son label à partir d'un fichier de dyade."""

    def __init__(
        self,
        classification_table,
        dataset_root,
        standardize: bool = True,
        expected_number_of_channels: int = 32,
        expected_number_of_timepoints: int = 5120,
    ) -> None:
        """Mémorise la table, le prétraitement et les dimensions attendues.

        Paramètres
        ----------
        classification_table : pandas.DataFrame
            Table contenant au minimum ``dyad_id``, ``filename`` et
            ``label``.

        dataset_root : str ou pathlib.Path
            Racine contenant les dossiers ``J1``, ``J2``, etc.

        standardize : bool
            Si vrai, applique le Z-score indépendamment à chaque canal du
            participant. Si faux, conserve les amplitudes du fichier.

        expected_number_of_channels, expected_number_of_timepoints : int
            Dimensions contrôlées à chaque chargement afin de détecter une
            confusion entre ``data_final`` et le dataset SignalJEPA.
        """

        self.classification_table = (
            classification_table.reset_index(drop=True)
        )
        self.dataset_root = Path(dataset_root)
        self.standardize = standardize
        self.expected_number_of_channels = expected_number_of_channels
        self.expected_number_of_timepoints = expected_number_of_timepoints

    def __len__(self) -> int:
        """Retourne deux exemples par fichier, un pour chaque participant."""

        return len(self.classification_table) * 2

    def __getitem__(self, index: int):
        """Charge un EEG ``[canaux, temps]`` et son label YO/YF."""

        file_index = index // 2
        participant_index = index % 2
        row = self.classification_table.iloc[file_index]

        file_path = (
            self.dataset_root
            / row["dyad_id"]
            / "epochs"
            / row["filename"]
        )

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier EEG introuvable : {file_path}")

        data = np.load(file_path)
        expected_shape = (
            2,
            self.expected_number_of_channels,
            self.expected_number_of_timepoints,
        )
        if data.shape != expected_shape:
            raise ValueError(
                f"Forme incorrecte pour {file_path} : {data.shape}. "
                f"Forme attendue : {expected_shape}."
            )

        eeg = data[participant_index].astype(np.float32, copy=True)

        if not np.isfinite(eeg).all():
            raise ValueError(
                f"Le participant {participant_index + 1} de {file_path} "
                "contient un NaN ou un infini."
            )

        if self.standardize:
            # Le calcul est effectué séparément pour chaque canal sur l'axe
            # temporel. Cette opération réduit les différences d'amplitude
            # entre participants dans les expériences ASC historiques.
            channel_mean = eeg.mean(axis=1, keepdims=True)
            channel_std = eeg.std(axis=1, keepdims=True)
            eeg = (eeg - channel_mean) / (channel_std + 1e-8)

        eeg_tensor = torch.from_numpy(eeg)
        label_tensor = torch.tensor(
            int(row["label"]),
            dtype=torch.long,
        )
        return eeg_tensor, label_tensor

