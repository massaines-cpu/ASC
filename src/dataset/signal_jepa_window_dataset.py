"""Dataset et DataLoaders pour les fenêtres SignalJEPA PreLocal.

Un fichier préparé possède la forme ``[2, 5, canaux, 256]`` :

* 2 participants ;
* 5 fenêtres de deux secondes ;
* le nombre de canaux du dataset préparé (19 pour le montage réduit, 32 pour
  le montage ASC complet — cf. ``expected_number_of_channels`` ci-dessous) ;
* 256 points temporels à 128 Hz.

Chaque fenêtre devient un exemple d'entraînement. Le découpage train /
validation reste cependant effectué par dyade avant la création du Dataset,
ce qui préserve strictement le protocole Leave-One-Dyad-Out.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


EXPECTED_PARTICIPANTS = 2
EXPECTED_WINDOWS_PER_PARTICIPANT = 5
EXPECTED_TIMEPOINTS = 256

# Valeur par défaut historique : le montage réduit à 19 électrodes. Passer
# expected_number_of_channels=32 pour le montage ASC complet.
DEFAULT_EXPECTED_CHANNELS = 19


class SignalJEPAWindowDataset(Dataset):
    """Retourne une fenêtre EEG, son label et son participant d'origine."""

    def __init__(
        self,
        classification_table,
        dataset_root: str | Path,
        cache_files: bool = True,
        expected_number_of_channels: int = DEFAULT_EXPECTED_CHANNELS,
    ) -> None:
        self.classification_table = classification_table.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.cache_files = cache_files
        self.expected_number_of_channels = expected_number_of_channels

        # Sans cache, le même fichier serait relu dix fois par epoch
        # (2 participants × 5 fenêtres). Le dataset préparé reste assez petit
        # pour conserver les tableaux déjà utilisés en mémoire.
        self._file_cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        """Retourne 2 participants × 5 fenêtres pour chaque fichier."""

        examples_per_file = (
            EXPECTED_PARTICIPANTS * EXPECTED_WINDOWS_PER_PARTICIPANT
        )
        return len(self.classification_table) * examples_per_file

    def __getitem__(self, index: int):
        """Charge une fenêtre de forme ``[19, 256]``.

        ``sample_id`` identifie le participant d'origine. Il permettra de
        moyenner ses cinq probabilités lors de la validation finale, afin de
        conserver une évaluation au niveau participant et non au niveau
        fenêtre.
        """

        examples_per_file = (
            EXPECTED_PARTICIPANTS * EXPECTED_WINDOWS_PER_PARTICIPANT
        )
        file_index = index // examples_per_file
        position_inside_file = index % examples_per_file

        participant_index = (
            position_inside_file // EXPECTED_WINDOWS_PER_PARTICIPANT
        )
        window_index = (
            position_inside_file % EXPECTED_WINDOWS_PER_PARTICIPANT
        )

        row = self.classification_table.iloc[file_index]
        file_path = (
            self.dataset_root
            / str(row["dyad_id"])
            / "epochs"
            / str(row["filename"])
        )

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier EEG introuvable : {file_path}")

        file_eeg = self._load_file(file_path)
        expected_shape = (
            EXPECTED_PARTICIPANTS,
            EXPECTED_WINDOWS_PER_PARTICIPANT,
            self.expected_number_of_channels,
            EXPECTED_TIMEPOINTS,
        )
        if file_eeg.shape != expected_shape:
            raise ValueError(
                f"Forme incorrecte pour {file_path} : {file_eeg.shape}. "
                f"Forme attendue : {expected_shape}."
            )

        eeg_window = file_eeg[
            participant_index,
            window_index,
        ].astype(np.float32, copy=True)

        if not np.isfinite(eeg_window).all():
            raise ValueError(
                f"P{participant_index + 1}, fenêtre {window_index} de "
                f"{file_path} contient un NaN ou un infini."
            )

        # Aucun Z-score n'est appliqué ici : les valeurs en microvolts sont
        # conservées pour suivre le prétraitement du checkpoint SignalJEPA.
        eeg_tensor = torch.from_numpy(eeg_window)
        label_tensor = torch.tensor(int(row["label"]), dtype=torch.long)

        sample_id = (
            f"{row['dyad_id']}/{row['filename']}/P{participant_index + 1}"
        )

        return (
            eeg_tensor,
            label_tensor,
            sample_id,
            window_index,
        )

    def _load_file(self, file_path: Path) -> np.ndarray:
        """Charge un fichier une seule fois lorsque le cache est activé."""

        if self.cache_files and file_path in self._file_cache:
            return self._file_cache[file_path]

        file_eeg = np.load(file_path)
        if self.cache_files:
            self._file_cache[file_path] = file_eeg

        return file_eeg


def split_table_by_dyad(
    classification_table,
    train_dyads: list[str],
    validation_dyads: list[str],
):
    """Sépare les métadonnées avant de générer la moindre fenêtre."""

    overlap = set(train_dyads) & set(validation_dyads)
    if overlap:
        raise ValueError(
            "Dyades présentes dans train et validation : "
            f"{sorted(overlap)}."
        )

    train_table = classification_table[
        classification_table["dyad_id"].isin(train_dyads)
    ].copy()
    validation_table = classification_table[
        classification_table["dyad_id"].isin(validation_dyads)
    ].copy()

    if train_table.empty:
        raise ValueError("La table d'entraînement est vide.")
    if validation_table.empty:
        raise ValueError("La table de validation est vide.")

    return train_table, validation_table


def create_signal_jepa_window_dataloaders(
    classification_table,
    dataset_root: str | Path,
    train_dyads: list[str],
    validation_dyads: list[str],
    batch_size: int,
    random_seed: int,
    expected_number_of_channels: int = DEFAULT_EXPECTED_CHANNELS,
):
    """Crée les DataLoaders train et validation de façon reproductible."""

    train_table, validation_table = split_table_by_dyad(
        classification_table=classification_table,
        train_dyads=train_dyads,
        validation_dyads=validation_dyads,
    )

    train_dataset = SignalJEPAWindowDataset(
        classification_table=train_table,
        dataset_root=dataset_root,
        expected_number_of_channels=expected_number_of_channels,
    )
    validation_dataset = SignalJEPAWindowDataset(
        classification_table=validation_table,
        dataset_root=dataset_root,
        expected_number_of_channels=expected_number_of_channels,
    )

    # Ce générateur rend l'ordre de mélange reproductible pour une seed fixe.
    train_generator = torch.Generator()
    train_generator.manual_seed(random_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, validation_loader
