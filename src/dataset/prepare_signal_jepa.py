"""Prépare le dataset ASC pour l'expérience SignalJEPA Contextual, 32 canaux.

Contrairement à ``prepare_signal_jepa_prelocal.py``, ce script ne réduit ni
ne fenêtre le signal :

    EEG ASC [2 participants, 32 canaux, 5120 points à 512 Hz]
                              ↓
    rééchantillonnage à 128 Hz et conversion volts → microvolts
                              ↓
    [2 participants, 32 canaux, 1280 points à 128 Hz]

Les 32 canaux et les 10 secondes de chaque essai sont conservés tels quels :
c'est justement pour exploiter les 32 canaux ASC sans réduction que
``SignalJEPA_Contextual`` (avec ``channel_embedding='pretrain_aligned'``) a
été choisi plutôt que ``SignalJEPA_PreLocal`` — voir la docstring de
``src/models/signal_jepa_model.py``. Le résultat est un exemple par
participant et par essai, chargé par ``ParticipantDataset``
(``src/dataset/participant_dataset.py``), la même classe que pour
``data_final`` — pas de Dataset spécialisé nécessaire ici, contrairement au
cas PreLocal qui doit gérer cinq fenêtres par essai.

Avant le premier lancement
--------------------------
Lancer d'abord ``python -m src.dataset.diagnose_bandpass_filtering`` : ce
script mesure si les données ASC sont déjà filtrées entre 0,5 et 40 Hz
(bande attendue par le checkpoint SignalJEPA) et donne un verdict explicite.
Régler ``APPLY_BANDPASS`` ci-dessous en conséquence, jamais au hasard.
"""

from fractions import Fraction
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


# ============================================================================
# 1. PARAMÈTRES À CHOISIR AVANT LE LANCEMENT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DATASET_NAME = "data_final"
TARGET_DATASET_NAME = "data_signal_jepa_128hz_uv"

# False : les données source sont déjà filtrées de manière compatible.
# True  : le script ajoute un filtre 0,5-40 Hz avant le rééchantillonnage.
#
# Confirmé auprès du collègue en charge du traitement du signal (2026-08-21) :
# l'acquisition applique déjà un passe-bande [0,5 ; 48] Hz — exactement
# cohérent avec la coupure nette observée sur la PSD (diagnose_bandpass_
# filtering.py) juste après 48 Hz. Le checkpoint SignalJEPA attend 0,5-40 Hz :
# il reste donc une tranche 40-48 Hz que l'acquisition ne filtre pas. Ce
# second filtre plus étroit la retire, sans risque puisqu'il s'applique à un
# signal déjà filtré dans une bande plus large.
APPLY_BANDPASS = True

# Les 338 fichiers existants viennent de la tentative précédente, produite
# sans cette décision documentée : à régénérer volontairement.
OVERWRITE_EXISTING_FILES = True


# ============================================================================
# 2. DESCRIPTION DES DONNÉES ASC ET DE LA CIBLE SIGNALJEPA
# ============================================================================

SOURCE_SAMPLING_FREQUENCY = 512.0
TARGET_SAMPLING_FREQUENCY = 128.0

EXPECTED_PARTICIPANTS = 2
EXPECTED_SOURCE_CHANNELS = 32
EXPECTED_SOURCE_TIMEPOINTS = 5120

TARGET_TIMEPOINTS = 1280  # 10 s à 128 Hz, comme le signal source à 512 Hz.

MICROVOLTS_PER_VOLT = 1_000_000.0

# Ordre réel des canaux ASC. Identique à prepare_signal_jepa_prelocal.py et
# src/eeg_model_registry/config.py : ne jamais trier alphabétiquement.
ASC_CHANNEL_NAMES = [
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3",
    "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
    "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
    "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
]


def validate_configuration() -> None:
    """Vérifie les constantes avant de parcourir plusieurs centaines de fichiers."""

    if len(ASC_CHANNEL_NAMES) != EXPECTED_SOURCE_CHANNELS:
        raise ValueError("ASC_CHANNEL_NAMES doit contenir exactement 32 noms.")

    expected_resampled_timepoints = int(
        EXPECTED_SOURCE_TIMEPOINTS
        * TARGET_SAMPLING_FREQUENCY
        / SOURCE_SAMPLING_FREQUENCY
    )
    if expected_resampled_timepoints != TARGET_TIMEPOINTS:
        raise ValueError(
            "Le rééchantillonnage ne produit pas le nombre de points attendu : "
            f"{expected_resampled_timepoints} calculés contre "
            f"{TARGET_TIMEPOINTS} déclarés."
        )


def validate_source_array(eeg: np.ndarray, source_path: Path) -> None:
    """Refuse un fichier dont la forme ou les valeurs sont incompatibles."""

    expected_shape = (
        EXPECTED_PARTICIPANTS,
        EXPECTED_SOURCE_CHANNELS,
        EXPECTED_SOURCE_TIMEPOINTS,
    )

    if eeg.shape != expected_shape:
        raise ValueError(
            f"Forme incorrecte pour {source_path} : {eeg.shape}. "
            f"Forme attendue : {expected_shape}."
        )

    if not np.isfinite(eeg).all():
        raise ValueError(
            f"Le fichier {source_path} contient au moins un NaN ou un infini."
        )


def apply_bandpass_if_requested(eeg: np.ndarray) -> np.ndarray:
    """Ajoute éventuellement le filtre 0,5-40 Hz attendu par le checkpoint."""

    if not APPLY_BANDPASS:
        return eeg

    filter_coefficients = butter(
        N=4,
        Wn=(0.5, 40.0),
        btype="bandpass",
        fs=SOURCE_SAMPLING_FREQUENCY,
        output="sos",
    )

    # Le filtre est appliqué indépendamment le long de l'axe temporel.
    return sosfiltfilt(filter_coefficients, eeg, axis=-1)


def resample_to_128_hz(eeg: np.ndarray) -> np.ndarray:
    """Passe de 512 à 128 Hz avec le filtre anti-repliement de resample_poly."""

    ratio = Fraction(
        TARGET_SAMPLING_FREQUENCY / SOURCE_SAMPLING_FREQUENCY
    ).limit_denominator()

    resampled_eeg = resample_poly(
        eeg,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=-1,
    )

    if resampled_eeg.shape[-1] != TARGET_TIMEPOINTS:
        raise RuntimeError(
            "Le rééchantillonnage a produit "
            f"{resampled_eeg.shape[-1]} points au lieu de {TARGET_TIMEPOINTS}."
        )

    return resampled_eeg


def transform_one_file(eeg: np.ndarray) -> np.ndarray:
    """Applique, dans un ordre fixe, le pipeline complet à un fichier ASC."""

    transformed_eeg = eeg.astype(np.float64, copy=False)
    transformed_eeg = apply_bandpass_if_requested(transformed_eeg)
    transformed_eeg = resample_to_128_hz(transformed_eeg)

    # SignalJEPA a été pré-entraîné sur des amplitudes exprimées en µV.
    transformed_eeg = transformed_eeg * MICROVOLTS_PER_VOLT
    transformed_eeg = transformed_eeg.astype(np.float32)

    expected_shape = (
        EXPECTED_PARTICIPANTS,
        EXPECTED_SOURCE_CHANNELS,
        TARGET_TIMEPOINTS,
    )
    if transformed_eeg.shape != expected_shape:
        raise RuntimeError(
            f"Forme produite : {transformed_eeg.shape}. "
            f"Forme attendue : {expected_shape}."
        )

    if not np.isfinite(transformed_eeg).all():
        raise ValueError("La transformation a produit un NaN ou un infini.")

    return transformed_eeg


def write_manifest(rows: list[dict], results_dir: Path) -> None:
    """Enregistre un contrôle fichier par fichier pour la reproductibilité."""

    import csv

    manifest_path = results_dir / "dataset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Crée le dataset cible sans modifier les fichiers d'origine."""

    validate_configuration()

    source_root = PROJECT_ROOT / "data" / SOURCE_DATASET_NAME
    target_root = PROJECT_ROOT / "data" / TARGET_DATASET_NAME
    results_dir = (
        PROJECT_ROOT / "results" / "signal_jepa_preprocessing" / TARGET_DATASET_NAME
    )

    if not source_root.exists():
        raise FileNotFoundError(f"Dataset source introuvable : {source_root}")

    if source_root.resolve() == target_root.resolve():
        raise ValueError("Le dossier cible doit être différent de la source.")

    source_files = sorted(source_root.rglob("*.npy"))
    if not source_files:
        raise FileNotFoundError(f"Aucun fichier .npy trouvé dans {source_root}.")

    existing_files = sorted(target_root.rglob("*.npy"))
    if existing_files and not OVERWRITE_EXISTING_FILES:
        raise FileExistsError(
            f"{len(existing_files)} fichiers existent déjà dans {target_root}. "
            "Pour les recréer volontairement, régler OVERWRITE_EXISTING_FILES = True."
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for file_number, source_path in enumerate(source_files, start=1):
        relative_path = source_path.relative_to(source_root)
        target_path = target_root / relative_path

        source_eeg = np.load(source_path)
        validate_source_array(source_eeg, source_path)
        transformed_eeg = transform_one_file(source_eeg)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path, transformed_eeg)

        manifest_rows.append({
            "relative_path": str(relative_path),
            "source_shape": str(tuple(source_eeg.shape)),
            "target_shape": str(tuple(transformed_eeg.shape)),
            "target_min_microvolts": float(transformed_eeg.min()),
            "target_max_microvolts": float(transformed_eeg.max()),
            "target_mean_microvolts": float(transformed_eeg.mean()),
            "target_std_microvolts": float(transformed_eeg.std()),
            "all_values_finite": bool(np.isfinite(transformed_eeg).all()),
        })

        print(
            f"[{file_number:03d}/{len(source_files):03d}] "
            f"{relative_path} -> {transformed_eeg.shape}"
        )

    target_files = sorted(target_root.rglob("*.npy"))
    if len(target_files) != len(source_files):
        raise RuntimeError(
            f"{len(source_files)} fichiers source mais {len(target_files)} fichiers cible."
        )

    write_manifest(manifest_rows, results_dir)

    report = {
        "source_dataset": SOURCE_DATASET_NAME,
        "target_dataset": TARGET_DATASET_NAME,
        "number_of_files": len(source_files),
        "source_shape": [2, 32, 5120],
        "target_shape": [2, 32, TARGET_TIMEPOINTS],
        "source_sampling_frequency_hz": SOURCE_SAMPLING_FREQUENCY,
        "target_sampling_frequency_hz": TARGET_SAMPLING_FREQUENCY,
        "channel_names": ASC_CHANNEL_NAMES,
        "bandpass_applied": APPLY_BANDPASS,
        "bandpass_hz": [0.5, 40.0] if APPLY_BANDPASS else None,
        "unit_conversion": "volts_to_microvolts_x_1e6",
        "z_score_applied": False,
    }
    report_path = results_dir / "preprocessing_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nPréparation terminée.")
    print("Fichiers validés :", len(target_files))
    print("Dataset créé      :", target_root)
    print("Rapport            :", report_path)


if __name__ == "__main__":
    main()
