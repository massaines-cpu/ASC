"""Prépare une copie du dataset ASC adaptée à SignalJEPA.

Le script ne modifie jamais le dataset source. Il reproduit son arborescence
dans un nouveau dossier et transforme chaque fichier :

    [2 participants, 32 canaux, 5120 points à 512 Hz]
                            ↓
    [2 participants, 32 canaux, 1280 points à 128 Hz]

La durée reste de 10 secondes. Les valeurs sont ensuite converties des volts
vers les microvolts, échelle utilisée lors du pré-entraînement de SignalJEPA.

Le filtrage 0,5-40 Hz est optionnel. Il ne faut activer ``--apply-bandpass``
que si la provenance des données confirme que ce filtrage n'a pas déjà été
réalisé.

Exemples
--------
Préparation sans ajouter de filtre :

    python -m src.dataset.prepare_signal_jepa

Si les données sont confirmées comme non filtrées :

    python -m src.dataset.prepare_signal_jepa \
        --apply-bandpass
"""

from argparse import ArgumentParser
import csv
from fractions import Fraction
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_SAMPLING_FREQUENCY = 512.0
TARGET_SAMPLING_FREQUENCY = 128.0
EXPECTED_PARTICIPANTS = 2
EXPECTED_CHANNELS = 32
EXPECTED_SOURCE_TIMEPOINTS = 5120
EXPECTED_TARGET_TIMEPOINTS = 1280
MICROVOLTS_PER_VOLT = 1_000_000.0


def parse_arguments():
    """Lit les paramètres sans imposer de chemins absolus au projet."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dataset",
        default="data_final",
        help="Nom du dossier source situé dans data/.",
    )
    parser.add_argument(
        "--target-dataset",
        default=None,
        help=(
            "Nom du nouveau dossier créé dans data/. Si l'option est omise, "
            "le nom indique automatiquement si le filtre a été appliqué."
        ),
    )
    parser.add_argument(
        "--apply-bandpass",
        action="store_true",
        help=(
            "Applique un filtre Butterworth 0,5-40 Hz avant le "
            "rééchantillonnage. À utiliser seulement si nécessaire."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Autorise le remplacement des fichiers cibles déjà présents.",
    )
    return parser.parse_args()


def validate_source_array(eeg: np.ndarray, file_path: Path) -> None:
    """Arrête le script si un fichier ne respecte pas le format ASC."""

    expected_shape = (
        EXPECTED_PARTICIPANTS,
        EXPECTED_CHANNELS,
        EXPECTED_SOURCE_TIMEPOINTS,
    )
    if eeg.shape != expected_shape:
        raise ValueError(
            f"Forme inattendue pour {file_path} : {eeg.shape}, "
            f"forme attendue : {expected_shape}."
        )

    if not np.isfinite(eeg).all():
        raise ValueError(
            f"Le fichier {file_path} contient au moins un NaN ou un infini."
        )


def apply_optional_bandpass(eeg: np.ndarray) -> np.ndarray:
    """Filtre les signaux entre 0,5 et 40 Hz, canal par canal.

    Le filtrage est réalisé avant la réduction de fréquence afin de maîtriser
    le contenu fréquentiel transmis au rééchantillonnage.
    """

    filter_coefficients = butter(
        N=4,
        Wn=(0.5, 40.0),
        btype="bandpass",
        fs=SOURCE_SAMPLING_FREQUENCY,
        output="sos",
    )
    return sosfiltfilt(
        filter_coefficients,
        eeg,
        axis=-1,
    )


def resample_to_128_hz(eeg: np.ndarray) -> np.ndarray:
    """Rééchantillonne de 512 à 128 Hz avec filtre anti-repliement."""

    ratio = Fraction(
        TARGET_SAMPLING_FREQUENCY / SOURCE_SAMPLING_FREQUENCY
    ).limit_denominator()

    resampled_eeg = resample_poly(
        eeg,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=-1,
    )

    if resampled_eeg.shape[-1] != EXPECTED_TARGET_TIMEPOINTS:
        raise RuntimeError(
            "Le rééchantillonnage a produit "
            f"{resampled_eeg.shape[-1]} points au lieu de "
            f"{EXPECTED_TARGET_TIMEPOINTS}."
        )

    return resampled_eeg


def prepare_one_file(
    source_path: Path,
    target_path: Path,
    apply_bandpass: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Transforme un fichier et sauvegarde une copie propre en float32."""

    source_eeg = np.load(source_path)
    validate_source_array(source_eeg, source_path)

    transformed_eeg = source_eeg.astype(np.float64, copy=False)

    if apply_bandpass:
        transformed_eeg = apply_optional_bandpass(transformed_eeg)

    transformed_eeg = resample_to_128_hz(transformed_eeg)

    # Les amplitudes ASC sont supposées être exprimées en volts. Cette
    # conversion rapproche les entrées de la distribution utilisée pour le
    # pré-entraînement de SignalJEPA. Elle devra être confirmée avec la
    # documentation d'acquisition.
    transformed_eeg = transformed_eeg * MICROVOLTS_PER_VOLT
    transformed_eeg = transformed_eeg.astype(np.float32)

    if not np.isfinite(transformed_eeg).all():
        raise ValueError(
            f"La transformation de {source_path} a produit un NaN ou infini."
        )

    expected_shape = (
        EXPECTED_PARTICIPANTS,
        EXPECTED_CHANNELS,
        EXPECTED_TARGET_TIMEPOINTS,
    )
    if transformed_eeg.shape != expected_shape:
        raise RuntimeError(
            f"Forme cible incorrecte pour {source_path} : "
            f"{transformed_eeg.shape} au lieu de {expected_shape}."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(target_path, transformed_eeg)

    return source_eeg, transformed_eeg


def save_example_plot(
    source_eeg: np.ndarray,
    transformed_eeg: np.ndarray,
    relative_path: Path,
    results_dir: Path,
) -> None:
    """Compare visuellement le même canal avant et après transformation."""

    source_time = np.arange(EXPECTED_SOURCE_TIMEPOINTS) / 512.0
    target_time = np.arange(EXPECTED_TARGET_TIMEPOINTS) / 128.0

    figure, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    axes[0].plot(source_time, source_eeg[0, 0], linewidth=0.7)
    axes[0].set_title("Signal source — P1, Fp1, 512 Hz, volts")
    axes[0].set_ylabel("Amplitude (V)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(target_time, transformed_eeg[0, 0], linewidth=0.7)
    axes[1].set_title("Signal SignalJEPA — P1, Fp1, 128 Hz, microvolts")
    axes[1].set_xlabel("Temps (secondes)")
    axes[1].set_ylabel("Amplitude (µV)")
    axes[1].grid(alpha=0.25)

    figure.suptitle(f"Contrôle du rééchantillonnage : {relative_path}")
    figure.tight_layout()
    figure.savefig(
        results_dir / "signal_before_after_resampling.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_manifest(rows: list[dict], results_dir: Path) -> None:
    """Enregistre un contrôle fichier par fichier pour la traçabilité."""

    manifest_path = results_dir / "signal_jepa_dataset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Génère le nouveau dataset puis vérifie son intégrité globale."""

    arguments = parse_arguments()
    target_dataset_name = arguments.target_dataset
    if target_dataset_name is None:
        target_dataset_name = (
            "data_signal_jepa_128hz_uv_bandpass_0_5_40"
            if arguments.apply_bandpass
            else "data_signal_jepa_128hz_uv"
        )

    source_root = PROJECT_ROOT / "data" / arguments.source_dataset
    target_root = PROJECT_ROOT / "data" / target_dataset_name
    results_dir = (
        PROJECT_ROOT
        / "results"
        / "signal_jepa_preprocessing"
        / target_dataset_name
    )

    if not source_root.exists():
        raise FileNotFoundError(f"Dataset source introuvable : {source_root}")

    if source_root.resolve() == target_root.resolve():
        raise ValueError(
            "Le dossier cible doit être différent du dossier source."
        )

    source_files = sorted(source_root.rglob("*.npy"))
    if not source_files:
        raise FileNotFoundError(
            f"Aucun fichier .npy trouvé dans {source_root}."
        )

    existing_target_files = sorted(target_root.rglob("*.npy"))
    if existing_target_files and not arguments.overwrite:
        raise FileExistsError(
            f"Le dossier cible contient déjà {len(existing_target_files)} "
            "fichiers. Relance avec --overwrite seulement si tu souhaites "
            "les remplacer."
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    example_saved = False

    for file_index, source_path in enumerate(source_files, start=1):
        relative_path = source_path.relative_to(source_root)
        target_path = target_root / relative_path

        source_eeg, transformed_eeg = prepare_one_file(
            source_path=source_path,
            target_path=target_path,
            apply_bandpass=arguments.apply_bandpass,
        )

        manifest_rows.append({
            "relative_path": str(relative_path),
            "source_shape": str(tuple(source_eeg.shape)),
            "target_shape": str(tuple(transformed_eeg.shape)),
            "source_dtype": str(source_eeg.dtype),
            "target_dtype": str(transformed_eeg.dtype),
            "source_max_absolute": float(np.abs(source_eeg).max()),
            "target_max_absolute_microvolts": float(
                np.abs(transformed_eeg).max()
            ),
            "all_values_finite": bool(np.isfinite(transformed_eeg).all()),
        })

        if not example_saved:
            save_example_plot(
                source_eeg=source_eeg,
                transformed_eeg=transformed_eeg,
                relative_path=relative_path,
                results_dir=results_dir,
            )
            example_saved = True

        print(
            f"[{file_index:03d}/{len(source_files):03d}] "
            f"{relative_path} -> {transformed_eeg.shape}"
        )

    target_files = sorted(target_root.rglob("*.npy"))
    source_relative_paths = {
        path.relative_to(source_root)
        for path in source_files
    }
    target_relative_paths = {
        path.relative_to(target_root)
        for path in target_files
    }

    if source_relative_paths != target_relative_paths:
        missing = source_relative_paths - target_relative_paths
        unexpected = target_relative_paths - source_relative_paths
        raise RuntimeError(
            "L'arborescence cible ne correspond pas à la source. "
            f"Manquants : {sorted(missing)}. "
            f"Inattendus : {sorted(unexpected)}."
        )

    write_manifest(manifest_rows, results_dir)

    preprocessing_report = {
        "source_dataset": arguments.source_dataset,
        "target_dataset": target_dataset_name,
        "number_of_files": len(source_files),
        "source_sampling_frequency_hz": SOURCE_SAMPLING_FREQUENCY,
        "target_sampling_frequency_hz": TARGET_SAMPLING_FREQUENCY,
        "source_timepoints": EXPECTED_SOURCE_TIMEPOINTS,
        "target_timepoints": EXPECTED_TARGET_TIMEPOINTS,
        "duration_seconds": (
            EXPECTED_TARGET_TIMEPOINTS / TARGET_SAMPLING_FREQUENCY
        ),
        "bandpass_applied": arguments.apply_bandpass,
        "bandpass_hz": [0.5, 40.0] if arguments.apply_bandpass else None,
        "unit_conversion": "volts_to_microvolts_x_1e6",
        "z_score_applied": False,
    }
    report_path = results_dir / "preprocessing_report.json"
    report_path.write_text(
        json.dumps(preprocessing_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nPréparation terminée.")
    print("Fichiers validés :", len(target_files))
    print("Dataset créé      :", target_root)
    print("Rapport            :", report_path)


if __name__ == "__main__":
    main()
