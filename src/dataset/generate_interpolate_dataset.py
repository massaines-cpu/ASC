"""Génère une copie du dataset avec interpolation régionale ciblée.

La détection reste identique au diagnostic actuel : un couple
participant–électrode est sélectionné lorsqu'au moins une de ses valeurs
vérifie ``abs(x) > 0.01``. La correction remplace alors le canal entier de
ce participant par une moyenne pondérée de voisins de la même région.

Les fichiers originaux ne sont jamais modifiés. Chaque opération est
consignée dans un CSV afin de rendre l'expérience reproductible.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from inspect_electrodes_participant import (
    ABSOLUTE_AMPLITUDE_THRESHOLD,
    ELECTRODE_NAMES,
    ELECTRODE_POSITIONS,
    ELECTRODE_REGIONS,
    analyse_by_electrode,
    apply_regional_interpolation,
    find_epoch_files,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET_ROOT = PROJECT_ROOT / "data" / "data_toy"

# Ce nom décrit une nouvelle expérience. Il évite de présenter la
# correction comme définitivement validée avant la comparaison LODO.
OUTPUT_DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "data_final"
)
RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiment_E_regional_participant"
)


def find_bad_targets(
    source_path: Path,
    threshold: float,
) -> tuple[list[tuple[int, str]], list[dict]]:
    """Identifie précisément les couples participant–électrode contaminés."""

    electrode_rows = analyse_by_electrode(
        npy_path=source_path,
        threshold=threshold,
        electrode_names=ELECTRODE_NAMES,
    )
    contaminated_rows = [
        row
        for row in electrode_rows
        if row["n_above_threshold"] > 0
    ]
    bad_targets = [
        (int(row["participant_index"]), row["electrode"])
        for row in contaminated_rows
    ]
    return bad_targets, contaminated_rows


def verify_untargeted_channels_are_unchanged(
    original: np.ndarray,
    corrected: np.ndarray,
    bad_targets: list[tuple[int, str]],
) -> None:
    """Vérifie que seules les zones explicitement ciblées ont changé."""

    allowed_changes = np.zeros(original.shape[:2], dtype=bool)
    for participant_index, electrode_name in bad_targets:
        channel_index = ELECTRODE_NAMES.index(electrode_name)
        allowed_changes[participant_index, channel_index] = True

    for participant_index in range(original.shape[0]):
        for channel_index in range(original.shape[1]):
            if allowed_changes[participant_index, channel_index]:
                continue
            if not np.array_equal(
                original[participant_index, channel_index, :],
                corrected[participant_index, channel_index, :],
                equal_nan=True,
            ):
                raise RuntimeError(
                    "Une donnée non ciblée a été modifiée : "
                    f"P{participant_index + 1}, "
                    f"{ELECTRODE_NAMES[channel_index]}."
                )


def generate_corrected_dataset(
    source_root: Path,
    output_root: Path,
    threshold: float,
) -> pd.DataFrame:
    """Copie tous les fichiers et corrige uniquement les cibles détectées."""

    if source_root.resolve() == output_root.resolve():
        raise ValueError(
            "Le dossier de sortie doit être différent du dataset original."
        )

    source_files = find_epoch_files(source_root)
    if not source_files:
        raise FileNotFoundError(f"Aucun fichier EEG trouvé dans {source_root}.")

    correction_rows = []

    for file_number, source_path in enumerate(source_files, start=1):
        relative_path = source_path.relative_to(source_root)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Le dtype original est conservé pour ne pas doubler artificiellement
        # la taille du dataset en enregistrant les signaux en float64.
        original = np.load(source_path)
        original_dtype = original.dtype

        if original.shape != (2, 32, 5120):
            raise ValueError(
                f"Forme inattendue dans {source_path}: {original.shape}."
            )

        bad_targets, contaminated_rows = find_bad_targets(
            source_path=source_path,
            threshold=threshold,
        )

        if bad_targets:
            corrected, interpolation_log = apply_regional_interpolation(
                data=original.astype(np.float64, copy=False),
                bad_targets=bad_targets,
                all_electrode_names=ELECTRODE_NAMES,
                electrode_regions=ELECTRODE_REGIONS,
                electrode_positions=ELECTRODE_POSITIONS,
            )
            corrected = corrected.astype(original_dtype, copy=False)
            verify_untargeted_channels_are_unchanged(
                original=original,
                corrected=corrected,
                bad_targets=bad_targets,
            )
        else:
            corrected = original.copy()
            interpolation_log = {}

        np.save(output_path, corrected)

        # Une ligne est enregistrée pour chaque canal effectivement ciblé.
        for row in contaminated_rows:
            participant_index = int(row["participant_index"])
            electrode_name = row["electrode"]
            channel_index = int(row["channel_index"])
            log_key = f"P{participant_index + 1}:{electrode_name}"
            corrected_channel = corrected[
                participant_index,
                channel_index,
                :,
            ]

            correction_rows.append({
                "dyad_id": row["dyad_id"],
                "filename": row["filename"],
                "participant": row["participant"],
                "participant_index": participant_index,
                "electrode": electrode_name,
                "channel_index": channel_index,
                "region": row["region"],
                "threshold": threshold,
                "outlier_values_before": row["n_above_threshold"],
                "outlier_values_after": int(
                    np.sum(np.abs(corrected_channel) > threshold)
                ),
                "values_replaced": corrected.shape[2],
                "neighbors": ";".join(
                    interpolation_log.get(log_key, [])
                ),
                "source_path": str(source_path),
                "output_path": str(output_path),
            })

        print(
            f"[{file_number:03d}/{len(source_files):03d}] "
            f"{relative_path} — {len(bad_targets)} correction(s)"
        )

    return pd.DataFrame(correction_rows)


def main() -> None:
    """Génère le dataset, le journal CSV et un résumé terminal."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    correction_log = generate_corrected_dataset(
        source_root=SOURCE_DATASET_ROOT,
        output_root=OUTPUT_DATASET_ROOT,
        threshold=ABSOLUTE_AMPLITUDE_THRESHOLD,
    )

    log_path = RESULTS_DIR / "regional_interpolation_log.csv"
    correction_log.to_csv(log_path, index=False)

    print("\n" + "=" * 70)
    print("GÉNÉRATION TERMINÉE")
    print("=" * 70)
    print(f"Dataset source  : {SOURCE_DATASET_ROOT}")
    print(f"Nouveau dataset : {OUTPUT_DATASET_ROOT}")
    print(f"Journal CSV     : {log_path}")
    print(f"Canaux corrigés : {len(correction_log)}")

    if not correction_log.empty:
        print(
            "Valeurs au-dessus du seuil avant : "
            f"{int(correction_log['outlier_values_before'].sum())}"
        )
        print(
            "Valeurs au-dessus du seuil après : "
            f"{int(correction_log['outlier_values_after'].sum())}"
        )
        print(
            "Valeurs remplacées au total      : "
            f"{int(correction_log['values_replaced'].sum())}"
        )


if __name__ == "__main__":
    main()
