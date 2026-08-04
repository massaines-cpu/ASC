"""Crée deux expériences de réparation des fichiers atypiques par MAD.

Expérience B
-------------
Les fichiers sont sélectionnés par le score MAD global déjà utilisé dans
``inspect_dataset.py``. La frontière d'amplitude correspondant au score 3 est
ensuite utilisée pour localiser les points à interpoler temporellement.

Expérience C
-------------
Les mêmes fichiers sont sélectionnés. Dans chacun, le participant et le canal
portant le maximum absolu sont identifiés. Le canal entier est remplacé par la
médiane temporelle des 31 autres canaux, selon le principe déjà testé sur
J7 E018. Cette méthode est appelée ``remplacement inter-canaux`` et non
``interpolation spatiale``, car le montage et les coordonnées des électrodes
ne sont pas disponibles dans le projet.

Les données originales ne sont jamais modifiées. Chaque expérience produit
son propre dataset, son journal et un diagnostic après réparation.
"""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.dataset.inspect_dataset import (
    PROJECT_ROOT,
    analyse_dataset,
    flag_relative_outliers,
)


SOURCE_ROOT = PROJECT_ROOT / "data" / "data_toy"
EXPERIMENT_B_ROOT = (
    PROJECT_ROOT / "data" / "data_experiment_C_mad_temporal"
)
EXPERIMENT_C_ROOT = (
    PROJECT_ROOT / "data" / "data_experiment_D_cross_channel_median"
)
RESULTS_ROOT = PROJECT_ROOT / "results" / "mad_repair_experiments"

MAD_SCORE_THRESHOLD = 3.0
EXPECTED_EEG_SHAPE = (2, 32, 5120)

def compute_mad_selection(
    source_root: Path,
    score_threshold: float,
) -> tuple[pd.DataFrame, float]:
    """Sélectionne les fichiers et retourne la frontière d'amplitude MAD."""

    file_results = analyse_dataset(
        dataset_root=source_root,
        threshold=float("inf"),
    )
    file_results = flag_relative_outliers(
        file_results,
        k=score_threshold,
    )

    selected_files = file_results[
        file_results["relative_amplitude_score"] > score_threshold
    ].copy()
    if selected_files.empty:
        raise RuntimeError("Aucun fichier n'a dépassé le score MAD choisi.")

    log_amplitudes = np.log10(
        file_results["max_absolute"].where(
            file_results["max_absolute"] > 0
        )
    )
    median_log = log_amplitudes.median()
    scaled_mad_log = (
        (log_amplitudes - median_log).abs().median()
        * 1.4826
    )
    amplitude_boundary = float(
        10 ** (
            median_log
            + score_threshold * scaled_mad_log
        )
    )

    return selected_files, amplitude_boundary


def validate_new_output_directory(output_root: Path) -> None:
    """Empêche l'écrasement d'un dataset ou d'une expérience précédente."""

    if output_root.exists():
        raise FileExistsError(
            f"Le dossier existe déjà : {output_root}. "
            "Conserve-le comme résultat ou choisis un nouveau nom."
        )


def interpolate_masked_points(
    channel_signal: np.ndarray,
    abnormal_mask: np.ndarray,
) -> np.ndarray:
    """Interpole les points masqués le long de l'axe temporel."""

    repaired_signal = channel_signal.astype(np.float64, copy=True)
    valid_mask = ~abnormal_mask & np.isfinite(repaired_signal)

    if int(valid_mask.sum()) < 2:
        raise ValueError(
            "Interpolation temporelle impossible : moins de deux points "
            "valides dans le canal."
        )

    time_indices = np.arange(repaired_signal.size)
    repaired_signal[abnormal_mask] = np.interp(
        time_indices[abnormal_mask],
        time_indices[valid_mask],
        repaired_signal[valid_mask],
    )
    return repaired_signal


def repair_file_temporally(
    data: np.ndarray,
    amplitude_boundary: float,
) -> tuple[np.ndarray, list[dict]]:
    """Répare les points franchissant la frontière MAD globale."""

    if data.shape != EXPECTED_EEG_SHAPE:
        raise ValueError(
            f"Forme inattendue : {data.shape}; "
            f"forme attendue : {EXPECTED_EEG_SHAPE}."
        )

    repaired_data = data.astype(np.float64, copy=True)
    details = []

    for participant_index in range(data.shape[0]):
        for channel_index in range(data.shape[1]):
            channel_signal = data[participant_index, channel_index]
            abnormal_mask = (
                (np.abs(channel_signal) > amplitude_boundary)
                | (~np.isfinite(channel_signal))
            )
            number_of_points = int(abnormal_mask.sum())
            if number_of_points == 0:
                continue

            repaired_data[participant_index, channel_index] = (
                interpolate_masked_points(
                    channel_signal=channel_signal,
                    abnormal_mask=abnormal_mask,
                )
            )
            abnormal_times = np.flatnonzero(abnormal_mask)
            details.append(
                {
                    "participant_index": participant_index,
                    "channel_index": channel_index,
                    "number_of_interpolated_points": number_of_points,
                    "first_time_index": int(abnormal_times.min()),
                    "last_time_index": int(abnormal_times.max()),
                }
            )

    return repaired_data.astype(data.dtype), details


def repair_maximum_channel(
    data: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Remplace le canal portant le maximum par la médiane des autres."""

    if data.shape != EXPECTED_EEG_SHAPE:
        raise ValueError(
            f"Forme inattendue : {data.shape}; "
            f"forme attendue : {EXPECTED_EEG_SHAPE}."
        )

    repaired_data = data.copy()
    maximum_flat_index = int(np.nanargmax(np.abs(data)))
    participant_index, channel_index, maximum_time_index = np.unravel_index(
        maximum_flat_index,
        data.shape,
    )
    original_channel = data[participant_index, channel_index]
    other_channels = np.delete(
        data[participant_index],
        channel_index,
        axis=0,
    )
    replacement_channel = np.nanmedian(other_channels, axis=0)

    if not np.isfinite(replacement_channel).all():
        raise ValueError(
            "Le signal de remplacement contient encore des valeurs non finies."
        )

    repaired_data[participant_index, channel_index] = replacement_channel

    details = {
        "participant_index": int(participant_index),
        "channel_index": int(channel_index),
        "maximum_time_index": int(maximum_time_index),
        "original_channel_max_absolute": float(
            np.nanmax(np.abs(original_channel))
        ),
        "replacement_channel_max_absolute": float(
            np.max(np.abs(replacement_channel))
        ),
        "number_of_replaced_points": int(original_channel.size),
    }
    return repaired_data, details


def prepare_repairs_before_copy(
    selected_files: pd.DataFrame,
    source_root: Path,
    amplitude_boundary: float,
) -> tuple[dict[tuple[str, str], np.ndarray], list[dict], dict, list[dict]]:
    """Calcule toutes les réparations avant de créer les deux copies."""

    temporal_repairs = {}
    temporal_log = []
    channel_repairs = {}
    channel_log = []

    for _, file_row in selected_files.iterrows():
        dyad_id = file_row["dyad_id"]
        filename = file_row["filename"]
        source_path = source_root / dyad_id / "epochs" / filename
        data = np.load(source_path)

        repaired_temporal, temporal_details = repair_file_temporally(
            data=data,
            amplitude_boundary=amplitude_boundary,
        )
        repaired_channel, channel_details = repair_maximum_channel(data)

        file_key = (dyad_id, filename)
        temporal_repairs[file_key] = repaired_temporal
        channel_repairs[file_key] = repaired_channel

        for detail in temporal_details:
            temporal_log.append(
                {
                    "dyad_id": dyad_id,
                    "filename": filename,
                    "file_mad_score": file_row["relative_amplitude_score"],
                    "amplitude_boundary": amplitude_boundary,
                    **detail,
                }
            )
        channel_log.append(
            {
                "dyad_id": dyad_id,
                "filename": filename,
                "file_mad_score": file_row["relative_amplitude_score"],
                **channel_details,
            }
        )

    return temporal_repairs, temporal_log, channel_repairs, channel_log


def create_experiment_dataset(
    source_root: Path,
    output_root: Path,
    repaired_files: dict[tuple[str, str], np.ndarray],
) -> None:
    """Copie le dataset puis remplace seulement les fichiers expérimentaux."""

    shutil.copytree(source_root, output_root)
    for (dyad_id, filename), repaired_data in repaired_files.items():
        output_path = output_root / dyad_id / "epochs" / filename
        np.save(output_path, repaired_data)


def save_experiment_diagnostic(
    dataset_root: Path,
    output_path: Path,
) -> None:
    """Sauvegarde le diagnostic MAD complet après une réparation."""

    results = analyse_dataset(
        dataset_root=dataset_root,
        threshold=float("inf"),
    )
    results = flag_relative_outliers(
        results,
        k=MAD_SCORE_THRESHOLD,
    )
    results.to_csv(output_path, index=False)


def main() -> None:
    """Construit uniquement l'expérience C."""

    validate_new_output_directory(EXPERIMENT_C_ROOT)

    selected_files, amplitude_boundary = compute_mad_selection(
        source_root=SOURCE_ROOT,
        score_threshold=MAD_SCORE_THRESHOLD,
    )

    print(
        f"{len(selected_files)} fichiers sélectionnés "
        f"avec un score MAD > {MAD_SCORE_THRESHOLD}."
    )

    channel_repairs = {}
    channel_log = []

    for _, file_row in selected_files.iterrows():
        dyad_id = file_row["dyad_id"]
        filename = file_row["filename"]

        source_path = (
            SOURCE_ROOT
            / dyad_id
            / "epochs"
            / filename
        )

        data = np.load(source_path)

        repaired_data, repair_details = repair_maximum_channel(
            data=data,
        )

        file_key = (dyad_id, filename)
        channel_repairs[file_key] = repaired_data

        channel_log.append(
            {
                "dyad_id": dyad_id,
                "filename": filename,
                "file_mad_score": float(
                    file_row["relative_amplitude_score"]
                ),
                **repair_details,
            }
        )

    create_experiment_dataset(
        source_root=SOURCE_ROOT,
        output_root=EXPERIMENT_C_ROOT,
        repaired_files=channel_repairs,
    )

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_files.to_csv(
        RESULTS_ROOT / "selected_files.csv",
        index=False,
    )

    pd.DataFrame(channel_log).to_csv(
        RESULTS_ROOT / "experiment_C_channel_log.csv",
        index=False,
    )

    save_experiment_diagnostic(
        dataset_root=EXPERIMENT_C_ROOT,
        output_path=(
            RESULTS_ROOT
            / "experiment_C_diagnostic.csv"
        ),
    )

    print(f"Expérience C créée : {EXPERIMENT_C_ROOT}")
    print(f"Résultats : {RESULTS_ROOT}")