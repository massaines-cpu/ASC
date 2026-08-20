"""Diagnostic étendu des amplitudes EEG — analyse par participant,
par électrode et par région cérébrale, avec interpolation spatiale régionale.

Ce script étend le diagnostic initial en ajoutant :
1. Analyse des outliers par participant (P1 / P2)
2. Identification des électrodes les plus fréquemment concernées
3. Vérification de la concentration des outliers par région cérébrale
4. Figures refaites participant par participant
5. Analyse globale : problème d'acquisition vs variabilité physiologique
6. Interpolation spatiale restreinte à la région cérébrale de l'électrode
7. Comparaison des stratégies d'interpolation
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "data_toy"
RESULTS_DIR = PROJECT_ROOT / "results" / "pretraitement_dataset_diagnostic_eeg_extended"

ABSOLUTE_AMPLITUDE_THRESHOLD = 0.01
RELATIVE_OUTLIER_THRESHOLD = 3.0

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Mapping électrodes → régions cérébrales (montage ASC à 32 électrodes)
# ------------------------------------------------------------------

# L'ordre doit rester strictement identique à l'axe des canaux dans les
# fichiers .npy. Une erreur d'ordre attribuerait les statistiques et les
# corrections à une mauvaise électrode.
#ordre réel des 32 électrodes
ELECTRODE_NAMES = [
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3",
    "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
    "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
    "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
]

# Chaque électrode est affectée à une région large afin que
# l'interpolation régionale n'utilise pas un canal provenant d'une zone
# anatomique différente.
#attribution de chaque électrode à une région
ELECTRODE_REGIONS = {
    # Frontal
    "Fp1": "frontal", "AF3": "frontal", "F7": "frontal",
    "F3": "frontal", "FC1": "frontal", "FC5": "frontal",
    "FC6": "frontal", "FC2": "frontal", "F4": "frontal",
    "F8": "frontal", "AF4": "frontal", "Fp2": "frontal",
    "Fz": "frontal",
    # Temporal
    "T7": "temporal", "T8": "temporal",
    # Central
    "C3": "central", "C4": "central", "Cz": "central",
    # Pariétal
    "CP1": "parietal", "CP5": "parietal", "P7": "parietal",
    "P3": "parietal", "Pz": "parietal", "P4": "parietal",
    "P8": "parietal", "CP6": "parietal", "CP2": "parietal",
    # Occipital
    "PO3": "occipital", "O1": "occipital", "Oz": "occipital",
    "O2": "occipital", "PO4": "occipital",
}

# Coordonnées 2D approximatives des électrodes (projection azimutale
# standard utilisée pour le topoplot). Unités arbitraires, cohérentes
# entre elles — seules les distances relatives comptent pour
# l'interpolation spatiale.
#coordonnées spatiales des électrodes
ELECTRODE_POSITIONS: dict[str, tuple[float, float]] = {
    "Fp1": (-0.22, 0.92), "AF3": (-0.30, 0.78),
    "F7": (-0.78, 0.55), "F3": (-0.40, 0.55),
    "FC1": (-0.22, 0.32), "FC5": (-0.62, 0.32),
    "T7": (-0.95, 0.00), "C3": (-0.48, 0.00),
    "CP1": (-0.22, -0.32), "CP5": (-0.62, -0.32),
    "P7": (-0.78, -0.55), "P3": (-0.40, -0.55),
    "Pz": (0.00, -0.62), "PO3": (-0.30, -0.78),
    "O1": (-0.22, -0.92), "Oz": (0.00, -0.98),
    "O2": (0.22, -0.92), "PO4": (0.30, -0.78),
    "P4": (0.40, -0.55), "P8": (0.78, -0.55),
    "CP6": (0.62, -0.32), "CP2": (0.22, -0.32),
    "C4": (0.48, 0.00), "T8": (0.95, 0.00),
    "FC6": (0.62, 0.32), "FC2": (0.22, 0.32),
    "F4": (0.40, 0.55), "F8": (0.78, 0.55),
    "AF4": (0.30, 0.78), "Fp2": (0.22, 0.92),
    "Fz": (0.00, 0.62), "Cz": (0.00, 0.00),
}


# ------------------------------------------------------------------
# Utilitaires de base (repris du script initial)
# ------------------------------------------------------------------

def find_epoch_files(dataset_root: Path) -> list[Path]:
    """Retourne tous les fichiers EEG .npy du dataset."""
    return sorted(dataset_root.glob("J*/epochs/*.npy"))


def load_eeg(npy_path: Path) -> np.ndarray:
    """Charge un fichier .npy en float64.

    Forme attendue : [participants, canaux, temps]
    """
    data = np.load(npy_path, mmap_mode="r")
    return np.asarray(data, dtype=np.float64)


# ------------------------------------------------------------------
# 1. Analyse par participant (P1 / P2)
# ------------------------------------------------------------------

def analyse_by_participant(
    npy_path: Path,
    threshold: float,
) -> list[dict]:
    """Calcule les statistiques d'amplitude pour chaque participant.

    Le tableau EEG a la forme [participants, canaux, temps].
    On itère sur l'axe 0 pour séparer P1 et P2.
    """
    data = load_eeg(npy_path)
    n_participants = data.shape[0]
    rows = []

    for p_idx in range(n_participants):
        participant_data = data[p_idx]                  # [canaux, temps]
        finite_values = participant_data[np.isfinite(participant_data)]
        abs_values = np.abs(finite_values)

        rows.append({
            "dyad_id":       npy_path.parents[1].name,
            "filename":      npy_path.name,
            "participant":   f"P{p_idx + 1}",
            "mean":          float(finite_values.mean()),
            "std":           float(finite_values.std()),
            "max_absolute":  float(abs_values.max()),
            "p95":           float(np.quantile(abs_values, 0.95)),
            "p99":           float(np.quantile(abs_values, 0.99)),
            "n_above_threshold": int(np.sum(abs_values > threshold)),
            "n_nan":         int(np.isnan(participant_data).sum()),
            "n_inf":         int(np.isinf(participant_data).sum()),
        })

    return rows


def build_participant_table(
    dataset_root: Path,
    threshold: float,
) -> pd.DataFrame:
    """Agrège les stats par participant sur l'ensemble du dataset."""
    rows = []
    for npy_path in find_epoch_files(dataset_root):
        rows.extend(analyse_by_participant(npy_path, threshold))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 2. Analyse par électrode
# ------------------------------------------------------------------
#détection des valeurs dépassant le seuil
def analyse_by_electrode(
    npy_path: Path,
    threshold: float,
    electrode_names: list[str] | None = None,
) -> list[dict]:
    """Calcule, pour chaque électrode, le nombre de valeurs hors seuil.

    Si electrode_names est fourni, les canaux sont nommés en conséquence.
    Sinon, on utilise l'indice numérique (canal_0, canal_1, …).
    """
    data = load_eeg(npy_path)                           # [participants, canaux, temps]
    n_channels = data.shape[1]

    if electrode_names is None:
        names = [f"canal_{i}" for i in range(n_channels)]
    else:
        if len(electrode_names) != n_channels:
            raise ValueError(
                "Le nombre de noms d'électrodes doit correspondre "
                f"au nombre de canaux : {len(electrode_names)} noms "
                f"pour {n_channels} canaux dans {npy_path.name}."
            )
        names = electrode_names

    rows = []
    for participant_index in range(data.shape[0]):
        for channel_index, electrode_name in enumerate(names):
            channel_data = data[participant_index, channel_index, :]
            finite_values = channel_data[np.isfinite(channel_data)]
            absolute_values = np.abs(finite_values)

            rows.append({
                "dyad_id": npy_path.parents[1].name,
                "filename": npy_path.name,
                "participant": f"P{participant_index + 1}",
                "participant_index": participant_index,
                "channel_index": channel_index,
                "electrode": electrode_name,
                "region": ELECTRODE_REGIONS.get(electrode_name, "unknown"),
                "max_absolute": (
                    float(absolute_values.max())
                    if absolute_values.size
                    else np.nan
                ),
                "n_above_threshold": int(
                    np.sum(absolute_values > threshold)
                ),
            })

    return rows


def build_electrode_table(
    dataset_root: Path,
    threshold: float,
    electrode_names: list[str] | None = None,
) -> pd.DataFrame:
    """Agrège les stats par électrode sur l'ensemble du dataset."""
    rows = []
    for npy_path in find_epoch_files(dataset_root):
        rows.extend(
            analyse_by_electrode(npy_path, threshold, electrode_names)
        )
    return pd.DataFrame(rows)


def summarise_outliers_by_electrode(
    electrode_df: pd.DataFrame,
) -> pd.DataFrame:
    """Classe les électrodes par fréquence d'apparition d'outliers."""
    return (
        electrode_df
        .groupby(["electrode", "region"], as_index=False)
        .agg(
            total_outlier_values=("n_above_threshold", "sum"),
            n_files_with_outliers=(
                "n_above_threshold",
                lambda v: int((v > 0).sum()),
            ),
            mean_max_absolute=("max_absolute", "mean"),
        )
        .sort_values("total_outlier_values", ascending=False)
        .reset_index(drop=True)
    )


def summarise_outliers_by_region(
    electrode_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Agrège les outliers par région cérébrale."""
    return (
        electrode_summary
        .groupby("region", as_index=False)
        .agg(
            total_outlier_values=("total_outlier_values", "sum"),
            n_electrodes=("electrode", "count"),
            mean_max_absolute=("mean_max_absolute", "mean"),
        )
        .sort_values("total_outlier_values", ascending=False)
        .reset_index(drop=True)
    )


# ------------------------------------------------------------------
# 3. Visualisations
# ------------------------------------------------------------------

def plot_participant_comparison(
    participant_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Boxplots comparant P1 et P2 par dyade pour l'amplitude maximale."""

    dyads = sorted(participant_df["dyad_id"].unique())
    participants = sorted(participant_df["participant"].unique())
    colors = {"P1": "#4C72B0", "P2": "#DD8452"}

    fig, ax = plt.subplots(figsize=(max(8, len(dyads) * 1.2), 6))

    n_p = len(participants)
    width = 0.35
    x = np.arange(len(dyads))

    for p_offset, p_label in enumerate(participants):
        subset = participant_df[participant_df["participant"] == p_label]
        means = [
            subset.loc[subset["dyad_id"] == d, "max_absolute"].mean()
            for d in dyads
        ]
        ax.bar(
            x + (p_offset - (n_p - 1) / 2) * width,
            means,
            width=width,
            label=p_label,
            color=colors.get(p_label, f"C{p_offset}"),
            alpha=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(dyads, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Amplitude absolue maximale moyenne (log)")
    ax.set_title("Comparaison P1 / P2 par dyade")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_electrode_outlier_heatmap(
    electrode_df: pd.DataFrame,
    output_path: Path,
    electrode_names: list[str],
) -> None:
    """Heatmap dyade × électrode du nombre d'outliers.

    Permet de repérer visuellement quelles électrodes sont
    problématiques et si le problème est localisé ou diffus.
    """
    pivot = (
        electrode_df
        .groupby(["dyad_id", "electrode"])["n_above_threshold"]
        .sum()
        .unstack(fill_value=0)
    )

    # Réordonne les colonnes selon le montage 10-20
    ordered_cols = [e for e in electrode_names if e in pivot.columns]
    pivot = pivot[ordered_cols]

    fig, ax = plt.subplots(
        figsize=(max(10, len(ordered_cols) * 0.6), max(4, len(pivot) * 0.5))
    )

    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Nombre de valeurs hors seuil")

    ax.set_xticks(range(len(ordered_cols)))
    ax.set_xticklabels(ordered_cols, rotation=90)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Outliers par électrode et par dyade")
    ax.set_xlabel("Électrode")
    ax.set_ylabel("Dyade")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_region_outlier_bar(
    region_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Barplot des outliers agrégés par région cérébrale."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        region_summary["region"],
        region_summary["total_outlier_values"],
        color="#4C72B0",
        alpha=0.8,
    )
    ax.set_xlabel("Total des valeurs hors seuil")
    ax.set_title("Concentration des outliers par région cérébrale")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_acquisition_vs_physiology(
    participant_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Aide à distinguer un problème d'acquisition d'une variabilité
    physiologique.

    Logique :
    - Si P1 ET P2 ont des outliers dans les mêmes fichiers → suspect
      d'un problème d'acquisition (bruit commun au dispositif).
    - Si seul P1 OU P2 est affecté → plutôt variabilité physiologique
      ou mauvais contact d'une électrode sur un seul participant.

    On trace le ratio outliers_P1 / (outliers_P1 + outliers_P2) par dyade.
    Un ratio proche de 0,5 suggère un problème partagé (acquisition) ;
    un ratio extrême (proche de 0 ou 1) pointe vers un participant précis.
    """
    pivot = (
        participant_df
        .groupby(["dyad_id", "participant"])["n_above_threshold"]
        .sum()
        .unstack(fill_value=0)
    )

    if "P1" not in pivot.columns or "P2" not in pivot.columns:
        return

    total = pivot["P1"] + pivot["P2"]
    # Évite la division par zéro quand aucun outlier n'est détecté.
    ratio = np.where(total > 0, pivot["P1"] / total, np.nan)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 0.8), 5))

    positions = np.arange(len(pivot.index))
    ax.bar(positions, ratio, color="#4C72B0", alpha=0.8)
    ax.axhline(0.5, linestyle="--", color="red",
               label="Ratio 0,5 = problème partagé (acquisition ?)")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Ratio outliers P1 / (P1 + P2)")
    ax.set_title(
        "Répartition des outliers entre participants\n"
        "Ratio ≈ 0,5 → problème d'acquisition | Ratio extrême → participant isolé"
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_electrode_positions(
    output_path: Path,
    highlighted_electrode: str | None = None,
    highlighted_neighbors: list[str] | None = None,
) -> None:
    """Affiche le montage 2D utilisé pour calculer les distances.

    Les coordonnées ne sont pas contenues dans les tableaux EEG : l'axe
    des canaux contient seulement 32 indices. Le dictionnaire
    ELECTRODE_POSITIONS apporte l'information spatiale nécessaire à
    l'interpolation. Si une électrode est indiquée, des segments montrent
    les voisins sélectionnés après le calcul des distances par ``cdist``.
    """

    highlighted_neighbors = highlighted_neighbors or []
    region_colors = {
        "frontal": "#4C72B0",
        "central": "#55A868",
        "temporal": "#C44E52",
        "parietal": "#8172B3",
        "occipital": "#CCB974",
    }

    fig, ax = plt.subplots(figsize=(9, 9))
    head = plt.Circle((0, 0), 1.05, fill=False, color="black", linewidth=2)
    ax.add_patch(head)

    for electrode_name in ELECTRODE_NAMES:
        x_position, y_position = ELECTRODE_POSITIONS[electrode_name]
        region = ELECTRODE_REGIONS[electrode_name]
        color = region_colors[region]
        point_size = 180 if electrode_name == highlighted_electrode else 90
        edge_color = "red" if electrode_name == highlighted_electrode else "black"

        ax.scatter(
            x_position,
            y_position,
            s=point_size,
            color=color,
            edgecolor=edge_color,
            linewidth=2 if electrode_name == highlighted_electrode else 0.8,
            zorder=3,
        )
        ax.text(
            x_position,
            y_position + 0.045,
            electrode_name,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if highlighted_electrode is not None:
        start_x, start_y = ELECTRODE_POSITIONS[highlighted_electrode]
        for neighbor_name in highlighted_neighbors:
            end_x, end_y = ELECTRODE_POSITIONS[neighbor_name]
            ax.plot(
                [start_x, end_x],
                [start_y, end_y],
                color="red",
                linestyle="--",
                linewidth=1.5,
                zorder=2,
            )

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_title("Positions 2D utilisées pour l'interpolation régionale")
    ax.set_xlabel("Axe gauche–droite (coordonnées relatives)")
    ax.set_ylabel("Axe arrière–avant (coordonnées relatives)")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# 4. Interpolation spatiale par région cérébrale
# ------------------------------------------------------------------

def get_regional_neighbors(
    bad_electrode: str,
    all_electrodes: list[str],
    electrode_regions: dict[str, str],
    electrode_positions: dict[str, tuple[float, float]],
    n_neighbors: int = 4,
    excluded_electrodes: set[str] | None = None,
) -> list[str]:
    """Retourne les N électrodes les plus proches dans la MÊME région.

    Contrainte clé : seules les électrodes de la même région anatomique
    que l'électrode à interpoler sont candidates. Cela évite d'introduire
    de l'information provenant de zones fonctionnellement différentes.

    Si la région ne contient pas assez de voisines valides, on prend
    toutes les disponibles plutôt que de croiser les régions.
    """
    if bad_electrode not in electrode_positions:
        return []

    excluded_electrodes = excluded_electrodes or set()
    bad_region = electrode_regions.get(bad_electrode, "unknown")
    bad_pos = np.array(electrode_positions[bad_electrode]).reshape(1, 2)

    # Candidats : même région, présents dans le dataset, pas l'électrode elle-même
    candidates = [
        e for e in all_electrodes
        if e != bad_electrode
        and e not in excluded_electrodes
        and electrode_regions.get(e) == bad_region
        and e in electrode_positions
    ]

    if not candidates:
        return []
#sélection des quatre voisins de la même région avec cdist
    candidate_positions = np.array([electrode_positions[e] for e in candidates])
    distances = cdist(bad_pos, candidate_positions, metric="euclidean").flatten()
#reconstruction pondérée du canal
    sorted_indices = np.argsort(distances)
    return [candidates[i] for i in sorted_indices[:n_neighbors]]


def interpolate_channel_regional(
    data: np.ndarray,
    participant_index: int,
    bad_ch_idx: int,
    neighbor_indices: list[int],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Remplace un canal par la moyenne pondérée de ses voisins régionaux.

    Paramètres
    ----------
    data : tableau [participants, canaux, temps]
    participant_index : indice du participant à corriger (0 pour P1, 1 pour P2)
    bad_ch_idx : indice du canal à interpoler
    neighbor_indices : indices des canaux voisins à utiliser
    weights : poids optionnels (inversement proportionnels à la distance
              par défaut si None)

    Retourne le tableau corrigé (copie, le tableau original n'est pas modifié).
    """
    corrected = data.copy()

    if participant_index < 0 or participant_index >= data.shape[0]:
        raise IndexError(
            f"Participant {participant_index} absent du tableau de forme {data.shape}."
        )

    if not neighbor_indices:
        # Pas de voisins disponibles : on laisse tel quel.
        return corrected

    # Seuls les voisins du participant contaminé sont utilisés. Le signal
    # de l'autre membre de la dyade ne doit pas intervenir dans la réparation.
    neighbor_data = data[participant_index, neighbor_indices, :]

    if weights is None:
        # Pondération uniforme si aucun poids fourni.
        weights = np.ones(len(neighbor_indices)) / len(neighbor_indices)
    else:
        weights = np.array(weights)
        weights /= weights.sum()

    # Produit pondéré : [voisins, temps] × [voisins] → [temps].
    interpolated = np.einsum("ct,c->t", neighbor_data, weights)

    # La correction cible uniquement le participant réellement contaminé.
    # Par exemple, participant_index=0 modifie P1 et conserve P2 intact.
    corrected[participant_index, bad_ch_idx, :] = interpolated
    return corrected

#application de la correction au participant et au canal ciblés

def apply_regional_interpolation(
    data: np.ndarray,
    bad_targets: list[tuple[int, str]],
    all_electrode_names: list[str],
    electrode_regions: dict[str, str],
    electrode_positions: dict[str, tuple[float, float]],
    n_neighbors: int = 4,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Applique l'interpolation régionale sur toutes les électrodes
    mauvaises détectées.

    Retourne le tableau corrigé et un dictionnaire
    {électrode → liste des voisins utilisés} pour la traçabilité.
    """
    corrected = data.copy()
    interpolation_log: dict[str, list[str]] = {}

    for participant_index, bad_name in bad_targets:
        if bad_name not in all_electrode_names:
            continue

        log_key = f"P{participant_index + 1}:{bad_name}"
        bad_names_for_participant = {
            electrode_name
            for target_participant, electrode_name in bad_targets
            if target_participant == participant_index
        }

        bad_idx = all_electrode_names.index(bad_name)
        neighbors = get_regional_neighbors(
            bad_electrode=bad_name,
            all_electrodes=all_electrode_names,
            electrode_regions=electrode_regions,
            electrode_positions=electrode_positions,
            n_neighbors=n_neighbors,
            excluded_electrodes=bad_names_for_participant,
        )

        if not neighbors:
            interpolation_log[log_key] = []
            continue

        neighbor_indices = [all_electrode_names.index(n) for n in neighbors]

        # Poids inversement proportionnels à la distance.
        bad_pos = np.array(electrode_positions[bad_name])
        distances = np.array([
            np.linalg.norm(bad_pos - np.array(electrode_positions[n]))
            for n in neighbors
        ])
        # On ajoute un epsilon pour éviter la division par zéro si
        # deux électrodes sont à la même position (cas théorique).
        inv_dist_weights = 1.0 / (distances + 1e-10)

        # Chaque canal est reconstruit depuis les données originales afin
        # qu'une correction précédente ne serve pas à en calculer une autre.
        one_channel_correction = interpolate_channel_regional(
            data=data,
            participant_index=participant_index,
            bad_ch_idx=bad_idx,
            neighbor_indices=neighbor_indices,
            weights=inv_dist_weights,
        )
        corrected[participant_index, bad_idx, :] = one_channel_correction[
            participant_index,
            bad_idx,
            :,
        ]
        interpolation_log[log_key] = neighbors

    return corrected, interpolation_log


# ------------------------------------------------------------------
# 5. Comparaison des stratégies d'interpolation
# ------------------------------------------------------------------

def compare_interpolation_strategies(
    data: np.ndarray,
    bad_targets: list[tuple[int, str]],
    all_electrode_names: list[str],
    electrode_regions: dict[str, str],
    electrode_positions: dict[str, tuple[float, float]],
    threshold: float,
) -> pd.DataFrame:
    """Compare trois stratégies d'interpolation.

    Stratégies comparées
    --------------------
    1. Aucune correction (données brutes)
    2. Interpolation globale : N voisins les plus proches sans contrainte
       de région (comportement historique)
    3. Interpolation régionale : N voisins les plus proches dans la même
       région anatomique (nouvelle stratégie)

    La métrique de comparaison est le nombre de valeurs dépassant le seuil
    absolu après correction, ce qui reflète directement l'efficacité.
    """
    results = []

    def count_outliers(arr: np.ndarray) -> int:
        return int(np.sum(np.abs(arr[np.isfinite(arr)]) > threshold))

    # Stratégie 1 : pas de correction
    results.append({
        "stratégie": "1_aucune_correction",
        "outliers_restants": count_outliers(data),
    })

    # Stratégie 2 : interpolation globale (sans contrainte régionale)
    data_global = data.copy()
    for participant_index, bad_name in bad_targets:
        if bad_name not in all_electrode_names:
            continue
        bad_idx = all_electrode_names.index(bad_name)

        # Tous les autres canaux comme candidats, sans filtre de région
        all_others = [
            e for e in all_electrode_names
            if e != bad_name and e in electrode_positions
        ]
        bad_position = np.array(
            electrode_positions[bad_name]
        ).reshape(1, 2)
        other_positions = np.array([
            electrode_positions[name]
            for name in all_others
        ])
        distances = cdist(
            bad_position,
            other_positions,
            metric="euclidean",
        ).flatten()
        nearest_names = [
            all_others[index]
            for index in np.argsort(distances)[:4]
        ]
        neighbor_indices = [
            all_electrode_names.index(name)
            for name in nearest_names
        ]
        data_global = interpolate_channel_regional(
            data=data_global,
            participant_index=participant_index,
            bad_ch_idx=bad_idx,
            neighbor_indices=neighbor_indices,
        )

    results.append({
        "stratégie": "2_interpolation_globale",
        "outliers_restants": count_outliers(data_global),
    })

    # Stratégie 3 : interpolation régionale (nouvelle)
    data_regional, _ = apply_regional_interpolation(
        data=data,
        bad_targets=bad_targets,
        all_electrode_names=all_electrode_names,
        electrode_regions=electrode_regions,
        electrode_positions=electrode_positions,
    )

    results.append({
        "stratégie": "3_interpolation_régionale",
        "outliers_restants": count_outliers(data_regional),
    })

    return pd.DataFrame(results)


# ------------------------------------------------------------------
# Programme principal
# ------------------------------------------------------------------

def main() -> None:
    """Lance l'analyse étendue et sauvegarde tous les résultats."""

    npy_files = find_epoch_files(DATASET_ROOT)
    print(f"{len(npy_files)} fichiers trouvés dans {DATASET_ROOT}\n")

    # ---- 1. Analyse par participant ----
    print("Analyse par participant (P1 / P2)…")
    participant_df = build_participant_table(DATASET_ROOT, ABSOLUTE_AMPLITUDE_THRESHOLD)
    participant_df.to_csv(RESULTS_DIR / "diagnostic_by_participant.csv", index=False)

    plot_participant_comparison(
        participant_df,
        RESULTS_DIR / "amplitudes_by_participant.png",
    )
    plot_acquisition_vs_physiology(
        participant_df,
        RESULTS_DIR / "acquisition_vs_physiology.png",
    )

    # ---- 2. Analyse par électrode et par région ----
    print("Analyse par électrode…")
    electrode_df = build_electrode_table(
        DATASET_ROOT, ABSOLUTE_AMPLITUDE_THRESHOLD, ELECTRODE_NAMES
    )
    electrode_summary = summarise_outliers_by_electrode(electrode_df)
    region_summary = summarise_outliers_by_region(electrode_summary)

    electrode_df.to_csv(RESULTS_DIR / "diagnostic_by_electrode.csv", index=False)
    electrode_summary.to_csv(RESULTS_DIR / "electrode_outlier_summary.csv", index=False)
    region_summary.to_csv(RESULTS_DIR / "region_outlier_summary.csv", index=False)

    plot_electrode_outlier_heatmap(
        electrode_df,
        RESULTS_DIR / "heatmap_electrodes_outliers.png",
        ELECTRODE_NAMES,
    )
    plot_region_outlier_bar(
        region_summary,
        RESULTS_DIR / "outliers_by_region.png",
    )

    # ---- 3. Comparaison des stratégies d'interpolation ----
    # On applique la comparaison sur le premier fichier disponible à titre
    # d'exemple. En production, boucler sur l'ensemble du dataset.
    if npy_files:
        print("Comparaison des stratégies d'interpolation…")
        # On choisit le premier fichier réellement contaminé, plutôt que
        # le premier fichier alphabétique du dataset.
        contaminated_rows = electrode_df[
            electrode_df["n_above_threshold"] > 0
        ]

        if not contaminated_rows.empty:
            first_contaminated = contaminated_rows.iloc[0]
            sample_path = next(
                path
                for path in npy_files
                if path.name == first_contaminated["filename"]
                and path.parents[1].name == first_contaminated["dyad_id"]
            )
            sample_data = load_eeg(sample_path)
            file_rows = contaminated_rows[
                (contaminated_rows["dyad_id"] == first_contaminated["dyad_id"])
                & (contaminated_rows["filename"] == first_contaminated["filename"])
            ]
            bad_targets = [
                (int(row.participant_index), row.electrode)
                for row in file_rows.itertuples()
            ]

            comparison_df = compare_interpolation_strategies(
                data=sample_data,
                bad_targets=bad_targets,
                all_electrode_names=ELECTRODE_NAMES,
                electrode_regions=ELECTRODE_REGIONS,
                electrode_positions=ELECTRODE_POSITIONS,
                threshold=ABSOLUTE_AMPLITUDE_THRESHOLD,
            )
            comparison_df.to_csv(
                RESULTS_DIR / "interpolation_comparison.csv", index=False
            )
            print("\nComparaison des stratégies :")
            print(comparison_df.to_string(index=False))

            highlighted_electrode = bad_targets[0][1]
            highlighted_neighbors = get_regional_neighbors(
                bad_electrode=highlighted_electrode,
                all_electrodes=ELECTRODE_NAMES,
                electrode_regions=ELECTRODE_REGIONS,
                electrode_positions=ELECTRODE_POSITIONS,
            )
            plot_electrode_positions(
                output_path=RESULTS_DIR / "electrode_positions_and_neighbors.png",
                highlighted_electrode=highlighted_electrode,
                highlighted_neighbors=highlighted_neighbors,
            )
        else:
            print("Aucune électrode mauvaise détectée dans le dataset.")

    # ---- Résumé terminal ----
    print("\n" + "=" * 70)
    print("RÉSUMÉ PAR PARTICIPANT")
    print("=" * 70)
    participant_summary = (
        participant_df
        .groupby(["dyad_id", "participant"])["n_above_threshold"]
        .sum()
        .unstack(fill_value=0)
    )
    print(participant_summary.to_string())

    print("\n" + "=" * 70)
    print("TOP 5 ÉLECTRODES LES PLUS PROBLÉMATIQUES")
    print("=" * 70)
    print(electrode_summary.head(5).to_string(index=False))

    print("\n" + "=" * 70)
    print("OUTLIERS PAR RÉGION CÉRÉBRALE")
    print("=" * 70)
    print(region_summary.to_string(index=False))

    print(f"\nRésultats sauvegardés dans : {RESULTS_DIR}")


if __name__ == "__main__":
    main()
