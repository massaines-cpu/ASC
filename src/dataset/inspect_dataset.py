"""Diagnostic des amplitudes EEG sur l'ensemble des dyades.

Le script :
1. parcourt tous les fichiers .npy de toutes les dyades
2. vérifie les formes, NaN et valeurs infinies
3. calcule des statistiques d'amplitude par fichier
4. repère les fichiers dépassant un seuil absolu exploratoire
5. repère aussi les fichiers anormaux PAR RAPPORT AU RESTE DU DATASET,
   même s'ils restent sous le seuil absolu
6. produit deux tableaux CSV :
   - un tableau détaillé par fichier
   - un résumé par dyade.

Important
---------
Le seuil absolu n'est interprétable que si l'unité des données est connue.
Un seuil absolu fixe peut aussi rater des anomalies plus modérées mais
réelles (ex: un cluster de fichiers 10-40x plus grands que la norme du
dataset, sans qu'aucun ne dépasse individuellement le seuil). Le critère
relatif ci-dessous comble ce manque.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "data_toy"
RESULTS_DIR = PROJECT_ROOT / "results" / "pretraitement_dataset_diagnostic_2"

# Seuil exploratoire utilisé pour repérer les très grandes amplitudes.
# Il ne constitue pas encore un seuil universel de rejet EEG.
ABSOLUTE_AMPLITUDE_THRESHOLD = 0.01
# Nombre de MAD (median absolute deviation) au-delà duquel un fichier
# est considéré comme anormal PAR RAPPORT AU RESTE DU DATASET, même
# s'il ne dépasse pas le seuil absolu. 3 correspond approximativement
# à 3 écarts-types dans une distribution gaussienne.
RELATIVE_OUTLIER_THRESHOLD = 3.0

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Recherche des fichiers
# ------------------------------------------------------------------

def find_epoch_files(dataset_root: Path) -> list[Path]:
    """Retourne tous les fichiers EEG .npy du dataset."""

    return sorted(dataset_root.glob("J*/epochs/*.npy"))


# ------------------------------------------------------------------
# Diagnostic d'un seul fichier
# ------------------------------------------------------------------

def analyse_eeg_file(npy_path: Path, threshold: float) -> dict:
    """Calcule les principales statistiques d'un fichier EEG.

    La forme attendue est :
        [participants, canaux, temps]

    Le calcul des quantiles permet de décrire la distribution sans se
    limiter au minimum et au maximum, qui peuvent dépendre d'un seul pic.
    """

    data = np.load(npy_path, mmap_mode="r")
    data = np.asarray(data, dtype=np.float64)

    finite_mask = np.isfinite(data)
    finite_values = data[finite_mask]

    n_nan = int(np.isnan(data).sum())
    n_inf = int(np.isinf(data).sum())

    absolute_values = np.abs(finite_values)

    return {
        "dyad_id": npy_path.parents[1].name,
        "filename": npy_path.name,
        "shape": str(data.shape),

        # Statistiques générales
        "moyenne": float(finite_values.mean()),
        "écart-type": float(finite_values.std()),
        "minimum": float(finite_values.min()),
        "maximum": float(finite_values.max()),

        # Étendue totale du signal
        "peak_to_peak": float(
            finite_values.max() - finite_values.min()
        ),

        # Distribution des amplitudes absolues
        "max_absolute": float(absolute_values.max()),
        "95_percentile": float(np.quantile(absolute_values, 0.95)),
        "99_percentile": float(np.quantile(absolute_values, 0.99)),
        "99_9_percentile": float(np.quantile(absolute_values, 0.999)),

        # Contrôles de qualité
        "n_above_threshold": int(
            np.sum(absolute_values > threshold)
        ),
        "n_nan": n_nan,
        "n_inf": n_inf,
    }


# ------------------------------------------------------------------
# Diagnostic de tout le dataset
# ------------------------------------------------------------------

def analyse_dataset(
    dataset_root: Path,
    threshold: float,
) -> pd.DataFrame:
    """Analyse tous les fichiers EEG et retourne un tableau détaillé."""

    npy_files = find_epoch_files(dataset_root)

    rows = [
        analyse_eeg_file(
            npy_path=npy_path,
            threshold=threshold,
        )
        for npy_path in npy_files
    ]

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Détection relative (nouveau) : anormal par rapport au reste du dataset
# ------------------------------------------------------------------

def flag_relative_outliers(
    file_results: pd.DataFrame,
    k: float = RELATIVE_OUTLIER_THRESHOLD,
) -> pd.DataFrame:
    """Signale les fichiers dont l'amplitude est anormalement grande
    PAR RAPPORT À LA DISTRIBUTION DU DATASET ENTIER, indépendamment
    du seuil absolu.

    Méthode :
    - on travaille en échelle log10, car les amplitudes EEG varient sur
      plusieurs ordres de grandeur (comparer les valeurs brutes n'aurait
      pas de sens)
    - on utilise la médiane et la MAD (median absolute deviation) plutôt
      que la moyenne et l'écart-type classiques, car la médiane et la
      MAD restent fiables même quand quelques fichiers sont déjà des
      valeurs extrêmes (elles ne "tirent" pas le seuil vers le haut,
      contrairement à la moyenne/écart-type).
    """

    file_results = file_results.copy()

    log_max_absolute = np.log10(file_results["max_absolute"])

    median_log = log_max_absolute.median()
    mad_log = (log_max_absolute - median_log).abs().median()

    # 1.4826 rend la MAD comparable à un écart-type dans le cas d'une
    # distribution approximativement gaussienne (facteur de normalisation
    # standard pour la MAD).
    scaled_mad = mad_log * 1.4826

    # Évite une division par zéro si toutes les valeurs sont identiques.
    if scaled_mad == 0:
        relative_score = pd.Series(0.0, index=file_results.index)
    else:
        relative_score = (log_max_absolute - median_log) / scaled_mad

    file_results["relative_amplitude_score"] = relative_score
    file_results["is_outlier_relative"] = relative_score > k

    return file_results


# ------------------------------------------------------------------
# Résumé par dyade
# ------------------------------------------------------------------

def create_dyad_summary(
    file_results: pd.DataFrame,
) -> pd.DataFrame:
    """Agrège les résultats des fichiers pour comparer les dyades."""

    return (
        file_results
        .groupby("dyad_id", as_index=False)
        .agg(
            n_files=("filename", "count"),
            mean_std=("écart-type", "mean"),
            maximum_absolute=("max_absolute", "max"),
            maximum_peak_to_peak=("peak_to_peak", "max"),
            median_q99_absolute=("99_percentile", "median"),
            n_extreme_files=(
                "n_above_threshold",
                lambda values: int((values > 0).sum()),
            ),
            total_extreme_values=(
                "n_above_threshold",
                "sum",
            ),
            n_relative_outliers=(
                "is_outlier_relative",
                "sum",
            ),
            total_nan=("n_nan", "sum"),
            total_inf=("n_inf", "sum"),
        )
        .sort_values("maximum_absolute", ascending=False)
        .reset_index(drop=True)
    )


# ------------------------------------------------------------------
# Visualisation de la distribution entre dyades
# ------------------------------------------------------------------

def save_amplitude_plot(
    file_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Trace l'amplitude absolue maximale de chaque fichier par dyade."""

    dyad_order = sorted(file_results["dyad_id"].unique())

    values = [
        file_results.loc[
            file_results["dyad_id"] == dyad_id,
            "max_absolute",
        ].to_numpy()
        for dyad_id in dyad_order
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(
        values,
        tick_labels=dyad_order,
        showfliers=True,
    )

    ax.axhline(
        ABSOLUTE_AMPLITUDE_THRESHOLD,
        linestyle="--",
        label=(
            "Seuil absolu exploratoire "
            f"({ABSOLUTE_AMPLITUDE_THRESHOLD})"
        ),
    )

    ax.set_yscale("log")
    ax.set_title(
        "Distribution des amplitudes absolues maximales par dyade"
    )
    ax.set_xlabel("Dyade")
    ax.set_ylabel("Amplitude absolue maximale (échelle log)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# Programme principal
# ------------------------------------------------------------------

def main() -> None:
    """Lance le diagnostic et sauvegarde les résultats."""

    file_results = analyse_dataset(
        dataset_root=DATASET_ROOT,
        threshold=ABSOLUTE_AMPLITUDE_THRESHOLD,
    )

    # Ajout du critère relatif, en complément du seuil absolu.
    file_results = flag_relative_outliers(
        file_results,
        k=RELATIVE_OUTLIER_THRESHOLD,
    )

    dyad_summary = create_dyad_summary(file_results)

    file_results.to_csv(
        RESULTS_DIR / "diagnostic_by_file.csv",
        index=False,
    )

    dyad_summary.to_csv(
        RESULTS_DIR / "diagnostic_by_dyad.csv",
        index=False,
    )

    save_amplitude_plot(
        file_results=file_results,
        output_path=RESULTS_DIR / "amplitudes_by_dyad.png",
    )

    # Un fichier est suspect s'il dépasse le seuil ABSOLU
    # OU s'il est anormal par rapport à la distribution RELATIVE
    # du dataset (ce second critère rattrape les clusters modérés
    # qui restent sous le seuil absolu).
    suspicious_files = file_results[
        (file_results["n_above_threshold"] > 0)
        | (file_results["is_outlier_relative"])
    ]

    print("\n" + "=" * 70)
    print("RÉSUMÉ PAR DYADE")
    print("=" * 70)
    print(dyad_summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("FICHIERS SUSPECTS (seuil absolu OU anomalie relative)")
    print("=" * 70)

    if suspicious_files.empty:
        print("Aucun fichier suspect détecté.")
    else:
        columns = [
            "dyad_id",
            "filename",
            "minimum",
            "maximum",
            "max_absolute",
            "n_above_threshold",
            "relative_amplitude_score",
            "is_outlier_relative",
        ]

        print(
            suspicious_files[columns]
            .sort_values("relative_amplitude_score", ascending=False)
            .to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}"
                if abs(value) < 1000
                else f"{value:.4e}",
            )
        )

    print(f"\nRésultats sauvegardés dans : {RESULTS_DIR}")


if __name__ == "__main__":
    main()