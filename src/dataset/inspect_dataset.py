"""Inspection du dataset EEG (dyades, fichiers .npy, manifests).

Ce script explore le contenu de data/data_toy pour vérifier :
- la présence et la cohérence des fichiers de chaque dyade ;
- la forme et le type des signaux EEG (.npy) ;
- le contenu des manifests (labels, conditions, fréquences...) ;
- quelques statistiques globales sur les signaux.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "data_toy"


def find_dyad_directories(dataset_root):
    """Retourne tous les dossiers de dyades (J1, J2, J4, ...), triés."""
    return sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name.startswith("J")
    )


def inspect_npy_file(npy_path):
    """Lit les métadonnées d'un fichier .npy sans le charger en mémoire.

    mmap_mode="r" ouvre le fichier en lecture "à la demande" : seules
    les métadonnées (shape, dtype) sont lues ici, pas les données elles-mêmes.
    """
    data = np.load(npy_path, mmap_mode="r")
    return {
        "shape": data.shape,
        "dtype": str(data.dtype),
        "n_participants": data.shape[0] if data.ndim >= 1 else None,
        "n_channels": data.shape[1] if data.ndim >= 2 else None,
        "n_times": data.shape[2] if data.ndim >= 3 else None,
    }


def inspect_manifest(manifest_path):
    """Lit le manifest d'une dyade et en extrait les infos clés."""
    manifest = pd.read_csv(manifest_path)

    def unique_values(column):
        return (
            sorted(manifest[column].dropna().unique().tolist())
            if column in manifest.columns
            else []
        )

    summary = {
        "n_rows": len(manifest),
        "columns": list(manifest.columns),
        "eyes_codes": unique_values("eyes_code"),
        "conditions": unique_values("condition_code"),
        "shapes": unique_values("matrix_shape"),
        "sampling_frequencies": unique_values("sampling_frequency_hz"),
    }
    return manifest, summary


def inspect_signal_statistics(npy_files):
    """Calcule des statistiques globales sur tous les fichiers .npy.

    Les fichiers sont parcourus un par un afin d'éviter de concaténer
    toutes les données EEG dans un seul énorme tableau.

    Statistiques calculées :
    - moyenne globale ;
    - écart-type global ;
    - minimum global ;
    - maximum global ;
    - nombre total de valeurs ;
    - nombre de valeurs non finies : NaN ou inf.
    """

    total_sum = 0.0
    total_squared_sum = 0.0
    total_values = 0

    global_min = np.inf
    global_max = -np.inf

    n_nan = 0
    n_inf = 0

    for npy_file in npy_files:
        # mmap_mode permet de ne pas charger immédiatement tout le fichier.
        data = np.load(npy_file, mmap_mode="r")

        # Conversion en float64 pour rendre les sommes plus précises.
        data_float = np.asarray(data, dtype=np.float64)

        # Vérification des valeurs anormales.
        n_nan += np.isnan(data_float).sum()
        n_inf += np.isinf(data_float).sum()

        # On conserve seulement les valeurs finies pour les statistiques.
        finite_values = data_float[np.isfinite(data_float)]

        if finite_values.size == 0:
            continue

        total_sum += finite_values.sum()
        total_squared_sum += np.square(finite_values).sum()
        total_values += finite_values.size

        global_min = min(global_min, finite_values.min())
        global_max = max(global_max, finite_values.max())

    if total_values == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "n_values": 0,
            "n_nan": int(n_nan),
            "n_inf": int(n_inf),
        }

    global_mean = total_sum / total_values

    # Var(X) = E[X²] - E[X]²
    global_variance = (
        total_squared_sum / total_values
        - global_mean**2
    )

    # Petite protection contre une valeur négative due aux arrondis numériques.
    global_variance = max(global_variance, 0.0)
    global_std = np.sqrt(global_variance)

    return {
        "mean": global_mean,
        "std": global_std,
        "min": global_min,
        "max": global_max,
        "n_values": int(total_values),
        "n_nan": int(n_nan),
        "n_inf": int(n_inf),
    }
def find_extreme_values(npy_files, threshold=0.01):
    """Recherche les fichiers contenant des valeurs EEG extrêmes.

    Parameters
    ----------
    npy_files : list[Path]
        Liste des fichiers .npy de la dyade.

    threshold : float
        Une valeur est considérée comme extrême lorsque sa valeur absolue
        dépasse ce seuil.

        Ici, 0.01 correspond à 10 mV si les données sont exprimées en volts,
        ce qui est déjà extrêmement grand pour de l'EEG.

    Returns
    -------
    pandas.DataFrame
        Tableau contenant, pour chaque fichier problématique :
        - le nom du fichier ;
        - le minimum et le maximum ;
        - la plus grande amplitude absolue ;
        - le nombre de valeurs dépassant le seuil ;
        - le participant, le canal et l'instant du maximum absolu.
    """
    rows = []

    for npy_file in npy_files:
        data = np.load(npy_file, mmap_mode="r")

        minimum = float(np.min(data))
        maximum = float(np.max(data))
        absolute_data = np.abs(data)

        max_absolute_value = float(np.max(absolute_data))
        n_extreme_values = int(np.sum(absolute_data > threshold))

        if n_extreme_values == 0:
            continue

        # np.argmax renvoie un indice aplati.
        flat_index = int(np.argmax(absolute_data))

        # On reconvertit cet indice en :
        # participant, canal et point temporel.
        participant_index, channel_index, time_index = np.unravel_index(
            flat_index,
            data.shape,
        )

        rows.append(
            {
                "filename": npy_file.name,
                "minimum": minimum,
                "maximum": maximum,
                "max_absolute_value": max_absolute_value,
                "n_values_above_threshold": n_extreme_values,
                "participant_index": participant_index,
                "channel_index": channel_index,
                "time_index": time_index,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "filename",
                "minimum",
                "maximum",
                "max_absolute_value",
                "n_values_above_threshold",
                "participant_index",
                "channel_index",
                "time_index",
            ]
        )

    result = pd.DataFrame(rows)

    return result.sort_values(
        by="max_absolute_value",
        ascending=False,
    ).reset_index(drop=True)
def inspect_extreme_file(npy_path, threshold=0.01):
    """Analyse en détail un fichier EEG contenant des valeurs extrêmes.

    Le fichier EEG possède la forme :

        (participants, canaux, temps)

    La fonction compte, pour chaque participant et chaque canal,
    combien de points dépassent le seuil en valeur absolue.
    """
    data = np.load(npy_path)

    rows = []

    for participant_index in range(data.shape[0]):
        for channel_index in range(data.shape[1]):
            signal = data[participant_index, channel_index, :]

            n_extreme = int(np.sum(np.abs(signal) > threshold))

            if n_extreme == 0:
                continue

            rows.append(
                {
                    "participant_index": participant_index,
                    "channel_index": channel_index,
                    "minimum": float(signal.min()),
                    "maximum": float(signal.max()),
                    "standard_deviation": float(signal.std()),
                    "n_values_above_threshold": n_extreme,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "participant_index",
                "channel_index",
                "minimum",
                "maximum",
                "standard_deviation",
                "n_values_above_threshold",
            ]
        )

    result = pd.DataFrame(rows)

    return result.sort_values(
        by="n_values_above_threshold",
        ascending=False,
    ).reset_index(drop=True)

def inspect_dyad(dyad_dir):
    """Inspecte tous les fichiers associés à une dyade et affiche un résumé."""
    dyad_id = dyad_dir.name

    epochs_dir = dyad_dir / "epochs"
    metadata_dir = dyad_dir / "metadata"
    manifest_path = dyad_dir / f"{dyad_id}_epochs_manifest.csv"
    channels_path = dyad_dir / f"{dyad_id}_channels.csv"

    npy_files = sorted(epochs_dir.glob("*.npy")) if epochs_dir.exists() else []
    metadata_files = (
        sorted(metadata_dir.glob("*.csv")) if metadata_dir.exists() else []
    )

    print("\n" + "=" * 70)
    print(f"Dyade : {dyad_id}")
    print("=" * 70)
    print("dossier :", dyad_dir)
    print("nb de fichiers .npy :", len(npy_files))
    print("nb de fichiers metadata :", len(metadata_files))
    print("Manifest présent :", manifest_path.exists())
    print("fichier canaux présent :", channels_path.exists())

    extreme_values = find_extreme_values(
        npy_files=npy_files,
        threshold=0.01,
    )

    print("\nRecherche de valeurs extrêmes |x| > 0.01")

    if extreme_values.empty:
        print("Aucune valeur extrême détectée.")
    else:
        print(
            extreme_values.to_string(
                index=False,
                float_format=lambda value: f"{value:.8e}",
            )
        )
    if dyad_id == "J7" and not extreme_values.empty:
        most_extreme_filename = extreme_values.iloc[0]["filename"]
        most_extreme_path = epochs_dir / most_extreme_filename

        channel_details = inspect_extreme_file(
            npy_path=most_extreme_path,
            threshold=0.01,
        )

        print(
            "\nDétail des participants et canaux contaminés dans :",
            most_extreme_filename,
        )

        print(
            channel_details.to_string(
                index=False,
                float_format=lambda value: f"{value:.8e}",
            )
        )
    # --- Inspection du premier fichier .npy ---
    if npy_files:
        first_npy = npy_files[0]
        npy_info = inspect_npy_file(first_npy)

        print("premier fichier .npy :", first_npy.name)
        print("shape :", npy_info["shape"])
        print("type :", npy_info["dtype"])
        print("participants :", npy_info["n_participants"])
        print("canaux :", npy_info["n_channels"])
        print("points temporels :", npy_info["n_times"])

    # --- Inspection du manifest ---
    # --- Inspection du manifest ---
    manifest = None

    if manifest_path.exists():
        manifest, manifest_info = inspect_manifest(manifest_path)

        print("\nManifest")
        print("nb de lignes :", manifest_info["n_rows"])
        print("labels yeux :", manifest_info["eyes_codes"])
        print("conditions :", manifest_info["conditions"])
        print("shapes déclarées :", manifest_info["shapes"])
        print("fréquences déclarées :", manifest_info["sampling_frequencies"])

        if "eyes_code" in manifest.columns:
            print("\nrépartition YO / YF :")
            print(manifest["eyes_code"].value_counts(dropna=False))

        if "condition_name" in manifest.columns:
            print("\nrépartition par condition :")
            print(manifest["condition_name"].value_counts(dropna=False))

        if len(manifest) != len(npy_files):
            print(
                "\nATTENTION : le nombre de lignes du manifest "
                "ne correspond pas au nombre de fichiers .npy."
            )
        else:
            print("\ncohérence manifest / fichiers .npy : OK")

    # Ce bloc est indépendant de l'existence du manifest.
    if npy_files:
        stats = inspect_signal_statistics(npy_files)

        print("\nStatistiques globales des signaux EEG")
        print(f"Moyenne : {stats['mean']:.8e}")
        print(f"Écart-type : {stats['std']:.8e}")
        print(f"Minimum : {stats['min']:.8e}")
        print(f"Maximum : {stats['max']:.8e}")
        print(f"Nombre total de valeurs : {stats['n_values']}")
        print(f"Nombre de NaN : {stats['n_nan']}")
        print(f"Nombre de valeurs infinies : {stats['n_inf']}")

    return {
        "dyad_id": dyad_id,
        "n_npy_files": len(npy_files),
        "n_metadata_files": len(metadata_files),
        "manifest_exists": manifest_path.exists(),
    }


def inspect_dataset(dataset_root):
    """Inspecte l'ensemble du dataset et retourne un résumé par dyade."""
    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"le dossier du dataset n'existe pas : {dataset_root}"
        )

    dyad_dirs = find_dyad_directories(dataset_root)

    print("racine du dataset :", dataset_root)
    print("nombre de dyades trouvées :", len(dyad_dirs))
    print("dyades :", [path.name for path in dyad_dirs])

    summary = [inspect_dyad(dyad_dir) for dyad_dir in dyad_dirs]
    summary_df = pd.DataFrame(summary)

    print("\n" + "=" * 70)
    print("RÉSUMÉ GLOBAL")
    print("=" * 70)
    print(summary_df)
    print("\nNombre total de fichiers .npy :", summary_df["n_npy_files"].sum())

    return summary_df


if __name__ == "__main__":
    inspect_dataset(DATASET_ROOT)