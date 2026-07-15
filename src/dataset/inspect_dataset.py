from pathlib import Path

import numpy as np
import pandas as pd


chemin_projet = Path(__file__).resolve().parents[2]

chemin_toy = chemin_projet / "data" / "data_toy"

#recherche tous les dossiers de dyades : J1, J2, J4, etc........
def find_dyad_directories(dataset_root):

    dyad_dirs = []

    for path in dataset_root.iterdir():
        if path.is_dir() and path.name.startswith("J"):
            dyad_dirs.append(path)

    return sorted(dyad_dirs)

#ouvre fichier .npy sans charger ttes données en mémoire + retourne ses principa caract
def inspect_npy_file(npy_path):

    data = np.load(npy_path, mmap_mode="r")

    return {
        "shape": data.shape,
        "dtype": str(data.dtype),
        "n_participants": data.shape[0] if data.ndim >= 1 else None,
        "n_channels": data.shape[1] if data.ndim >= 2 else None,
        "n_times": data.shape[2] if data.ndim >= 3 else None,
    }

#lit le manifest d'une dyade + retourne qlq info
def inspect_manifest(manifest_path):

    manifest = pd.read_csv(manifest_path)

    result = {
        "n_rows": len(manifest),
        "columns": list(manifest.columns),
        "eyes_codes": (
            sorted(manifest["eyes_code"].dropna().unique().tolist())
            if "eyes_code" in manifest.columns
            else []
        ),
        "conditions": (
            sorted(manifest["condition_code"].dropna().unique().tolist())
            if "condition_code" in manifest.columns
            else []
        ),
        "shapes": (
            sorted(manifest["matrix_shape"].dropna().unique().tolist())
            if "matrix_shape" in manifest.columns
            else []
        ),
        "sampling_frequencies": (
            sorted(
                manifest["sampling_frequency_hz"]
                .dropna()
                .unique()
                .tolist()
            )
            if "sampling_frequency_hz" in manifest.columns
            else []
        ),
    }

    return manifest, result

#inspecte tous les fichiers associés à une dyade
def inspect_dyad(dyad_dir):

    dyad_id = dyad_dir.name

    epochs_dir = dyad_dir / "epochs"
    metadata_dir = dyad_dir / "metadata"
    manifest_path = dyad_dir / f"{dyad_id}_epochs_manifest.csv"
    channels_path = dyad_dir / f"{dyad_id}_channels.csv"

    npy_files = sorted(epochs_dir.glob("*.npy")) if epochs_dir.exists() else []
    metadata_files = (
        sorted(metadata_dir.glob("*.csv"))
        if metadata_dir.exists()
        else []
    )

    print("\n" + "=" * 70)
    print(f"DYade : {dyad_id}")
    print("=" * 70)

    print("dossier :", dyad_dir)
    print("nb de fichiers .npy :", len(npy_files))
    print("nb de fichiers metadata :", len(metadata_files))
    print("Manifest présent :", manifest_path.exists())
    print("fichier canaux présent :", channels_path.exists())

#inspection du premier fichier .npy
    if npy_files:
        first_npy = npy_files[0]
        npy_info = inspect_npy_file(first_npy)

        print("premier fichier .npy :", first_npy.name)
        print("shape :", npy_info["shape"])
        print("type :", npy_info["dtype"])
        print("participants :", npy_info["n_participants"])
        print("canaux :", npy_info["n_channels"])
        print("points temporels :", npy_info["n_times"])

#inspection du manifest
    if manifest_path.exists():
        manifest, manifest_info = inspect_manifest(manifest_path)

        print("\nManifest")
        print("nb de lignes :", manifest_info["n_rows"])
        print("labels yeux :", manifest_info["eyes_codes"])
        print("conditions :", manifest_info["conditions"])
        print("shapes déclarées :", manifest_info["shapes"])
        print(
            "fréquences déclarées :",
            manifest_info["sampling_frequencies"],
        )

        if "eyes_code" in manifest.columns:
            print("\nrépartition YO / YF :")
            print(manifest["eyes_code"].value_counts(dropna=False))

        if "condition_name" in manifest.columns:
            print("\nrépartition par condition :")
            print(manifest["condition_name"].value_counts(dropna=False))

#vérification cohérence manifest / fichiers
        if len(manifest) != len(npy_files):
            print(
                "\nATTENTION!!!!!! : le nombre de lignes du manifest "
                "ne correspond pas au nombre de fichiers .npy."
            )
        else:
            print(
                "\ncohérence manifest / fichiers .npy : OK"
            )

    return {
        "dyad_id": dyad_id,
        "n_npy_files": len(npy_files),
        "n_metadata_files": len(metadata_files),
        "manifest_exists": manifest_path.exists(),
    }

#inspecte l'ensemble du dataset
def inspect_dataset(dataset_root):
    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"le dossier du dataset n'existe pas : {dataset_root}"
        )

    dyad_dirs = find_dyad_directories(dataset_root)

    print("racine du dataset :", dataset_root)
    print("nombre de dyades trouvées :", len(dyad_dirs))
    print("dyades :", [path.name for path in dyad_dirs])

    summary = []

    for dyad_dir in dyad_dirs:
        dyad_summary = inspect_dyad(dyad_dir)
        summary.append(dyad_summary)

    summary_df = pd.DataFrame(summary)

    print("RÉSUMÉ GLOBAL")
    print(summary_df)

    print(
        "\nNombre total de fichiers .npy :",
        summary_df["n_npy_files"].sum(),
    )

    return summary_df


if __name__ == "__main__":
    summary = inspect_dataset(chemin_toy)