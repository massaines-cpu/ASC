from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "data_toy"

# Parcourt tous les dossiers de dyades et récupère tous les fichiers .npy présents dans les dossiers epochs.

def build_filename_table(dataset_root):

    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Le dossier du dataset n'existe pas : {dataset_root}"
        )

    rows = []

    # Recherche les dossiers J2, J4, J5, J10, etc.
    dyad_directories = sorted(dataset_root.glob("J*"))

    for dyad_directory in dyad_directories:

        #ignore les éléments qui ne sont pas des dossiers
        if not dyad_directory.is_dir():
            continue

        dyad_id = dyad_directory.name

        epochs_directory = dyad_directory / "epochs"

        if not epochs_directory.exists():
            print(
                f"Attention : dossier epochs absent pour {dyad_id}"
            )
            continue

        npy_files = sorted(
            epochs_directory.glob("*.npy")
        )

        if len(npy_files) == 0:
            print(
                f"Attention : aucun fichier .npy pour {dyad_id}"
            )
            continue

        for npy_file in npy_files:
            filename_info = parse_filename(
                npy_file.name
            )

            relative_path = npy_file.relative_to(
                PROJECT_ROOT
            )

            row = {
                "dyad_id": dyad_id,
                "filename": npy_file.name,
                "epoch_file": relative_path.as_posix(),
                **filename_info,
            }
            rows.append(row)

    filename_table = pd.DataFrame(rows)

    return filename_table

# Lit les manifests de toutes les dyades et les rassemble dans un seul DataFrame.
def load_all_manifests(dataset_root):

    dataset_root = Path(dataset_root)

    manifest_tables = []

    dyad_directories = sorted(dataset_root.glob("J*"))

    for dyad_directory in dyad_directories:

        if not dyad_directory.is_dir():
            continue

        dyad_id = dyad_directory.name

        manifest_path = (
            dyad_directory
            / f"{dyad_id}_epochs_manifest.csv"
        )

        if not manifest_path.exists():
            print(
                f"Attention : manifest absent pour {dyad_id}"
            )
            continue

        manifest = pd.read_csv(manifest_path)

        # On s'assure que la dyade est bien indiquée.
        manifest["dyad_id"] = dyad_id

        manifest_tables.append(manifest)

    if len(manifest_tables) == 0:
        raise RuntimeError(
            "Aucun manifest n'a été trouvé."
        )

    all_manifests = pd.concat(
        manifest_tables,
        ignore_index=True,
    )

    return all_manifests

# Décompose un nom comme : J10_E004_C4_YF_S12_5A15.npy

def parse_filename(filename):

    filename_without_extension = Path(filename).stem

    parts = filename_without_extension.split("_")

    if len(parts) != 6:
        raise ValueError(
            f"Format de nom inattendu : {filename}"
        )

    return {
        "filename_dyad": parts[0],
        "epoch_index": int(parts[1][1:]),
        "filename_condition": parts[2],
        "filename_eyes": parts[3],
        "filename_event": parts[4],
        "filename_segment": parts[5],
    }

def attach_filenames_to_manifests(
    filename_table,
    manifests_table,
):
    """
    Associe les fichiers .npy aux lignes des manifests,
    dyade par dyade, selon leur ordre.
    """

    merged_tables = []

    dyad_ids = sorted(
        filename_table["dyad_id"].unique()
    )

    for dyad_id in dyad_ids:

        dyad_files = filename_table[
            filename_table["dyad_id"] == dyad_id
        ].copy()

        dyad_manifest = manifests_table[
            manifests_table["dyad_id"] == dyad_id
        ].copy()

        # On trie les fichiers selon E000, E001, E002...
        dyad_files = dyad_files.sort_values(
            "epoch_index"
        ).reset_index(drop=True)

        # On conserve l'ordre du manifest.
        dyad_manifest = dyad_manifest.reset_index(
            drop=True
        )

        if len(dyad_files) != len(dyad_manifest):
            raise ValueError(
                f"{dyad_id} : "
                f"{len(dyad_files)} fichiers mais "
                f"{len(dyad_manifest)} lignes de manifest."
            )

        # On ajoute les informations du fichier au manifest.
        dyad_manifest["filename"] = (
            dyad_files["filename"]
        )

        dyad_manifest["epoch_file"] = (
            dyad_files["epoch_file"]
        )

        dyad_manifest["epoch_index"] = (
            dyad_files["epoch_index"]
        )

        dyad_manifest["filename_condition"] = (
            dyad_files["filename_condition"]
        )

        dyad_manifest["filename_eyes"] = (
            dyad_files["filename_eyes"]
        )

        dyad_manifest["filename_segment"] = (
            dyad_files["filename_segment"]
        )

        merged_tables.append(dyad_manifest)

    global_table = pd.concat(
        merged_tables,
        ignore_index=True,
    )

    return global_table
def validate_filename_matching(global_table):
    """
    Vérifie que les informations du nom du fichier
    correspondent aux informations du manifest.
    """

    condition_errors = (
        global_table["condition_code"]
        != global_table["filename_condition"]
    )

    eyes_errors = (
        global_table["eyes_code"]
        != global_table["filename_eyes"]
    )

    segment_errors = (
        global_table["segment"]
        != global_table["filename_segment"]
    )

    print(
        "Erreurs condition :",
        condition_errors.sum(),
    )

    print(
        "Erreurs yeux :",
        eyes_errors.sum(),
    )

    print(
        "Erreurs segment :",
        segment_errors.sum(),
    )

    if (
        condition_errors.any()
        or eyes_errors.any()
        or segment_errors.any()
    ):
        incorrect_rows = global_table[
            condition_errors
            | eyes_errors
            | segment_errors
        ]

        print("\nLignes incorrectes :")
        print(
            incorrect_rows[
                [
                    "filename",
                    "condition_code",
                    "filename_condition",
                    "eyes_code",
                    "filename_eyes",
                    "segment",
                    "filename_segment",
                ]
            ]
        )

        raise ValueError(
            "Les fichiers et les manifests "
            "ne sont pas correctement alignés."
        )

    print(
        "Correspondance fichiers / manifests : OK"
    )

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "all_metadata.csv"
)
# Sauvegarde le DataFrame complet contenant toutes les métadonnées du dataset.
def save_all_metadata(dataframe, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(f"\nall_metadata.csv sauvegardé : {output_path}")
if __name__ == "__main__":

    filename_table = build_filename_table(
        DATASET_ROOT
    )

    manifests_table = load_all_manifests(
        DATASET_ROOT
    )

    global_table = attach_filenames_to_manifests(
        filename_table,
        manifests_table,
    )

    validate_filename_matching(
        global_table
    )
    global_table = global_table.drop(
        columns=[
            "filename_condition",
            "filename_eyes",
            "filename_segment",
            "filename_dyad",
            'epoch_index',
            'epoch_file',
        ],
        errors="ignore",  # évite une erreur si une colonne n'existe pas
    )
    save_all_metadata(
        global_table,
        OUTPUT_CSV,
    )

    print("\ntable globale :")
    print(
        global_table[
            [
                "filename",
                "dyad_id",
                "condition_code",
                "eyes_code",
                "segment",
                "epoch_file",
            ]
        ].head()
    )

    print(
        "\nnombre de lignes :",
        len(global_table)
    )
