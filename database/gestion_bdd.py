"""Initialise et alimente la base SQLite du projet ASC.

Exemples
--------
Créer les tables :

    python database/gestion_bdd.py init

Importer les 338 epochs :

    python database/gestion_bdd.py import-epochs \
        --metadata data/all_metadata.csv \
        --dataset-root data/data_toy

Afficher un résumé :

    python database/gestion_bdd.py status
"""

import csv
from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "database" / "asc.sqlite3"
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "database" / "schema.sql"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "all_metadata.csv"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "data_toy"

LABELS = {"YO": 0, "YF": 1}


def connect(database_path: Path) -> sqlite3.Connection:
    """Ouvre SQLite en activant les clés étrangères."""

    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(
    database_path: Path,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> None:
    """Crée les tables et index de manière idempotente."""

    if not schema_path.exists():
        raise FileNotFoundError(f"Schéma SQL introuvable : {schema_path}")

    with connect(database_path) as connection:
        connection.executescript(schema_path.read_text(encoding="utf-8"))


def build_relative_epoch_path(dyad_id: str, filename: str) -> Path:
    """Construit le chemin attendu sans accepter un chemin fourni par CSV."""

    return Path(dyad_id) / "epochs" / filename


def import_epochs(
    database_path: Path,
    metadata_path: Path,
    dataset_root: Path,
) -> tuple[int, int]:
    """Importe les métadonnées et retourne (insérées, mises à jour)."""

    if not metadata_path.exists():
        raise FileNotFoundError(f"Métadonnées introuvables : {metadata_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset introuvable : {dataset_root}")

    initialize_database(database_path)
    inserted = 0
    updated = 0

    with metadata_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {
            "dyad_id",
            "filename",
            "eyes_code",
            "sampling_frequency_hz",
            "number_of_channels",
            "number_of_time_points",
            "matrix_dtype",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Colonnes manquantes dans les métadonnées : "
                + ", ".join(sorted(missing_columns))
            )

        with connect(database_path) as connection:
            for row in reader:
                condition = row["eyes_code"]
                if condition not in LABELS:
                    continue

                dyad_id = row["dyad_id"]
                filename = row["filename"]
                relative_path = build_relative_epoch_path(dyad_id, filename)
                absolute_path = dataset_root / relative_path

                if not absolute_path.is_file():
                    raise FileNotFoundError(
                        f"Fichier référencé mais absent : {absolute_path}"
                    )

                number_of_timepoints = int(row["number_of_time_points"])
                sampling_frequency_hz = float(row["sampling_frequency_hz"])
                duration_seconds = (
                    number_of_timepoints / sampling_frequency_hz
                )

                existing = connection.execute(
                    """
                    SELECT id FROM epochs
                    WHERE dyad_id = ? AND filename = ?
                    """,
                    (dyad_id, filename),
                ).fetchone()

                connection.execute(
                    """
                    INSERT INTO epochs (
                        dyad_id,
                        filename,
                        relative_path,
                        condition,
                        label,
                        number_of_participants,
                        number_of_channels,
                        number_of_timepoints,
                        sampling_frequency_hz,
                        duration_seconds,
                        matrix_dtype
                    )
                    VALUES (?, ?, ?, ?, ?, 2, ?, ?, ?, ?, ?)
                    ON CONFLICT(dyad_id, filename) DO UPDATE SET
                        relative_path = excluded.relative_path,
                        condition = excluded.condition,
                        label = excluded.label,
                        number_of_channels = excluded.number_of_channels,
                        number_of_timepoints = excluded.number_of_timepoints,
                        sampling_frequency_hz = excluded.sampling_frequency_hz,
                        duration_seconds = excluded.duration_seconds,
                        matrix_dtype = excluded.matrix_dtype
                    """,
                    (
                        dyad_id,
                        filename,
                        str(relative_path),
                        condition,
                        LABELS[condition],
                        int(row["number_of_channels"]),
                        number_of_timepoints,
                        sampling_frequency_hz,
                        duration_seconds,
                        row["matrix_dtype"],
                    ),
                )

                if existing is None:
                    inserted += 1
                else:
                    updated += 1

    return inserted, updated


def print_status(database_path: Path) -> None:
    """Affiche les effectifs qui alimenteront plus tard l'API."""

    if not database_path.exists():
        raise FileNotFoundError(
            f"Base absente : {database_path}. Lance d'abord la commande init."
        )

    with connect(database_path) as connection:
        epoch_count = connection.execute(
            "SELECT COUNT(*) AS count FROM epochs"
        ).fetchone()["count"]
        dyad_count = connection.execute(
            "SELECT COUNT(DISTINCT dyad_id) AS count FROM epochs"
        ).fetchone()["count"]
        diagnostic_count = connection.execute(
            "SELECT COUNT(*) AS count FROM diagnostics"
        ).fetchone()["count"]
        model_count = connection.execute(
            "SELECT COUNT(*) AS count FROM models"
        ).fetchone()["count"]
        prediction_count = connection.execute(
            "SELECT COUNT(*) AS count FROM predictions"
        ).fetchone()["count"]
        condition_rows = connection.execute(
            """
            SELECT condition, COUNT(*) AS count
            FROM epochs
            GROUP BY condition
            ORDER BY condition
            """
        ).fetchall()

    print("\n=== BASE DE DONNÉES ASC ===")
    print(f"Fichier       : {database_path}")
    print(f"Dyades        : {dyad_count}")
    print(f"Epochs        : {epoch_count}")
    print(f"Diagnostics   : {diagnostic_count}")
    print(f"Modèles       : {model_count}")
    print(f"Prédictions   : {prediction_count}")
    print("Conditions    :")
    for row in condition_rows:
        print(f"  {row['condition']} : {row['count']}")


def main() -> None:
    """Initialise, alimente et vérifie la base automatiquement."""

    print("Initialisation de la base de données ASC...")
    initialize_database(DEFAULT_DATABASE_PATH)

    print("Import des métadonnées des epochs...")
    inserted, updated = import_epochs(
        database_path=DEFAULT_DATABASE_PATH,
        metadata_path=DEFAULT_METADATA_PATH,
        dataset_root=DEFAULT_DATASET_ROOT,
    )

    print(f"Epochs insérés    : {inserted}")
    print(f"Epochs mis à jour : {updated}")
    print_status(DEFAULT_DATABASE_PATH)


if __name__ == "__main__":
    main()
