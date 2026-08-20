"""Accès SQLite du registre de modèles EEG."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .records import CandidateProfile, RepositoryRecord


def utc_now() -> str:
    """Renvoie une date ISO 8601 comparable entre plusieurs exécutions."""

    return datetime.now(timezone.utc).isoformat()


class ModelRegistryDatabase:
    """Regroupe les opérations SQL afin de ne pas les dupliquer."""

    def __init__(self, database_path: Path, schema_path: Path):
        self.database_path = database_path
        self.schema_path = schema_path
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> "ModelRegistryDatabase":
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.initialize_schema()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        if self.connection is None:
            return
        if exception is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()
        self.connection = None

    def require_connection(self) -> sqlite3.Connection:
        if self.connection is None:
            raise RuntimeError("La connexion SQLite n'est pas ouverte.")
        return self.connection

    def initialize_schema(self) -> None:
        """Crée les tables et la vue sans supprimer les données existantes."""

        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schéma SQL introuvable : {self.schema_path}")
        schema = self.schema_path.read_text(encoding="utf-8")
        self.require_connection().executescript(schema)

    def start_search_run(self, provider: str, configuration: dict) -> int:
        cursor = self.require_connection().execute(
            """
            INSERT INTO search_run(
                provider, started_at, status, configuration_json
            ) VALUES (?, ?, 'running', ?)
            """,
            (provider, utc_now(), json.dumps(configuration, ensure_ascii=False)),
        )
        self.require_connection().commit()
        return int(cursor.lastrowid)

    def finish_search_run(
        self,
        run_id: int,
        status: str,
        error_message: str | None = None,
    ) -> None:
        self.require_connection().execute(
            """
            UPDATE search_run
            SET finished_at = ?, status = ?, error_message = ?
            WHERE run_id = ?
            """,
            (utc_now(), status, error_message, run_id),
        )
        self.require_connection().commit()

    def upsert_repository(self, repository: RepositoryRecord) -> None:
        """Ajoute un dépôt ou actualise ses métadonnées changeantes."""

        now = utc_now()
        self.require_connection().execute(
            """
            INSERT INTO repository(
                repository_id, provider, model_url, author, license,
                pipeline_tag, library_name, downloads, likes, revision,
                last_modified, card_text, discovered_at, last_checked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(repository_id) DO UPDATE SET
                provider = excluded.provider,
                model_url = excluded.model_url,
                author = excluded.author,
                license = excluded.license,
                pipeline_tag = excluded.pipeline_tag,
                library_name = excluded.library_name,
                downloads = excluded.downloads,
                likes = excluded.likes,
                revision = excluded.revision,
                last_modified = excluded.last_modified,
                card_text = excluded.card_text,
                last_checked_at = excluded.last_checked_at
            """,
            (
                repository.repository_id,
                repository.provider,
                repository.model_url,
                repository.author,
                repository.license,
                repository.pipeline_tag,
                repository.library_name,
                repository.downloads,
                repository.likes,
                repository.revision,
                repository.last_modified,
                repository.card_text,
                now,
                now,
            ),
        )

    def upsert_candidate(self, profile: CandidateProfile) -> int:
        """Enregistre un profil puis remplace ses relations détaillées."""

        connection = self.require_connection()
        connection.execute(
            """
            INSERT INTO candidate(
                repository_id, checkpoint_path, display_name, architecture,
                weights_available, pretrained_channel_count, channel_mode,
                electrode_positions_available, sampling_frequency_hz,
                window_duration_seconds, number_of_timepoints, preprocessing,
                pretraining_dataset, source_task, asc_32_status,
                compatibility_reason, compatibility_score, confidence,
                metadata_status, inspected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(repository_id, checkpoint_path) DO UPDATE SET
                display_name = excluded.display_name,
                architecture = excluded.architecture,
                weights_available = excluded.weights_available,
                pretrained_channel_count = excluded.pretrained_channel_count,
                channel_mode = excluded.channel_mode,
                electrode_positions_available = excluded.electrode_positions_available,
                sampling_frequency_hz = excluded.sampling_frequency_hz,
                window_duration_seconds = excluded.window_duration_seconds,
                number_of_timepoints = excluded.number_of_timepoints,
                preprocessing = excluded.preprocessing,
                pretraining_dataset = excluded.pretraining_dataset,
                source_task = excluded.source_task,
                asc_32_status = excluded.asc_32_status,
                compatibility_reason = excluded.compatibility_reason,
                compatibility_score = excluded.compatibility_score,
                confidence = excluded.confidence,
                metadata_status = excluded.metadata_status,
                inspected_at = excluded.inspected_at
            """,
            (
                profile.repository_id,
                profile.checkpoint_path,
                profile.display_name,
                profile.architecture,
                int(profile.weights_available),
                profile.pretrained_channel_count,
                profile.channel_mode,
                (
                    None
                    if profile.electrode_positions_available is None
                    else int(profile.electrode_positions_available)
                ),
                profile.sampling_frequency_hz,
                profile.window_duration_seconds,
                profile.number_of_timepoints,
                profile.preprocessing,
                profile.pretraining_dataset,
                profile.source_task,
                profile.asc_32_status,
                profile.compatibility_reason,
                profile.compatibility_score,
                profile.confidence,
                profile.metadata_status,
                utc_now(),
            ),
        )

        row = connection.execute(
            """
            SELECT candidate_id FROM candidate
            WHERE repository_id = ? AND checkpoint_path = ?
            """,
            (profile.repository_id, profile.checkpoint_path),
        ).fetchone()
        candidate_id = int(row["candidate_id"])

        # Les détails sont reconstruits à chaque inspection afin de ne pas
        # conserver une ancienne électrode ou une ancienne preuve.
        for table in ("weight_file", "candidate_electrode", "evidence"):
            connection.execute(
                f"DELETE FROM {table} WHERE candidate_id = ?",
                (candidate_id,),
            )

        for weight_file in profile.weight_files:
            connection.execute(
                """
                INSERT INTO weight_file(
                    candidate_id, file_path, file_format, size_bytes, download_url
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    weight_file.path,
                    Path(weight_file.path).suffix.lower(),
                    weight_file.size_bytes,
                    weight_file.download_url,
                ),
            )

        electrode_source = next(
            (
                evidence.source_url
                for evidence in profile.evidence
                if evidence.field_name == "electrode_names"
            ),
            None,
        )
        for electrode_order, electrode_name in enumerate(profile.electrode_names):
            if electrode_order < len(profile.electrode_positions):
                x, y, z = profile.electrode_positions[electrode_order]
            else:
                x, y, z = None, None, None
            connection.execute(
                """
                INSERT INTO candidate_electrode(
                    candidate_id, electrode_order, electrode_name,
                    x, y, z, source_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    electrode_order,
                    electrode_name,
                    x,
                    y,
                    z,
                    electrode_source,
                ),
            )

        for evidence in profile.evidence:
            connection.execute(
                """
                INSERT INTO evidence(
                    candidate_id, field_name, value_text, source_url,
                    extraction_method, confidence, collected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    evidence.field_name,
                    evidence.value_text,
                    evidence.source_url,
                    evidence.extraction_method,
                    evidence.confidence,
                    utc_now(),
                ),
            )

        return candidate_id

    def link_search_result(
        self,
        run_id: int,
        candidate_id: int,
        query_text: str,
        result_rank: int,
    ) -> None:
        self.require_connection().execute(
            """
            INSERT OR REPLACE INTO search_result(
                run_id, candidate_id, query_text, result_rank
            ) VALUES (?, ?, ?, ?)
            """,
            (run_id, candidate_id, query_text, result_rank),
        )

    def export_catalog(self, output_path: Path) -> Path:
        """Exporte la vue métier avec les intitulés demandés par Amel."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.require_connection().execute(
            """
            SELECT * FROM asc_model_catalog
            ORDER BY "Score de compatibilité" DESC, "Nom du modèle"
            """
        ).fetchall()

        with output_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            if rows:
                writer.writerow(rows[0].keys())
                writer.writerows([tuple(row) for row in rows])
        return output_path

    def export_evidence(self, output_path: Path) -> Path:
        """Exporte les preuves pour pouvoir justifier chaque valeur."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.require_connection().execute(
            """
            SELECT
                c.display_name,
                e.field_name,
                e.value_text,
                e.source_url,
                e.extraction_method,
                e.confidence,
                e.collected_at
            FROM evidence e
            JOIN candidate c ON c.candidate_id = e.candidate_id
            ORDER BY c.display_name, e.field_name
            """
        ).fetchall()

        with output_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            if rows:
                writer.writerow(rows[0].keys())
                writer.writerows([tuple(row) for row in rows])
        return output_path

    def export_electrodes(self, output_path: Path) -> Path:
        """Exporte une ligne par électrode pour garder les positions lisibles."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.require_connection().execute(
            """
            SELECT
                c.display_name,
                ce.electrode_order,
                ce.electrode_name,
                ce.x,
                ce.y,
                ce.z,
                ce.source_url
            FROM candidate_electrode ce
            JOIN candidate c ON c.candidate_id = ce.candidate_id
            ORDER BY c.display_name, ce.electrode_order
            """
        ).fetchall()

        with output_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            if rows:
                writer.writerow(rows[0].keys())
                writer.writerows([tuple(row) for row in rows])
        return output_path
