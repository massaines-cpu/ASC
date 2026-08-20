"""Orchestration complète : recherche, analyse, SQLite et exports CSV."""

from __future__ import annotations

import json

from . import config
from .compatibility import evaluate_asc_compatibility
from .database import ModelRegistryDatabase
from .huggingface_provider import HuggingFaceProvider
from .manual_overrides import apply_manual_overrides
from .metadata_extractor import MetadataExtractor


def run_registry_pipeline() -> None:
    """Construit ou actualise le registre sans modifier le projet ASC."""

    provider = HuggingFaceProvider()
    extractor = MetadataExtractor()
    errors: list[str] = []
    inspected_candidates = 0

    run_configuration = {
        "queries": list(config.SEARCH_QUERIES),
        "seed_repositories": list(config.SEED_REPOSITORIES),
        "max_models_per_query": config.MAX_MODELS_PER_QUERY,
        "max_unique_repositories": config.MAX_UNIQUE_REPOSITORIES,
        "asc_channel_names": list(config.ASC_CHANNEL_NAMES),
        "asc_sampling_frequency_hz": config.ASC_SAMPLING_FREQUENCY_HZ,
    }

    with ModelRegistryDatabase(
        database_path=config.DATABASE_PATH,
        schema_path=config.SCHEMA_PATH,
    ) as database:
        run_id = database.start_search_run(
            provider="huggingface_api",
            configuration=run_configuration,
        )

        try:
            discovered = provider.discover_repository_ids()
        except Exception as exception:
            database.finish_search_run(
                run_id,
                status="failed",
                error_message=f"{type(exception).__name__}: {exception}",
            )
            raise

        print(f"{len(discovered)} dépôts uniques à inspecter.")

        for repository_index, (repository_id, matches) in enumerate(
            discovered.items(),
            start=1,
        ):
            print(
                f"[{repository_index}/{len(discovered)}] "
                f"Inspection de {repository_id}"
            )
            try:
                repository = provider.inspect_repository(repository_id, matches)
                database.upsert_repository(repository)

                for checkpoint_path in provider.candidate_checkpoint_paths(repository):
                    documents = provider.load_candidate_documents(
                        repository,
                        checkpoint_path,
                    )
                    weight_files = provider.weight_files_for_candidate(
                        repository,
                        checkpoint_path,
                    )
                    profile = extractor.extract(
                        repository=repository,
                        candidate_documents=documents,
                        weight_files=weight_files,
                    )
                    profile = apply_manual_overrides(
                        profile,
                        config.MANUAL_OVERRIDES_PATH,
                    )
                    profile = evaluate_asc_compatibility(profile)

                    candidate_id = database.upsert_candidate(profile)
                    for query_text, result_rank in matches:
                        database.link_search_result(
                            run_id=run_id,
                            candidate_id=candidate_id,
                            query_text=query_text,
                            result_rank=result_rank,
                        )

                    inspected_candidates += 1
                    print(
                        f"  {profile.checkpoint_path} | "
                        f"poids={profile.weights_available} | "
                        f"canaux={profile.pretrained_channel_count} | "
                        f"ASC={profile.asc_32_status} | "
                        f"score={profile.compatibility_score:.1f}"
                    )
            except Exception as exception:
                message = (
                    f"{repository_id}: {type(exception).__name__}: {exception}"
                )
                errors.append(message)
                print(f"  ERREUR NON BLOQUANTE — {message}")

        catalog_path = database.export_catalog(
            config.EXPORT_DIRECTORY / "asc_model_catalog.csv"
        )
        evidence_path = database.export_evidence(
            config.EXPORT_DIRECTORY / "asc_model_evidence.csv"
        )
        electrodes_path = database.export_electrodes(
            config.EXPORT_DIRECTORY / "asc_model_electrodes.csv"
        )

        status = "partial_success" if errors else "success"
        database.finish_search_run(
            run_id,
            status=status,
            error_message=(
                json.dumps(errors, ensure_ascii=False) if errors else None
            ),
        )

        print("\n" + "=" * 72)
        print("REGISTRE TERMINÉ")
        print("=" * 72)
        print(f"Candidats inspectés : {inspected_candidates}")
        print(f"Erreurs isolées     : {len(errors)}")
        print(f"Base SQLite         : {config.DATABASE_PATH}")
        print(f"Catalogue CSV       : {catalog_path}")
        print(f"Preuves CSV         : {evidence_path}")
        print(f"Électrodes CSV      : {electrodes_path}")


if __name__ == "__main__":
    run_registry_pipeline()
