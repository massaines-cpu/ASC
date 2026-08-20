"""Recherche et lecture prudente des dépôts de modèles Hugging Face.

Le module utilise l'API officielle ``huggingface_hub``. Il ne dépend donc pas
de la mise en page HTML du site. Seuls les petits fichiers de métadonnées sont
téléchargés automatiquement : les checkpoints lourds sont seulement listés.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.parse import quote

from huggingface_hub import HfApi, hf_hub_download

from . import config
from .records import RemoteFile, RepositoryRecord


@dataclass(frozen=True)
class CandidateDocuments:
    """Documents descriptifs disponibles pour un checkpoint."""

    checkpoint_path: str
    documents: dict[str, bytes]
    document_urls: dict[str, str]


def _download_url(repository_id: str, revision: str, path: str) -> str:
    """Construit une URL lisible et enregistrable comme preuve."""

    encoded_path = quote(path, safe="/")
    return (
        f"https://huggingface.co/{repository_id}/resolve/"
        f"{revision}/{encoded_path}"
    )


def _is_weight_file(path: str) -> bool:
    """Distingue un poids probable d'un simple fichier auxiliaire.

    Un fichier ``kwargs.pkl`` n'est par exemple pas un checkpoint. Pour les
    formats ambigus ``.pkl``, la présence d'un mot lié au modèle est demandée
    afin de limiter les faux positifs.
    """

    lower_path = path.lower()
    suffix = PurePosixPath(lower_path).suffix

    if suffix not in config.WEIGHT_SUFFIXES:
        return False

    if any(part in lower_path for part in config.EXCLUDED_WEIGHT_NAME_PARTS):
        return False

    if suffix in {".safetensors", ".bin", ".ckpt", ".pt", ".pth"}:
        return True

    positive_words = (
        "model", "weight", "checkpoint", "params", "state_dict", "eegnet"
    )
    return any(word in lower_path for word in positive_words)


def _is_metadata_file(path: str) -> bool:
    """Indique si un petit fichier peut documenter l'expérience source."""

    name = PurePosixPath(path).name
    lower_name = name.lower()
    return (
        name in config.METADATA_FILE_NAMES
        or lower_name in {item.lower() for item in config.METADATA_FILE_NAMES}
        or lower_name.endswith("config.json")
        or lower_name.endswith("metadata.json")
    )


class HuggingFaceProvider:
    """Client responsable uniquement des échanges avec Hugging Face."""

    def __init__(self) -> None:
        token = os.getenv(config.HUGGINGFACE_TOKEN_ENVIRONMENT_VARIABLE)
        self.token = token or None
        self.api = HfApi(token=self.token)

    def discover_repository_ids(
        self,
    ) -> dict[str, list[tuple[str, int]]]:
        """Recherche des dépôts et mémorise requête et rang d'apparition."""

        matches: dict[str, list[tuple[str, int]]] = defaultdict(list)

        for query in config.SEARCH_QUERIES:
            models = self.api.list_models(
                search=query,
                limit=config.MAX_MODELS_PER_QUERY,
                full=True,
                cardData=True,
                fetch_config=True,
            )
            for rank, model in enumerate(models, start=1):
                repository_id = getattr(model, "id", None)
                if not repository_id:
                    continue
                matches[repository_id].append((query, rank))

                if len(matches) >= config.MAX_UNIQUE_REPOSITORIES:
                    break

            if len(matches) >= config.MAX_UNIQUE_REPOSITORIES:
                break

        # Les modèles déjà identifiés par le projet restent contrôlés à chaque
        # exécution, même si leur popularité ou leur rang de recherche change.
        for repository_id in config.SEED_REPOSITORIES:
            matches.setdefault(repository_id, []).append(("seed_repository", 0))

        return dict(matches)

    def inspect_repository(
        self,
        repository_id: str,
        matched_queries: list[tuple[str, int]],
    ) -> RepositoryRecord:
        """Récupère la fiche, la liste des fichiers et leurs tailles."""

        info = self.api.model_info(
            repo_id=repository_id,
            files_metadata=True,
        )
        revision = getattr(info, "sha", None) or "main"

        remote_files: list[RemoteFile] = []
        for sibling in getattr(info, "siblings", None) or []:
            path = getattr(sibling, "rfilename", None)
            if not path:
                continue
            remote_files.append(
                RemoteFile(
                    path=path,
                    size_bytes=getattr(sibling, "size", None),
                    download_url=_download_url(repository_id, revision, path),
                )
            )

        card_text = self._download_text_if_small(
            repository_id=repository_id,
            filename="README.md",
            revision=revision,
            known_size=self._known_size(remote_files, "README.md"),
        ) or ""

        card_data = getattr(info, "card_data", None)
        license_name = getattr(card_data, "license", None)
        if isinstance(license_name, list):
            license_name = ", ".join(str(item) for item in license_name)

        last_modified = getattr(info, "last_modified", None)
        if last_modified is not None:
            last_modified = last_modified.isoformat()

        api_metadata = {
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "library_name": getattr(info, "library_name", None),
            "tags": list(getattr(info, "tags", None) or []),
            "config": getattr(info, "config", None) or {},
            "datasets": getattr(card_data, "datasets", None),
        }

        return RepositoryRecord(
            repository_id=repository_id,
            provider="huggingface",
            model_url=f"https://huggingface.co/{repository_id}",
            author=getattr(info, "author", None),
            license=license_name,
            pipeline_tag=getattr(info, "pipeline_tag", None),
            library_name=getattr(info, "library_name", None),
            downloads=getattr(info, "downloads", None),
            likes=getattr(info, "likes", None),
            revision=revision,
            last_modified=last_modified,
            card_text=card_text,
            api_metadata=api_metadata,
            files=remote_files,
            matched_queries=matched_queries,
        )

    def candidate_checkpoint_paths(
        self,
        repository: RepositoryRecord,
    ) -> list[str]:
        """Retourne un candidat par dossier contenant des poids.

        C'est indispensable pour ``PierreGtch/EEGNetv4`` : un même dépôt
        contient plusieurs checkpoints entraînés sur des datasets différents.
        """

        paths = {
            str(PurePosixPath(file.path).parent)
            for file in repository.files
            if _is_weight_file(file.path)
        }
        normalized_paths = {"." if path == "." else path for path in paths}
        return sorted(normalized_paths) or ["."]

    def weight_files_for_candidate(
        self,
        repository: RepositoryRecord,
        checkpoint_path: str,
    ) -> list[RemoteFile]:
        """Associe uniquement les poids du dossier du candidat."""

        return [
            file
            for file in repository.files
            if _is_weight_file(file.path)
            and str(PurePosixPath(file.path).parent) == checkpoint_path
        ]

    def load_candidate_documents(
        self,
        repository: RepositoryRecord,
        checkpoint_path: str,
    ) -> CandidateDocuments:
        """Télécharge les petits fichiers utiles au candidat.

        Le README racine est partagé par tous les checkpoints. Les fichiers de
        configuration sont retenus s'ils se trouvent à la racine ou dans le
        dossier exact du checkpoint.
        """

        selected_files: list[RemoteFile] = []
        for remote_file in repository.files:
            parent = str(PurePosixPath(remote_file.path).parent)
            is_root_readme = remote_file.path == "README.md"
            is_relevant_directory = parent in {".", checkpoint_path}
            if (is_root_readme or is_relevant_directory) and _is_metadata_file(
                remote_file.path
            ):
                selected_files.append(remote_file)

        documents: dict[str, bytes] = {}
        document_urls: dict[str, str] = {}
        for remote_file in selected_files:
            if (
                remote_file.size_bytes is not None
                and remote_file.size_bytes > config.MAX_METADATA_FILE_BYTES
            ):
                continue
            content = self._download_bytes(
                repository_id=repository.repository_id,
                filename=remote_file.path,
                revision=repository.revision or "main",
            )
            if content is None:
                continue
            documents[remote_file.path] = content
            document_urls[remote_file.path] = remote_file.download_url

        return CandidateDocuments(
            checkpoint_path=checkpoint_path,
            documents=documents,
            document_urls=document_urls,
        )

    def _download_bytes(
        self,
        repository_id: str,
        filename: str,
        revision: str,
    ) -> bytes | None:
        """Télécharge un document sans interrompre tout le catalogue en cas d'échec."""

        try:
            local_path = hf_hub_download(
                repo_id=repository_id,
                filename=filename,
                revision=revision,
                token=self.token,
                cache_dir=config.DOWNLOAD_CACHE_DIRECTORY,
            )
            path = Path(local_path)
            if path.stat().st_size > config.MAX_METADATA_FILE_BYTES:
                return None
            return path.read_bytes()
        except Exception as exception:
            print(
                f"  Métadonnée ignorée ({repository_id}/{filename}) : "
                f"{type(exception).__name__}"
            )
            return None

    def _download_text_if_small(
        self,
        repository_id: str,
        filename: str,
        revision: str,
        known_size: int | None,
    ) -> str | None:
        if known_size is not None and known_size > config.MAX_METADATA_FILE_BYTES:
            return None
        content = self._download_bytes(repository_id, filename, revision)
        if content is None:
            return None
        return content.decode("utf-8", errors="replace")

    @staticmethod
    def _known_size(files: Iterable[RemoteFile], path: str) -> int | None:
        for remote_file in files:
            if remote_file.path == path:
                return remote_file.size_bytes
        return None
