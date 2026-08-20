"""Structures de données échangées entre les modules du registre."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RemoteFile:
    """Fichier décrit par l'API du fournisseur."""

    path: str
    size_bytes: int | None
    download_url: str


@dataclass(frozen=True)
class EvidenceRecord:
    """Preuve associée à une valeur extraite."""

    field_name: str
    value_text: str
    source_url: str
    extraction_method: str
    confidence: str


@dataclass
class RepositoryRecord:
    """Métadonnées générales d'un dépôt Hugging Face."""

    repository_id: str
    provider: str
    model_url: str
    author: str | None = None
    license: str | None = None
    pipeline_tag: str | None = None
    library_name: str | None = None
    downloads: int | None = None
    likes: int | None = None
    revision: str | None = None
    last_modified: str | None = None
    card_text: str = ""
    api_metadata: dict = field(default_factory=dict)
    files: list[RemoteFile] = field(default_factory=list)
    matched_queries: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class CandidateProfile:
    """Profil technique d'un checkpoint candidat."""

    repository_id: str
    checkpoint_path: str
    display_name: str
    architecture: str | None = None
    weights_available: bool = False
    weight_files: list[RemoteFile] = field(default_factory=list)
    pretrained_channel_count: int | None = None
    channel_mode: str = "inconnu"
    electrode_names: list[str] = field(default_factory=list)
    electrode_positions: list[tuple[float | None, float | None, float | None]] = (
        field(default_factory=list)
    )
    electrode_positions_available: bool | None = None
    sampling_frequency_hz: float | None = None
    window_duration_seconds: float | None = None
    number_of_timepoints: int | None = None
    preprocessing: str | None = None
    pretraining_dataset: str | None = None
    source_task: str | None = None
    asc_32_status: str = "à vérifier"
    compatibility_reason: str = "Métadonnées insuffisantes."
    compatibility_score: float = 0.0
    confidence: str = "faible"
    metadata_status: str = "incomplet"
    evidence: list[EvidenceRecord] = field(default_factory=list)
