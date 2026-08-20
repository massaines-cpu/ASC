"""Extraction des caractéristiques scientifiques des checkpoints EEG.

L'extraction est volontairement conservatrice. Une valeur n'est conservée que
si une clé explicite ou une phrase suffisamment précise la documente. Quand les
sources se contredisent, la configuration située près du checkpoint est
prioritaire sur une description générale du dataset dans le README.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import pickletools
from pathlib import PurePosixPath
import re
from typing import Any, Iterable

import yaml

from .huggingface_provider import CandidateDocuments
from .records import CandidateProfile, EvidenceRecord, RepositoryRecord


CHANNEL_COUNT_KEYS = {
    "n_chans", "n_channels", "in_chans", "in_channels", "num_channels",
    "number_of_channels", "eeg_channels", "n_eeg_channels",
}
CHANNEL_NAME_KEYS = {
    "ch_names", "channel_names", "electrode_names", "electrodes",
    "eeg_channel_names", "channels",
}
CHANNEL_INFO_KEYS = {"chs_info", "channels_info", "electrode_info"}
SAMPLING_FREQUENCY_KEYS = {
    "sfreq", "sampling_frequency", "sampling_frequency_hz", "sampling_rate",
    "sample_rate", "fs", "resample_freq",
}
TIMEPOINT_KEYS = {
    "n_times", "n_timepoints", "number_of_timepoints", "input_window_samples",
    "window_samples", "sequence_length", "input_length",
}
WINDOW_DURATION_KEYS = {
    "window_duration", "window_duration_seconds", "window_size_seconds",
    "input_window_seconds", "duration_seconds", "trial_duration",
}
ARCHITECTURE_KEYS = {
    "architecture", "architectures", "model_type", "model_name", "network",
}
DATASET_KEYS = {
    "dataset", "dataset_name", "datasets", "pretraining_dataset",
    "pretrain_dataset", "source_dataset",
}
PREPROCESSING_KEYS = {
    "preprocessing", "preprocess", "normalization", "standardization",
    "filtering", "bandpass", "band_pass",
}
TASK_KEYS = {"task", "task_name", "source_task", "paradigm"}
VARIABLE_CHANNEL_KEYS = {
    "channel_mode", "channels_are_variable", "variable_channels",
    "supports_variable_channels", "channel_embedding",
}


@dataclass(frozen=True)
class ExtractedValue:
    value: Any
    source_url: str
    method: str
    confidence: str
    priority: int


def _normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def _walk_mapping(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[str, Any]]:
    """Parcourt récursivement JSON sans supposer une architecture précise."""

    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = _normalize_key(key)
            yield normalized_key, child
            yield from _walk_mapping(child, path + (normalized_key,))
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mapping(child, path)


def _simple_scalar(value: Any) -> str | int | float | bool | None:
    if isinstance(value, (str, int, float, bool)) and not isinstance(value, complex):
        return value
    return None


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return [item.strip() for item in value if item.strip()]


def extract_pickle_scalars(content: bytes) -> dict[str, str | int | float | bool]:
    """Lit les couples simples d'un pickle sans exécuter le pickle.

    ``pickle.load`` pourrait exécuter du code contenu dans un fichier distant.
    ``pickletools`` ne fait qu'inspecter les opcodes. Cette fonction suffit pour
    les fichiers ``kwargs.pkl`` courants contenant des clés et valeurs simples.
    """

    allowed_keys = (
        CHANNEL_COUNT_KEYS
        | SAMPLING_FREQUENCY_KEYS
        | TIMEPOINT_KEYS
        | WINDOW_DURATION_KEYS
        | VARIABLE_CHANNEL_KEYS
    )
    extracted: dict[str, str | int | float | bool] = {}
    pending_key: str | None = None

    try:
        operations = pickletools.genops(content)
        for opcode, argument, _ in operations:
            if opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}:
                normalized = _normalize_key(argument)
                pending_key = normalized if normalized in allowed_keys else None
                continue

            if pending_key is None:
                continue

            if opcode.name in {
                "BININT", "BININT1", "BININT2", "INT", "LONG", "LONG1",
                "LONG4", "BINFLOAT", "NEWTRUE", "NEWFALSE",
            }:
                if opcode.name == "NEWTRUE":
                    extracted[pending_key] = True
                elif opcode.name == "NEWFALSE":
                    extracted[pending_key] = False
                else:
                    extracted[pending_key] = argument
                pending_key = None
    except Exception:
        return {}

    return extracted


class MetadataExtractor:
    """Transforme documents et fiche du dépôt en un profil candidat."""

    def extract(
        self,
        repository: RepositoryRecord,
        candidate_documents: CandidateDocuments,
        weight_files,
    ) -> CandidateProfile:
        checkpoint_path = candidate_documents.checkpoint_path
        display_suffix = (
            repository.repository_id
            if checkpoint_path == "."
            else f"{repository.repository_id}/{checkpoint_path}"
        )
        profile = CandidateProfile(
            repository_id=repository.repository_id,
            checkpoint_path=checkpoint_path,
            display_name=display_suffix,
            weights_available=bool(weight_files),
            weight_files=list(weight_files),
        )

        fields: dict[str, list[ExtractedValue]] = {
            "channel_count": [],
            "channel_mode": [],
            "electrode_names": [],
            "electrode_positions": [],
            "sampling_frequency": [],
            "timepoints": [],
            "window_duration": [],
            "architecture": [],
            "dataset": [],
            "preprocessing": [],
            "task": [],
        }

        for path, content in candidate_documents.documents.items():
            source_url = candidate_documents.document_urls[path]
            suffix = PurePosixPath(path).suffix.lower()
            name = PurePosixPath(path).name.lower()
            near_checkpoint = (
                str(PurePosixPath(path).parent) == checkpoint_path
                and checkpoint_path != "."
            )
            priority = 100 if near_checkpoint else 70

            if suffix == ".json":
                self._extract_json(content, source_url, priority, fields)
            elif suffix in (".yaml", ".yml"):
                self._extract_yaml(content, source_url, priority, fields)
            elif suffix == ".pkl" and "kwargs" in name:
                self._extract_pickle(content, source_url, priority + 5, fields)
            elif name == "readme.md":
                text = content.decode("utf-8", errors="replace")
                self._extract_readme(text, source_url, fields)

        # L'API expose parfois une configuration ou le nom du dataset alors
        # qu'aucun fichier JSON séparé n'est présent dans le dépôt.
        api_source = repository.model_url
        api_config = repository.api_metadata.get("config")
        if isinstance(api_config, dict) and api_config:
            self._extract_mapping(
                api_config,
                api_source,
                priority=55,
                fields=fields,
                method_prefix="huggingface_api_config",
            )
        api_datasets = repository.api_metadata.get("datasets")
        dataset_text = self._text_value(api_datasets)
        if dataset_text:
            fields["dataset"].append(
                ExtractedValue(
                    dataset_text,
                    api_source,
                    "huggingface_model_card_metadata",
                    "moyenne",
                    50,
                )
            )

        self._apply_fields(profile, fields)

        # Informations générales du model card renvoyées par l'API. Elles sont
        # moins spécifiques qu'un fichier de configuration du checkpoint.
        if repository.card_text and not any(
            path == "README.md" for path in candidate_documents.documents
        ):
            self._extract_readme(
                repository.card_text,
                repository.model_url,
                fields,
            )
            self._apply_fields(profile, fields)

        profile.metadata_status = self._metadata_status(profile)
        return profile

    def _extract_json(
        self,
        content: bytes,
        source_url: str,
        priority: int,
        fields: dict[str, list[ExtractedValue]],
    ) -> None:
        try:
            data = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return

        self._extract_mapping(
            data,
            source_url,
            priority,
            fields,
            method_prefix="json",
        )

    def _extract_yaml(
        self,
        content: bytes,
        source_url: str,
        priority: int,
        fields: dict[str, list[ExtractedValue]],
    ) -> None:
        try:
            data = yaml.safe_load(content.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError):
            return

        if not isinstance(data, dict):
            return

        self._extract_mapping(
            data,
            source_url,
            priority,
            fields,
            method_prefix="yaml",
        )

    def _extract_mapping(
        self,
        data: dict,
        source_url: str,
        priority: int,
        fields: dict[str, list[ExtractedValue]],
        method_prefix: str,
    ) -> None:
        """Applique les mêmes règles aux JSON et à la config de l'API."""

        for key, value in _walk_mapping(data):
            scalar = _simple_scalar(value)

            if key in CHANNEL_COUNT_KEYS and isinstance(scalar, (int, float)):
                fields["channel_count"].append(
                    ExtractedValue(int(scalar), source_url, f"{method_prefix}:{key}", "forte", priority)
                )
            elif key in CHANNEL_NAME_KEYS:
                names = _string_list(value)
                if names:
                    fields["electrode_names"].append(
                        ExtractedValue(names, source_url, f"{method_prefix}:{key}", "forte", priority)
                    )
            elif key in CHANNEL_INFO_KEYS:
                names, positions = self._channel_information(value)
                if names:
                    fields["electrode_names"].append(
                        ExtractedValue(
                            names,
                            source_url,
                            f"{method_prefix}:{key}:names",
                            "forte",
                            priority,
                        )
                    )
                    fields["electrode_positions"].append(
                        ExtractedValue(
                            positions,
                            source_url,
                            f"{method_prefix}:{key}:positions",
                            "forte",
                            priority,
                        )
                    )
                    fields["channel_count"].append(
                        ExtractedValue(
                            len(names),
                            source_url,
                            f"{method_prefix}:{key}:length",
                            "forte",
                            priority,
                        )
                    )
            elif key in SAMPLING_FREQUENCY_KEYS and isinstance(scalar, (int, float)):
                fields["sampling_frequency"].append(
                    ExtractedValue(float(scalar), source_url, f"{method_prefix}:{key}", "forte", priority)
                )
            elif key in TIMEPOINT_KEYS and isinstance(scalar, (int, float)):
                fields["timepoints"].append(
                    ExtractedValue(int(scalar), source_url, f"{method_prefix}:{key}", "forte", priority)
                )
            elif key in WINDOW_DURATION_KEYS and isinstance(scalar, (int, float)):
                fields["window_duration"].append(
                    ExtractedValue(float(scalar), source_url, f"{method_prefix}:{key}", "forte", priority)
                )
            elif key in ARCHITECTURE_KEYS:
                value_text = self._text_value(value)
                if value_text:
                    fields["architecture"].append(
                        ExtractedValue(value_text, source_url, f"{method_prefix}:{key}", "forte", priority)
                    )
            elif key in DATASET_KEYS:
                value_text = self._text_value(value)
                if value_text:
                    fields["dataset"].append(
                        ExtractedValue(value_text, source_url, f"{method_prefix}:{key}", "moyenne", priority)
                    )
            elif key in PREPROCESSING_KEYS:
                value_text = self._text_value(value)
                if value_text:
                    fields["preprocessing"].append(
                        ExtractedValue(value_text, source_url, f"{method_prefix}:{key}", "moyenne", priority)
                    )
            elif key in TASK_KEYS:
                value_text = self._text_value(value)
                if value_text:
                    fields["task"].append(
                        ExtractedValue(value_text, source_url, f"{method_prefix}:{key}", "moyenne", priority)
                    )
            elif key in VARIABLE_CHANNEL_KEYS and scalar is not None:
                mode = self._channel_mode_from_value(scalar)
                if mode:
                    fields["channel_mode"].append(
                        ExtractedValue(mode, source_url, f"{method_prefix}:{key}", "forte", priority)
                    )

    def _extract_pickle(
        self,
        content: bytes,
        source_url: str,
        priority: int,
        fields: dict[str, list[ExtractedValue]],
    ) -> None:
        for key, value in extract_pickle_scalars(content).items():
            if key in CHANNEL_COUNT_KEYS and isinstance(value, (int, float)):
                field_name = "channel_count"
                cast_value: Any = int(value)
            elif key in SAMPLING_FREQUENCY_KEYS and isinstance(value, (int, float)):
                field_name = "sampling_frequency"
                cast_value = float(value)
            elif key in TIMEPOINT_KEYS and isinstance(value, (int, float)):
                field_name = "timepoints"
                cast_value = int(value)
            elif key in WINDOW_DURATION_KEYS and isinstance(value, (int, float)):
                field_name = "window_duration"
                cast_value = float(value)
            elif key in VARIABLE_CHANNEL_KEYS:
                field_name = "channel_mode"
                cast_value = self._channel_mode_from_value(value)
                if cast_value is None:
                    continue
            else:
                continue

            fields[field_name].append(
                ExtractedValue(
                    cast_value,
                    source_url,
                    f"pickletools:{key}",
                    "forte",
                    priority,
                )
            )

    def _extract_readme(
        self,
        text: str,
        source_url: str,
        fields: dict[str, list[ExtractedValue]],
    ) -> None:
        """Extrait uniquement des formulations explicites du README."""

        patterns = {
            "channel_count": (
                r"(?i)(?:n[_ -]?chans?|number of channels?|EEG channels?)"
                r"\s*[:=]\s*(\d{1,3})",
                int,
            ),
            "sampling_frequency": (
                r"(?i)(?:sfreq|sampling (?:frequency|rate)|resampl(?:ed|ing))"
                r"\s*(?:to|at|[:=])?\s*(\d+(?:\.\d+)?)\s*Hz",
                float,
            ),
            "timepoints": (
                r"(?i)(?:input[_ -]?window[_ -]?samples|n[_ -]?times|timepoints?)"
                r"\s*[:=]\s*(\d+)",
                int,
            ),
            "window_duration": (
                r"(?i)(?:window|epoch|trial)\s+(?:duration|length|size)?"
                r"\s*(?:of|[:=])?\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b",
                float,
            ),
        }
        documented_channel_counts = {
            int(value)
            for value in re.findall(r"(?i)\b(\d{1,3})\s+EEG channels?\b", text)
        }
        if len(documented_channel_counts) == 1:
            fields["channel_count"].append(
                ExtractedValue(
                    next(iter(documented_channel_counts)),
                    source_url,
                    "readme_eeg_channel_count",
                    "moyenne",
                    40,
                )
            )

        dataset_match = re.search(
            r"(?i)pre[- ]trained on (?:the )?([A-Za-z0-9_.-]+) dataset",
            text,
        )
        if dataset_match:
            fields["dataset"].append(
                ExtractedValue(
                    dataset_match.group(1),
                    source_url,
                    "readme_pretraining_dataset",
                    "moyenne",
                    45,
                )
            )
        for field_name, (pattern, caster) in patterns.items():
            values = {caster(match) for match in re.findall(pattern, text)}
            # Plusieurs valeurs dans un README peuvent décrire plusieurs sous-
            # modèles. Dans ce cas, aucune valeur n'est attribuée arbitrairement.
            if len(values) == 1:
                value = next(iter(values))
                fields[field_name].append(
                    ExtractedValue(value, source_url, "readme_regex", "faible", 30)
                )

        # Recherche d'une liste explicite introduite par un libellé de canaux.
        channel_line = re.search(
            r"(?im)^(?:channels?|ch_names|electrodes?)\s*[:=]\s*([^\n]+)$",
            text,
        )
        if channel_line:
            names = [
                item.strip(" `[]()")
                for item in re.split(r"[,;|]", channel_line.group(1))
                if item.strip(" `[]()")
            ]
            if 1 < len(names) <= 256:
                fields["electrode_names"].append(
                    ExtractedValue(names, source_url, "readme_channel_list", "moyenne", 45)
                )

        # Quelques indices textuels sont conservés, sans prétendre reconstruire
        # tout le protocole expérimental depuis du texte libre.
        preprocessing_sentences = []
        for line in text.splitlines():
            cleaned = line.strip(" -*#\t")
            if len(cleaned) > 240:
                continue
            lower = cleaned.lower()
            if any(word in lower for word in (
                "bandpass", "band-pass", "notch", "resampl", "normaliz",
                "standardiz", "microvolt", "µv", "filter",
            )):
                preprocessing_sentences.append(cleaned)
        if preprocessing_sentences:
            unique_sentences = list(dict.fromkeys(preprocessing_sentences))[:4]
            fields["preprocessing"].append(
                ExtractedValue(
                    " | ".join(unique_sentences),
                    source_url,
                    "readme_preprocessing_lines",
                    "faible",
                    25,
                )
            )

    def _apply_fields(
        self,
        profile: CandidateProfile,
        fields: dict[str, list[ExtractedValue]],
    ) -> None:
        winners = {
            field_name: self._best(values)
            for field_name, values in fields.items()
        }

        assignments = {
            "channel_count": "pretrained_channel_count",
            "channel_mode": "channel_mode",
            "sampling_frequency": "sampling_frequency_hz",
            "timepoints": "number_of_timepoints",
            "window_duration": "window_duration_seconds",
            "architecture": "architecture",
            "dataset": "pretraining_dataset",
            "preprocessing": "preprocessing",
            "task": "source_task",
        }
        for field_name, attribute_name in assignments.items():
            winner = winners[field_name]
            if winner is not None:
                setattr(profile, attribute_name, winner.value)

        electrode_winner = winners["electrode_names"]
        if electrode_winner is not None:
            profile.electrode_names = list(electrode_winner.value)
            profile.electrode_positions_available = False
            if profile.pretrained_channel_count is None:
                profile.pretrained_channel_count = len(profile.electrode_names)

        positions_winner = winners["electrode_positions"]
        if positions_winner is not None:
            profile.electrode_positions = list(positions_winner.value)
            profile.electrode_positions_available = any(
                any(coordinate is not None for coordinate in position)
                for position in profile.electrode_positions
            )

        if profile.channel_mode == "inconnu" and profile.pretrained_channel_count:
            # Une dimension ``in_chans`` dans les poids/configs est fixe pour le
            # checkpoint étudié, sauf documentation explicite du contraire.
            profile.channel_mode = "fixes"

        # Si durée, fréquence et nombre de points sont cohérents, une valeur
        # manquante peut être calculée et sa nature dérivée est explicitée.
        self._derive_temporal_metadata(profile, winners)

        existing_evidence = {
            (item.field_name, item.value_text, item.source_url, item.extraction_method)
            for item in profile.evidence
        }
        for field_name, winner in winners.items():
            if winner is None:
                continue
            value_text = self._text_value(winner.value) or str(winner.value)
            key = (field_name, value_text, winner.source_url, winner.method)
            if key in existing_evidence:
                continue
            profile.evidence.append(
                EvidenceRecord(
                    field_name=field_name,
                    value_text=value_text,
                    source_url=winner.source_url,
                    extraction_method=winner.method,
                    confidence=winner.confidence,
                )
            )
            existing_evidence.add(key)

    @staticmethod
    def _best(values: list[ExtractedValue]) -> ExtractedValue | None:
        if not values:
            return None
        confidence_rank = {"faible": 0, "moyenne": 1, "forte": 2}
        return max(values, key=lambda item: (item.priority, confidence_rank[item.confidence]))

    @staticmethod
    def _text_value(value: Any) -> str | None:
        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, (list, tuple)) and value:
            return ", ".join(str(item) for item in value)
        if isinstance(value, dict) and value:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return None

    @staticmethod
    def _channel_mode_from_value(value: Any) -> str | None:
        if isinstance(value, bool):
            return "variables" if value else "fixes"
        normalized = str(value).strip().lower()
        if normalized in {
            "variable", "variables", "dynamic", "true", "yes",
            "pretrain_aligned", "scratch",
        }:
            return "variables"
        if normalized in {"fixed", "fixe", "fixes", "false", "no"}:
            return "fixes"
        return None

    @staticmethod
    def _channel_information(
        value: Any,
    ) -> tuple[list[str], list[tuple[float | None, float | None, float | None]]]:
        """Extrait noms et coordonnées depuis le format MNE/Braindecode."""

        if not isinstance(value, list):
            return [], []

        names: list[str] = []
        positions: list[tuple[float | None, float | None, float | None]] = []
        for channel in value:
            if not isinstance(channel, dict):
                return [], []
            name = channel.get("ch_name") or channel.get("name")
            if not isinstance(name, str) or not name.strip():
                return [], []
            location = channel.get("loc") or channel.get("position")
            if isinstance(location, list) and len(location) >= 3:
                xyz = tuple(
                    float(coordinate) if isinstance(coordinate, (int, float)) else None
                    for coordinate in location[:3]
                )
            else:
                xyz = (None, None, None)
            names.append(name.strip())
            positions.append(xyz)
        return names, positions

    @staticmethod
    def _derive_temporal_metadata(profile: CandidateProfile, winners) -> None:
        source_url = next(
            (
                winner.source_url
                for key in ("timepoints", "sampling_frequency", "window_duration")
                if (winner := winners.get(key)) is not None
            ),
            "derived://temporal-metadata",
        )
        if (
            profile.window_duration_seconds is None
            and profile.number_of_timepoints
            and profile.sampling_frequency_hz
        ):
            profile.window_duration_seconds = (
                profile.number_of_timepoints / profile.sampling_frequency_hz
            )
            profile.evidence.append(
                EvidenceRecord(
                    "window_duration",
                    str(profile.window_duration_seconds),
                    source_url,
                    "calculated:n_timepoints/sfreq",
                    "moyenne",
                )
            )

    @staticmethod
    def _metadata_status(profile: CandidateProfile) -> str:
        present = sum(
            value not in (None, [], "", "inconnu")
            for value in (
                profile.pretrained_channel_count,
                profile.channel_mode,
                profile.electrode_names,
                profile.sampling_frequency_hz,
                profile.window_duration_seconds,
                profile.preprocessing,
                profile.pretraining_dataset,
            )
        )
        if present >= 6:
            return "complet"
        if present >= 3:
            return "partiel"
        return "incomplet"
