"""Corrections humaines traçables pour les métadonnées ambiguës."""

from __future__ import annotations

import csv
from pathlib import Path

from .records import CandidateProfile, EvidenceRecord


FIELD_MAPPING = {
    "display_name": ("display_name", str),
    "architecture": ("architecture", str),
    "pretrained_channel_count": ("pretrained_channel_count", int),
    "channel_mode": ("channel_mode", str),
    "sampling_frequency_hz": ("sampling_frequency_hz", float),
    "window_duration_seconds": ("window_duration_seconds", float),
    "number_of_timepoints": ("number_of_timepoints", int),
    "preprocessing": ("preprocessing", str),
    "pretraining_dataset": ("pretraining_dataset", str),
    "source_task": ("source_task", str),
}


def apply_manual_overrides(
    profile: CandidateProfile,
    overrides_path: Path,
) -> CandidateProfile:
    """Applique seulement la ligne correspondant exactement au checkpoint."""

    if not overrides_path.exists():
        return profile

    with overrides_path.open(newline="", encoding="utf-8-sig") as file:
        rows = csv.DictReader(file)
        for row in rows:
            if row.get("repository_id", "").strip() != profile.repository_id:
                continue
            if row.get("checkpoint_path", "").strip() != profile.checkpoint_path:
                continue

            source_url = row.get("source_url", "").strip() or "manual://review"
            for csv_field, (attribute, converter) in FIELD_MAPPING.items():
                raw_value = row.get(csv_field, "").strip()
                if not raw_value:
                    continue
                converted_value = converter(raw_value)
                setattr(profile, attribute, converted_value)
                profile.evidence.append(
                    EvidenceRecord(
                        field_name=csv_field,
                        value_text=str(converted_value),
                        source_url=source_url,
                        extraction_method="manual_scientific_review",
                        confidence="forte",
                    )
                )

            electrode_text = row.get("electrode_names", "").strip()
            if electrode_text:
                profile.electrode_names = [
                    name.strip() for name in electrode_text.split("|") if name.strip()
                ]
                profile.electrode_positions_available = False
                profile.evidence.append(
                    EvidenceRecord(
                        field_name="electrode_names",
                        value_text=", ".join(profile.electrode_names),
                        source_url=source_url,
                        extraction_method="manual_scientific_review",
                        confidence="forte",
                    )
                )

            profile.metadata_status = "revu manuellement"
            break

    return profile

