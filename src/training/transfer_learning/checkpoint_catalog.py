"""Crée le tableau de recherche des checkpoints EEG candidats.

Les valeurs inconnues restent explicitement « à vérifier ». Une ligne n'est
considérée comme exploitable qu'après consultation de la carte du modèle, de
l'article ou du code d'entraînement associé.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.config.settings import REPORT_OUTPUT_ROOT


@dataclass(frozen=True)
class CheckpointCandidate:
    """Informations nécessaires pour justifier un transfert EEG."""

    name: str
    source: str
    architecture: str
    number_of_channels: str
    sampling_frequency_hz: str
    window_duration_seconds: str
    source_paradigm: str
    source_dataset: str
    license: str
    expected_adaptations: str
    documentation_status: str
    decision: str


CANDIDATES = (
    CheckpointCandidate(
        name="Guido151 / EEGNetv4",
        source="Hugging Face — checkpoint déjà étudié dans ASC",
        architecture="EEGNetv4 / Braindecode",
        number_of_channels="19 dans l'expérience source à confirmer",
        sampling_frequency_hz="à vérifier dans la carte du modèle",
        window_duration_seconds="à vérifier",
        source_paradigm="ERP à confirmer",
        source_dataset="à vérifier",
        license="à vérifier",
        expected_adaptations=(
            "Classifieur remplacé ; convolution spatiale incompatible si le "
            "nombre ou l'ordre des canaux diffère."
        ),
        documentation_status="partiel",
        decision="référence déjà testée, compatibilité insuffisante",
    ),
    CheckpointCandidate(
        name="PierreGtch / EEGNetv4 — Cho2017",
        source="Candidat cité dans le compte rendu ; URL exacte à enregistrer",
        architecture="EEGNetv4 / Braindecode",
        number_of_channels="à vérifier dans le checkpoint",
        sampling_frequency_hz="512 annoncé dans les notes, à confirmer",
        window_duration_seconds="à vérifier",
        source_paradigm="imagerie motrice probable, à confirmer",
        source_dataset="Cho2017, à confirmer",
        license="à vérifier",
        expected_adaptations=(
            "Vérifier noms et formes de tous les poids ; remplacer le "
            "classifieur source par la tête binaire YO/YF."
        ),
        documentation_status="non vérifié",
        decision="candidat prioritaire à auditer",
    ),
    CheckpointCandidate(
        name="Pré-entraînement ASC sur dataset public 32 canaux",
        source="Checkpoint produit localement avec pretrain_eegnet_public.py",
        architecture="Même EEGNet que le modèle cible ASC",
        number_of_channels="32 exigés par le contrat de préparation",
        sampling_frequency_hz="enregistrée dans le manifeste",
        window_duration_seconds="enregistrée dans le manifeste",
        source_paradigm="dépend du dataset public retenu",
        source_dataset="WAY-EEG-GAL ou autre candidat documenté",
        license="à vérifier avant téléchargement et redistribution",
        expected_adaptations="Classifieur source ignoré ; backbone compatible",
        documentation_status="pipeline prêt, dataset à sélectionner",
        decision="solution de repli maîtrisée",
    ),
)


def candidate_is_documented(candidate: CheckpointCandidate) -> bool:
    """Refuse de marquer prêt un candidat contenant une information inconnue."""

    values = [str(value).lower() for value in asdict(candidate).values()]
    unknown_markers = ("à vérifier", "à confirmer", "non vérifié")
    return not any(
        marker in value
        for value in values
        for marker in unknown_markers
    )


def main() -> None:
    """Sauvegarde le tableau de compatibilité à compléter."""

    rows = []
    for candidate in CANDIDATES:
        row = asdict(candidate)
        row["ready_for_transfer_experiment"] = candidate_is_documented(
            candidate
        )
        rows.append(row)

    table = pd.DataFrame(rows)
    output_directory = REPORT_OUTPUT_ROOT / "transfer_learning"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "checkpoint_candidates.csv"
    table.to_csv(output_path, index=False)

    print(table.to_string(index=False))
    print(f"\nTableau sauvegardé : {output_path}")
    print(
        "Une valeur False dans ready_for_transfer_experiment signifie que "
        "la documentation doit être complétée avant l'entraînement."
    )


if __name__ == "__main__":
    main()

