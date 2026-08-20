"""Règles transparentes de compatibilité entre un checkpoint et ASC."""

from __future__ import annotations

from . import config
from .records import CandidateProfile, EvidenceRecord


def _normalized_electrode(name: str) -> str:
    """Compare les noms sans être sensible à la casse ou aux espaces."""

    return "".join(character for character in name.lower() if character.isalnum())


def evaluate_asc_compatibility(profile: CandidateProfile) -> CandidateProfile:
    """Calcule statut et score sans modifier les poids du modèle.

    Le statut répond strictement à la question « ce checkpoint peut-il recevoir
    les 32 canaux ASC ? ». La fréquence et la durée influencent le score global,
    mais pas cette décision spatiale : elles peuvent être adaptées par un
    prétraitement distinct et documenté.
    """

    expected_names = [_normalized_electrode(name) for name in config.ASC_CHANNEL_NAMES]
    actual_names = [_normalized_electrode(name) for name in profile.electrode_names]

    if not profile.weights_available:
        status = "non"
        reason = "Aucun fichier de poids réellement disponible n'a été détecté."
    elif profile.channel_mode == "variables":
        if profile.electrode_names and not set(expected_names).issubset(
            set(actual_names)
        ):
            status = "à adapter"
            reason = (
                "Le modèle accepte un sous-ensemble variable de son montage, "
                "mais au moins une électrode ASC est absente de la table "
                "pré-entraînée. Une nouvelle représentation spatiale est requise."
            )
        elif profile.electrode_names:
            status = "oui"
            reason = (
                "Le modèle accepte un sous-ensemble variable de ses électrodes "
                "pré-entraînées et les 32 noms ASC figurent tous dans sa table."
            )
        else:
            status = "oui"
            reason = (
                "Le modèle documente des canaux variables et une représentation "
                "spatiale pouvant être initialisée pour le montage ASC."
            )
    elif profile.pretrained_channel_count is None:
        status = "à vérifier"
        reason = (
            "Les poids existent, mais leur nombre réel de canaux d'entrée n'est "
            "pas documenté avec assez de précision."
        )
    elif profile.pretrained_channel_count != config.ASC_NUMBER_OF_CHANNELS:
        status = "à adapter"
        reason = (
            f"Le checkpoint attend {profile.pretrained_channel_count} canaux, "
            f"contre {config.ASC_NUMBER_OF_CHANNELS} pour ASC. Une sélection, "
            "une projection ou une nouvelle couche spatiale serait nécessaire."
        )
    elif not profile.electrode_names:
        status = "à vérifier"
        reason = (
            "Le checkpoint annonce 32 canaux, mais leurs noms et leur ordre ne "
            "sont pas documentés ; la compatibilité spatiale n'est pas prouvée."
        )
    elif actual_names == expected_names:
        status = "oui"
        reason = (
            "Le checkpoint utilise les 32 électrodes ASC dans le même ordre."
        )
    elif set(actual_names) == set(expected_names):
        status = "à adapter"
        reason = (
            "Les 32 noms correspondent à ASC, mais leur ordre diffère : un "
            "réordonnancement déterministe des canaux est requis."
        )
    else:
        status = "à adapter"
        reason = (
            "Le checkpoint utilise 32 canaux, mais leur montage ne correspond "
            "pas exactement aux 32 électrodes ASC."
        )

    score = 0.0
    if profile.weights_available:
        score += 25
    if profile.channel_mode == "variables":
        score += 20
    elif profile.pretrained_channel_count == config.ASC_NUMBER_OF_CHANNELS:
        score += 18
    elif profile.pretrained_channel_count is not None:
        score += max(0, 8 - abs(profile.pretrained_channel_count - 32) * 0.25)

    if actual_names == expected_names and actual_names:
        score += 25
    elif set(actual_names) == set(expected_names) and actual_names:
        score += 20
    elif actual_names:
        overlap = len(set(actual_names) & set(expected_names))
        score += 15 * overlap / config.ASC_NUMBER_OF_CHANNELS

    if profile.sampling_frequency_hz is not None:
        score += 8
    if profile.window_duration_seconds is not None:
        score += 7
    if profile.preprocessing:
        score += 7
    if profile.pretraining_dataset:
        score += 5
    if profile.source_task:
        score += 3
    if profile.architecture:
        score += 2

    if status == "non":
        score = min(score, 10)

    strong_evidence = sum(item.confidence == "forte" for item in profile.evidence)
    medium_evidence = sum(item.confidence == "moyenne" for item in profile.evidence)
    if status == "oui" and strong_evidence >= 2:
        confidence = "forte"
    elif strong_evidence >= 1 or medium_evidence >= 2:
        confidence = "moyenne"
    else:
        confidence = "faible"

    profile.asc_32_status = status
    profile.compatibility_reason = reason
    profile.compatibility_score = round(min(score, 100.0), 2)
    profile.confidence = confidence
    profile.evidence.append(
        EvidenceRecord(
            field_name="asc_32_status",
            value_text=status,
            source_url="rule://asc-32-channel-compatibility-v1",
            extraction_method="deterministic_compatibility_rule",
            confidence=confidence,
        )
    )
    return profile
