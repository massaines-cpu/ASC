"""Configuration éditable des analyses demandées par Amel.

Ce fichier est le seul endroit où modifier les chemins et la liste des
expériences. Les autres scripts lisent ces constantes, ce qui évite de changer
plusieurs fichiers en même temps.
"""

from dataclasses import dataclass
import os
from pathlib import Path


# Projet ASC réel. Le pack livré reste séparé et ne modifie pas ce dossier.
PROJECT_ROOT = Path(
    os.environ.get("ASC_PROJECT_ROOT", "/Users/nini/ASC")
)
RESULTS_ROOT = PROJECT_ROOT / "results"
MODELS_ROOT = PROJECT_ROOT / "models"

# Toutes les nouvelles figures et tables sont regroupées ici.
REPORT_OUTPUT_ROOT = Path(
    os.environ.get(
        "ASC_REPORT_OUTPUT_ROOT",
        str(RESULTS_ROOT / "semaine_10"),
    )
)

EXPECTED_DYADS = (
    "J1",
    "J2",
    "J4",
    "J5",
    "J7",
    "J8",
    "J10",
    "J15",
)


@dataclass(frozen=True)
class ExperimentSpec:
    """Décrit une expérience déjà terminée et son libellé scientifique."""

    label: str
    result_directory_name: str
    family: str

    @property
    def result_directory(self) -> Path:
        return RESULTS_ROOT / self.result_directory_name


# ---------------------------------------------------------------------------
# Comparaison de la taille de la couche cachée du MLP.
# Une seule variable change : 128, 64 ou 32 neurones.
# ---------------------------------------------------------------------------
MLP_HIDDEN_SIZE_EXPERIMENTS = (
    ExperimentSpec(
        label="128 neurones",
        result_directory_name=(
            "data_final_non_linear_hidden_128_dropout_0.0_standardized"
        ),
        family="mlp_hidden_size",
    ),
    ExperimentSpec(
        label="64 neurones",
        result_directory_name=(
            "data_final_non_linear_hidden_64_dropout_0.0_standardized"
        ),
        family="mlp_hidden_size",
    ),
    ExperimentSpec(
        label="32 neurones",
        result_directory_name=(
            "data_final_non_linear_hidden_32_dropout_0.0_standardized"
        ),
        family="mlp_hidden_size",
    ),
)


# ---------------------------------------------------------------------------
# Comparaison du Dropout à architecture constante : MLP 32 neurones.
# Les valeurs réellement testées dans les résultats sont 0.0, 0.2 et 0.5.
# ---------------------------------------------------------------------------
MLP_DROPOUT_EXPERIMENTS = (
    ExperimentSpec(
        label="Dropout 0,0",
        result_directory_name=(
            "data_final_non_linear_hidden_32_dropout_0.0_standardized"
        ),
        family="mlp_dropout",
    ),
    ExperimentSpec(
        label="Dropout 0,2",
        result_directory_name=(
            "data_final_non_linear_hidden_32_dropout_0.2_standardized"
        ),
        family="mlp_dropout",
    ),
    ExperimentSpec(
        label="Dropout 0,5",
        result_directory_name=(
            "data_final_non_linear_hidden_32_dropout_0.5_standardized"
        ),
        family="mlp_dropout",
    ),
)


# ---------------------------------------------------------------------------
# Comparaison des architectures from scratch sur le même dataset A.
# Ajouter ici le MLP linéaire/non linéaire final si son nom diffère.
# ---------------------------------------------------------------------------
ARCHITECTURE_EXPERIMENTS = (
    ExperimentSpec(
        label="MLP linéaire",
        result_directory_name="experience_A_linear_standardized",
        family="architecture",
    ),
    ExperimentSpec(
        label="MLP non linéaire",
        result_directory_name=(
            "data_final_non_linear_hidden_32_dropout_0.0_standardized"
        ),
        family="architecture",
    ),
    ExperimentSpec(
        label="Petit CNN",
        result_directory_name="experience_A_small_cnn_standardized",
        family="architecture",
    ),
    ExperimentSpec(
        label="EEGNet",
        result_directory_name="experience_A_eegnet_standardized",
        family="architecture",
    ),
)


SIGNAL_JEPA_EXPERIMENTS = (
    ExperimentSpec(
        label="Pré-entraîné — classifieur",
        result_directory_name=(
            "data_signal_jepa_prelocal_19ch_128hz_2s_uv_"
            "signal_jepa_prelocal_pretrained_classifier_only_"
            "19ch_128hz_2s_microvolts_no_zscore_lr_0.005_"
            "batch_16_epochs_10_patience_5_seed_42"
        ),
        family="signal_jepa",
    ),
    ExperimentSpec(
        label="From scratch — complet",
        result_directory_name=(
            "data_signal_jepa_prelocal_19ch_128hz_2s_uv_"
            "signal_jepa_prelocal_scratch_full_finetuning_"
            "19ch_128hz_2s_microvolts_no_zscore_lr_0.005_"
            "batch_16_epochs_10_patience_5_seed_42"
        ),
        family="signal_jepa",
    ),
    ExperimentSpec(
        label="Pré-entraîné — complet",
        result_directory_name=(
            "data_signal_jepa_prelocal_19ch_128hz_2s_uv_"
            "signal_jepa_prelocal_pretrained_full_finetuning_"
            "19ch_128hz_2s_microvolts_no_zscore_lr_0.005_"
            "batch_16_epochs_10_patience_5_seed_42"
        ),
        family="signal_jepa",
    ),
)


# ---------------------------------------------------------------------------
# 19 canaux (montage réduit, tutoriel) vs 32 canaux (montage ASC complet),
# même variante (scratch/full_finetuning) pour isoler l'effet du nombre de
# canaux — SignalJEPA_Contextual à 32 canaux ne convergeait pas
# (déséquilibre structurel documenté dans signal_jepa_model.py), d'où le
# passage à SignalJEPA_PreLocal, qui n'a jamais dépendu du nombre de canaux.
# ---------------------------------------------------------------------------
SIGNAL_JEPA_CHANNELS_EXPERIMENTS = (
    ExperimentSpec(
        label="PreLocal 19ch — scratch",
        result_directory_name=(
            "data_signal_jepa_prelocal_19ch_128hz_2s_uv_"
            "signal_jepa_prelocal_scratch_full_finetuning_"
            "19ch_128hz_2s_microvolts_no_zscore_lr_0.005_"
            "batch_16_epochs_10_patience_5_seed_42"
        ),
        family="signal_jepa_channels",
    ),
    ExperimentSpec(
        label="PreLocal 32ch — scratch",
        result_directory_name=(
            "data_signal_jepa_prelocal_32ch_128hz_2s_uv_"
            "signal_jepa_prelocal_scratch_full_finetuning_"
            "32ch_128hz_2s_microvolts_no_zscore_lr_0.005_"
            "batch_16_epochs_10_patience_5_seed_42"
        ),
        family="signal_jepa_channels",
    ),
)


# ---------------------------------------------------------------------------
# Les trois stratégies de transfert, montage ASC complet (32 canaux) —
# équivalent de SIGNAL_JEPA_EXPERIMENTS (19ch) pour comparer scratch,
# pré-entraîné gelé et pré-entraîné entièrement affiné.
# ---------------------------------------------------------------------------
SIGNAL_JEPA_32CH_EXPERIMENTS = (
    ExperimentSpec(
        label="Pré-entraîné — classifieur",
        result_directory_name=(
            "data_signal_jepa_prelocal_32ch_128hz_2s_uv_"
            "signal_jepa_prelocal_pretrained_classifier_only_"
            "32ch_128hz_2s_microvolts_no_zscore_lr_0.005_"
            "batch_16_epochs_10_patience_5_seed_42"
        ),
        family="signal_jepa_32ch",
    ),
    ExperimentSpec(
        label="From scratch — complet",
        result_directory_name=(
            "data_signal_jepa_prelocal_32ch_128hz_2s_uv_"
            "signal_jepa_prelocal_scratch_full_finetuning_"
            "32ch_128hz_2s_microvolts_no_zscore_lr_0.005_"
            "batch_16_epochs_10_patience_5_seed_42"
        ),
        family="signal_jepa_32ch",
    ),
    ExperimentSpec(
        label="Pré-entraîné — complet",
        result_directory_name=(
            "data_signal_jepa_prelocal_32ch_128hz_2s_uv_"
            "signal_jepa_prelocal_pretrained_full_finetuning_"
            "32ch_128hz_2s_microvolts_no_zscore_lr_0.005_"
            "batch_16_epochs_10_patience_5_seed_42"
        ),
        family="signal_jepa_32ch",
    ),
)


COMPARISON_GROUPS = {
    "mlp_hidden_size": MLP_HIDDEN_SIZE_EXPERIMENTS,
    "mlp_dropout": MLP_DROPOUT_EXPERIMENTS,
    "architectures": ARCHITECTURE_EXPERIMENTS,
    "signal_jepa": SIGNAL_JEPA_EXPERIMENTS,
    "signal_jepa_channels": SIGNAL_JEPA_CHANNELS_EXPERIMENTS,
    "signal_jepa_32ch": SIGNAL_JEPA_32CH_EXPERIMENTS,
}


# ---------------------------------------------------------------------------
# Paramètres qui doivent être rappelés sous les figures.
# Les scripts essaient d'abord de lire experiment_config.json. Ces valeurs
# servent de repli pour les anciens résultats qui ne possèdent pas ce fichier.
# ---------------------------------------------------------------------------
DEFAULT_EARLY_STOPPING_PATIENCE = 15
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-4
EARLY_STOPPING_MONITOR = "validation_loss"
CHECKPOINT_SELECTION = "minimum de validation_loss"


# ---------------------------------------------------------------------------
# Configuration du futur transfert EEGNet.
# Le chemin reste None tant qu'un checkpoint n'a pas été inspecté et validé.
# ---------------------------------------------------------------------------
EEGNET_CHECKPOINT_PATH: Path | None = None
EEGNET_CHECKPOINT_NAME = "à renseigner après audit de compatibilité"

# Si le checkpoint et le modèle ASC utilisent des noms différents pour une
# même couche, écrire la correspondance explicite ici :
# "nom_dans_le_checkpoint": "nom_dans_eegnet_asc"
# Une correspondance n'est acceptée que si les formes sont aussi identiques.
CHECKPOINT_KEY_RENAMES: dict[str, str] = {}

EEGNET_TRANSFER_STRATEGIES = (
    "full_finetuning",
    "freeze_temporal",
    "freeze_temporal_spatial",
    "classifier_only",
)

# L'expérience courte sert uniquement à détecter une erreur technique.
# Pour un résultat scientifique, utiliser les huit dyades.
TRANSFER_SELECTED_DYADS = list(EXPECTED_DYADS)
TRANSFER_BATCH_SIZE = 5
TRANSFER_MAXIMUM_EPOCHS = 100
TRANSFER_PATIENCE = 15
TRANSFER_LEARNING_RATE = 1e-3
TRANSFER_RANDOM_SEED = 42
TRANSFER_DEVICE = "mps"
TRANSFER_DATASET_VERSION = "data_final"
TRANSFER_STANDARDIZE = True


# ---------------------------------------------------------------------------
# Dataset public éventuel pour pré-entraîner EEGNet.
# ---------------------------------------------------------------------------
PUBLIC_DATASET_ROOT: Path | None = None
PUBLIC_DATASET_MANIFEST: Path | None = None
PUBLIC_PRETRAINING_OUTPUT = PROJECT_ROOT / "models" / "public_pretraining"
