"""Configuration unique de la recherche de modèles EEG.

Les choix sont écrits ici afin de lancer le catalogue depuis PyCharm sans
arguments de terminal. Modifier une recherche ne demande donc pas de toucher au
code de la base de données ou aux extracteurs.
"""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_ROOT = PACKAGE_ROOT / "registry"
DATABASE_PATH = PACKAGE_ROOT / "database" / "eeg_model_catalog.sqlite3"
SCHEMA_PATH = REGISTRY_ROOT / "schema.sql"
EXPORT_DIRECTORY = PACKAGE_ROOT / "exports"
MANUAL_OVERRIDES_PATH = REGISTRY_ROOT / "manual_overrides.csv"
DOWNLOAD_CACHE_DIRECTORY = PACKAGE_ROOT / ".cache" / "huggingface"

# Recherches volontairement ciblées. Une recherche « EEG » seule renvoie de
# nombreux dépôts sans poids ou sans rapport avec l'électrophysiologie.
SEARCH_QUERIES = (
    "EEG pretrained",
    "EEGNet",
    "braindecode EEG",
    "EEG foundation model",
    "resting state EEG",
    "eyes open eyes closed EEG",
    "EEG representation learning",
    # La recherche Hugging Face fonctionne par correspondance de nom, pas par
    # sens : il faut donc connaître les noms propres des modèles pour les
    # trouver. Ajoutés après avoir constaté que "EEG foundation model" seul
    # ne remontait pas ces familles pourtant très documentées.
    "EEGPT",
    "LaBraM",
    "CBraMod",
)

# Ces dépôts sont toujours vérifiés, même s'ils ne remontent pas dans les
# premières réponses de l'API.
SEED_REPOSITORIES = (
    "PierreGtch/EEGNetv4",
    "braindecode/SignalJEPA",
    "braindecode/signal-jepa",
    "braindecode/signal-jepa_without-chans",
)

# Les limites réduisent la durée, le trafic et le risque de rate limiting.
MAX_MODELS_PER_QUERY = 15
MAX_UNIQUE_REPOSITORIES = 60
MAX_METADATA_FILE_BYTES = 2 * 1024 * 1024

# Le pipeline télécharge uniquement de petits fichiers descriptifs. Les gros
# checkpoints ne sont jamais téléchargés automatiquement.
METADATA_FILE_NAMES = {
    "config.json",
    "model_config.json",
    "preprocessor_config.json",
    "dataset_info.json",
    "kwargs.json",
    "kwargs.pkl",
    "README.md",
    "model_cfg.yaml",
}

WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
)

EXCLUDED_WEIGHT_NAME_PARTS = (
    "optimizer",
    "optim",
    "history",
    "kwargs",
    "config",
    "scheduler",
)

# Ordre réel des 32 électrodes ASC.
ASC_CHANNEL_NAMES = (
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3",
    "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
    "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
    "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
)

ASC_NUMBER_OF_CHANNELS = len(ASC_CHANNEL_NAMES)
ASC_NUMBER_OF_TIMEPOINTS = 5120

# La fréquence doit être confirmée dans les métadonnées ASC. Elle reste None
# pour empêcher le registre d'affirmer une compatibilité temporelle supposée.
ASC_SAMPLING_FREQUENCY_HZ: float | None = None

# Une variable HF_TOKEN peut être définie dans l'environnement pour augmenter
# les limites de requêtes. Le token n'est jamais écrit dans SQLite.
HUGGINGFACE_TOKEN_ENVIRONMENT_VARIABLE = "HF_TOKEN"
