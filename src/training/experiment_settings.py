"""Paramètres de l'expérience ASC à modifier avant un lancement.

Ce fichier contient uniquement les choix expérimentaux. Le code de la LODO
reste dans ``run_lodo.py`` et ne doit pas être modifié pour changer de modèle,
de learning rate ou de fold.

Pour respecter le protocole scientifique, modifier une seule variable entre
deux expériences comparées.
"""


# =====================================================================
# 1. DONNÉES ET MODÈLE
# =====================================================================

# Modèles possibles :
# linear, non_linear, small_cnn, eegnet,
# signal_jepa_scratch, signal_jepa_pretrained.
MODEL_NAME = "signal_jepa_pretrained"

# Dataset préparé à 128 Hz et exprimé en microvolts.
DATASET_VERSION = "data_signal_jepa_128hz_uv"

# Une liste limite le test aux folds indiqués.
# Mettre None uniquement lorsque les huit folds doivent être lancés.
SELECTED_FOLDS = ["J1"]


# =====================================================================
# 2. PRÉTRAITEMENT ET DIMENSIONS
# =====================================================================

# SignalJEPA attend ici les microvolts sans Z-score.
STANDARDIZE = False
NUMBER_OF_CHANNELS = 32
NUMBER_OF_TIMEPOINTS = 1280
SAMPLING_FREQUENCY = 128.0


# =====================================================================
# 3. ENTRAÎNEMENT
# =====================================================================

BATCH_SIZE = 2
NUMBER_OF_EPOCHS = 20
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_MIN_DELTA = 1e-4
RANDOM_SEED = 42

# Ces deux valeurs servent uniquement au MLP non linéaire.
HIDDEN_LAYER_SIZE = 32
DROPOUT_RATE = 0.0


# =====================================================================
# 4. TRANSFERT D'APPRENTISSAGE SIGNALJEPA
# =====================================================================

PRETRAINED_CHECKPOINT = "braindecode/signal-jepa"

# full_finetuning : encodeur et tête entraînés.
# classifier_only : encodeur gelé, seule la tête YO/YF apprend.
FREEZE_STRATEGY = "classifier_only"


# =====================================================================
# 5. EXÉCUTION ET SUIVI
# =====================================================================

# Valeurs possibles : auto, cpu, mps, cuda.
DEVICE_NAME = "mps"

# False permet de travailler sans serveur MLflow.
ENABLE_MLFLOW = False
