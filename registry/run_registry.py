"""Lance le catalogue depuis PyCharm avec un simple bouton Run."""

from pathlib import Path
import os
import sys


# Le dossier src est ajouté uniquement pour ce projet autonome. Il n'est pas
# nécessaire d'installer le package dans l'environnement Python.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRECTORY = PROJECT_ROOT / "src"
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface_home"))
os.environ.setdefault("HF_XET_CACHE", str(PROJECT_ROOT / ".cache" / "xet"))
sys.path.insert(0, str(SOURCE_DIRECTORY))

from eeg_model_registry.pipeline import run_registry_pipeline  # noqa: E402


if __name__ == "__main__":
    run_registry_pipeline()
