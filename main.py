from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "data" / "data_toy"

eeg_files = sorted(
    (DATASET_ROOT / "J1" / "epochs").glob("*.npy")
)

eeg_path = eeg_files[0]

print("Fichier choisi :", eeg_path.name)

data = np.load(eeg_path, mmap_mode="r")

print("\nShape complète :", data.shape)
print("Type :", data.dtype)

print("\nParticipant 1 :", data[0].shape)
print("Participant 2 :", data[1].shape)

print("\nCanal 0 du participant 1 :", data[0, 0].shape)

print("\n10 premières valeurs du canal 0 de P1 :")
print(data[0, 0, :10])

print("\nQuelques statistiques :")
print("Minimum :", data.min())
print("Maximum :", data.max())
print("Moyenne :", data.mean())
print("Std :", data.std())