from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPAIRED_ROOT = PROJECT_ROOT / "data" / "data_toy_repaired"
J7_EPOCHS_DIR = REPAIRED_ROOT / "J7" / "epochs"

THRESHOLD = 0.01


def inspect_j7_repaired() -> None:
    """
    Inspecte tous les fichiers .npy de J7 dans data_toy_repaired.

    Pour chaque fichier, on vérifie :
    - présence de NaN ;
    - présence de Inf ;
    - valeur minimale ;
    - valeur maximale ;
    - valeur absolue maximale ;
    - nombre de valeurs dont |x| dépasse THRESHOLD.
    """

    npy_files = sorted(J7_EPOCHS_DIR.glob("*.npy"))

    if not npy_files:
        raise FileNotFoundError(
            f"Aucun fichier .npy trouvé dans : {J7_EPOCHS_DIR}"
        )

    print(f"Dossier inspecté : {J7_EPOCHS_DIR}")
    print(f"Nombre de fichiers : {len(npy_files)}")
    print(f"Seuil utilisé : |x| > {THRESHOLD}")
    print()

    problematic_files = []

    for npy_path in npy_files:
        data = np.load(npy_path)

        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()

        minimum = np.min(data)
        maximum = np.max(data)
        max_absolute_value = np.max(np.abs(data))

        extreme_mask = np.abs(data) > THRESHOLD
        extreme_count = extreme_mask.sum()

        if (
            nan_count > 0
            or inf_count > 0
            or extreme_count > 0
        ):
            problematic_files.append(
                {
                    "path": npy_path,
                    "shape": data.shape,
                    "minimum": minimum,
                    "maximum": maximum,
                    "max_absolute_value": max_absolute_value,
                    "nan_count": int(nan_count),
                    "inf_count": int(inf_count),
                    "extreme_count": int(extreme_count),
                }
            )

    if not problematic_files:
        print("Aucun problème détecté dans J7 repaired.")
        return

    print(
        f"{len(problematic_files)} fichier(s) potentiellement problématique(s) :"
    )
    print()

    for result in problematic_files:
        print("=" * 80)
        print(f"Fichier : {result['path'].name}")
        print(f"Shape : {result['shape']}")
        print(f"Minimum : {result['minimum']:.8e}")
        print(f"Maximum : {result['maximum']:.8e}")
        print(
            "Valeur absolue maximale : "
            f"{result['max_absolute_value']:.8e}"
        )
        print(f"NaN : {result['nan_count']}")
        print(f"Inf : {result['inf_count']}")
        print(
            f"Nombre de valeurs avec |x| > {THRESHOLD} : "
            f"{result['extreme_count']}"
        )


if __name__ == "__main__":
    inspect_j7_repaired()

from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

original_path = (
    PROJECT_ROOT
    / "data"
    / "data_toy"
    / "J7"
    / "epochs"
    / "J7_E018_C5_YF_S1_5A15.npy"
)

repaired_path = (
    PROJECT_ROOT
    / "data"
    / "data_toy_repaired"
    / "J7"
    / "epochs"
    / "J7_E018_C5_YF_S1_5A15.npy"
)

original = np.load(original_path)
repaired = np.load(repaired_path)

print("Même shape :", original.shape == repaired.shape)
print("Fichiers identiques :", np.array_equal(original, repaired))
print("Différence absolue maximale :", np.max(np.abs(original - repaired)))

print()
print("Canal original, participant 0 canal 5")
print("min :", original[0, 5].min())
print("max :", original[0, 5].max())
print("std :", original[0, 5].std())

print()
print("Canal repaired, participant 0 canal 5")
print("min :", repaired[0, 5].min())
print("max :", repaired[0, 5].max())
print("std :", repaired[0, 5].std())

data = np.load(repaired_path)

for ch in range(32):
    print(ch, data[0, ch].std())