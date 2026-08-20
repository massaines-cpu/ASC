"""Vérifie empiriquement si les données ASC sont déjà filtrées 0,5-40 Hz.

Contexte
--------
``prepare_signal_jepa_prelocal.py`` documente ce point comme non tranché :
le checkpoint SignalJEPA a été pré-entraîné sur des données filtrées entre
0,5 et 40 Hz, et personne ne sait avec certitude si l'acquisition ASC
applique déjà ce filtrage.

"Acquisition" désigne ici la chaîne matérielle qui enregistre le signal EEG
brut (électrodes → amplificateur → conversion numérique), avant tout
traitement logiciel. Les amplificateurs EEG appliquent souvent eux-mêmes des
filtres analogiques (passe-haut anti-dérive, passe-bas anti-repliement)
pendant cette étape, documentés en principe dans la configuration de
l'amplificateur ou le protocole d'acquisition du laboratoire — documentation
à laquelle ce script n'a pas accès. À défaut, la seule façon fiable de savoir
est de mesurer directement le contenu fréquentiel du signal enregistré.

Méthode
-------
Sur un échantillon de fichiers ``data_final``, on calcule la densité
spectrale de puissance (PSD, méthode de Welch) par canal, moyennée sur les
canaux et les fichiers. Trois critères, chacun vérifiable sans ambiguïté :

1. **Bruit secteur à 50 Hz (France)** : un filtre passe-bas à 40 Hz
   supprimerait entièrement un pic à 50 Hz. Sa présence est donc une preuve
   directe qu'aucun filtrage jusqu'à 40 Hz n'a été appliqué. C'est le
   critère le plus fiable des trois : il ne dépend d'aucun seuil arbitraire.
2. **Dérive basse fréquence** : proportion de la puissance totale située
   sous 0,5 Hz. Un signal filtré passe-haut ne devrait presque plus en
   contenir.
3. **Contenu au-delà de 40 Hz** : proportion de la puissance totale située
   au-dessus de 40 Hz (hors bande du pic secteur, compté séparément). Une
   puissance non négligeable qui se prolonge vers les hautes fréquences
   indique l'absence de filtre passe-bas.

Le script ne modifie aucune donnée : il ne fait que produire un verdict et
une figure pour trancher ``APPLY_BANDPASS`` dans
``prepare_signal_jepa_prelocal.py`` et ``prepare_signal_jepa.py`` en
connaissance de cause plutôt qu'au hasard.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET_NAME = "data_final"
SAMPLING_FREQUENCY = 512.0

# Un sous-ensemble suffit pour une estimation spectrale stable et reste
# rapide à charger. None pour utiliser tous les fichiers disponibles.
MAXIMUM_FILES_TO_INSPECT = 60

LOW_FREQUENCY_CUTOFF_HZ = 0.5
HIGH_FREQUENCY_CUTOFF_HZ = 40.0
MAINS_FREQUENCY_HZ = 50.0  # Fréquence secteur en France.
MAINS_BAND_HALF_WIDTH_HZ = 1.0

# Un pic secteur au moins deux fois plus puissant que ses voisins immédiats
# est considéré comme présent, sans dépendre d'un seuil de puissance absolu.
MAINS_PEAK_PROMINENCE_RATIO = 2.0

# En dessous de ce pourcentage de la puissance totale, la dérive basse
# fréquence ou le contenu haute fréquence sont jugés négligeables. Seuil
# indicatif, à lire avec le verdict du pic secteur qui est lui sans seuil.
NEGLIGIBLE_POWER_FRACTION_PERCENT = 1.0

OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "results" / "signal_jepa_preprocessing" / "bandpass_diagnostic"
)


def list_source_files(source_root: Path) -> list[Path]:
    """Sélectionne un échantillon reproductible de fichiers à inspecter."""

    all_files = sorted(source_root.rglob("*.npy"))
    if not all_files:
        raise FileNotFoundError(f"Aucun fichier .npy trouvé dans {source_root}.")

    if MAXIMUM_FILES_TO_INSPECT is None:
        return all_files

    # Un pas régulier couvre les huit dyades au lieu de ne lire que les
    # premiers fichiers d'une seule dyade (les fichiers sont triés par dyade).
    step = max(1, len(all_files) // MAXIMUM_FILES_TO_INSPECT)
    return all_files[::step][:MAXIMUM_FILES_TO_INSPECT]


def compute_average_psd(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Calcule la PSD de Welch moyennée sur tous les canaux et fichiers."""

    all_power_spectra = []
    frequencies = None

    for file_path in files:
        eeg = np.load(file_path)  # [2 participants, 32 canaux, 5120 points]
        for participant_index in range(eeg.shape[0]):
            channel_frequencies, channel_power = welch(
                eeg[participant_index],
                fs=SAMPLING_FREQUENCY,
                nperseg=min(1024, eeg.shape[-1]),
                axis=-1,
            )
            if frequencies is None:
                frequencies = channel_frequencies
            all_power_spectra.append(channel_power.mean(axis=0))

    average_power = np.mean(all_power_spectra, axis=0)
    return frequencies, average_power


def band_power_fraction(
    frequencies: np.ndarray,
    power: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    """Retourne la part de puissance totale contenue dans [low_hz, high_hz]."""

    total_power = np.trapezoid(power, frequencies)
    band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    band_power = np.trapezoid(power[band_mask], frequencies[band_mask])
    return 100.0 * band_power / total_power


def detect_mains_peak(
    frequencies: np.ndarray,
    power: np.ndarray,
) -> tuple[bool, float]:
    """Compare la puissance à 50 Hz à celle de son voisinage immédiat."""

    mains_band = (
        (frequencies >= MAINS_FREQUENCY_HZ - MAINS_BAND_HALF_WIDTH_HZ)
        & (frequencies <= MAINS_FREQUENCY_HZ + MAINS_BAND_HALF_WIDTH_HZ)
    )
    surrounding_band = (
        (frequencies >= MAINS_FREQUENCY_HZ - 5.0)
        & (frequencies <= MAINS_FREQUENCY_HZ + 5.0)
        & ~mains_band
    )

    if not mains_band.any() or not surrounding_band.any():
        return False, 0.0

    peak_power = power[mains_band].max()
    surrounding_median_power = np.median(power[surrounding_band])
    ratio = peak_power / surrounding_median_power if surrounding_median_power > 0 else 0.0

    return ratio >= MAINS_PEAK_PROMINENCE_RATIO, ratio


def save_psd_plot(frequencies: np.ndarray, power: np.ndarray) -> Path:
    """Enregistre la PSD avec les bandes d'intérêt annotées."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIRECTORY / "power_spectral_density.png"

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.semilogy(frequencies, power)
    axis.axvspan(0, LOW_FREQUENCY_CUTOFF_HZ, color="orange", alpha=0.15, label="< 0,5 Hz")
    axis.axvspan(
        HIGH_FREQUENCY_CUTOFF_HZ,
        frequencies.max(),
        color="red",
        alpha=0.1,
        label="> 40 Hz",
    )
    axis.axvline(MAINS_FREQUENCY_HZ, color="black", linestyle="--", linewidth=1, label="50 Hz secteur")
    axis.set_xlabel("Fréquence (Hz)")
    axis.set_ylabel("Puissance (échelle log)")
    axis.set_title("PSD moyenne — data_final (échantillon)")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    return plot_path


def main() -> None:
    source_root = PROJECT_ROOT / "data" / SOURCE_DATASET_NAME
    files = list_source_files(source_root)
    print(f"{len(files)} fichiers inspectés (sur {source_root}).")

    frequencies, power = compute_average_psd(files)

    low_fraction = band_power_fraction(frequencies, power, 0.0, LOW_FREQUENCY_CUTOFF_HZ)
    high_fraction = band_power_fraction(
        frequencies, power, HIGH_FREQUENCY_CUTOFF_HZ, frequencies.max()
    )
    mains_present, mains_ratio = detect_mains_peak(frequencies, power)
    plot_path = save_psd_plot(frequencies, power)

    print("\n" + "=" * 72)
    print("DIAGNOSTIC — FILTRAGE 0,5-40 Hz DES DONNÉES ASC")
    print("=" * 72)
    print(f"Puissance sous {LOW_FREQUENCY_CUTOFF_HZ} Hz      : {low_fraction:.2f} % du total")
    print(f"Puissance au-dessus de {HIGH_FREQUENCY_CUTOFF_HZ} Hz : {high_fraction:.2f} % du total")
    print(
        f"Pic secteur à {MAINS_FREQUENCY_HZ:.0f} Hz         : "
        f"{'PRÉSENT' if mains_present else 'absent'} "
        f"(ratio vs voisinage = {mains_ratio:.2f})"
    )
    print(f"Figure enregistrée              : {plot_path}")

    print("\n--- Verdict ---")
    if mains_present:
        print(
            "Un pic secteur à 50 Hz est détecté : les données NE sont PAS "
            "filtrées jusqu'à 40 Hz (un passe-bas à 40 Hz l'aurait supprimé "
            "entièrement). Régler APPLY_BANDPASS = True dans "
            "prepare_signal_jepa_prelocal.py et prepare_signal_jepa.py."
        )
    elif (
        low_fraction < NEGLIGIBLE_POWER_FRACTION_PERCENT
        and high_fraction < NEGLIGIBLE_POWER_FRACTION_PERCENT
    ):
        print(
            "Aucun pic secteur, et la puissance hors bande [0,5-40] Hz est "
            f"négligeable (< {NEGLIGIBLE_POWER_FRACTION_PERCENT} % chacune). "
            "Cohérent avec des données déjà filtrées 0,5-40 Hz. "
            "APPLY_BANDPASS = False reste défendable, mais vérifier si possible "
            "la configuration réelle de l'amplificateur avant de conclure "
            "définitivement."
        )
    else:
        print(
            "Résultat ambigu : pas de pic secteur net, mais une part non "
            f"négligeable de la puissance reste hors de [0,5-40] Hz "
            f"({low_fraction:.2f} % < 0,5 Hz, {high_fraction:.2f} % > 40 Hz). "
            "Inspecter la figure avant de décider. Par prudence, un "
            "APPLY_BANDPASS = True reste sans risque scientifique : filtrer un "
            "signal déjà filtré dans la même bande ne change rien d'important, "
            "alors que ne pas filtrer un signal non filtré fausse le transfert "
            "depuis le checkpoint SignalJEPA."
        )


if __name__ == "__main__":
    main()
