"""Vérifie la compatibilité de SignalJEPA avec le montage EEG ASC.

Ce script est volontairement limité à un test de construction et de
dimensions. Il ne lance ni entraînement, ni validation LODO, et ne modifie
aucun fichier EEG.

Hypothèse testée
----------------
Les 32 électrodes ASC sont un sous-ensemble des 62 électrodes utilisées
pendant le pré-entraînement de SignalJEPA. Le checkpoint complet peut donc
réutiliser ses embeddings spatiaux pré-entraînés en sélectionnant uniquement
les lignes correspondant aux électrodes ASC.

Après un futur rééchantillonnage de 512 Hz vers 128 Hz, une fenêtre ASC de
10 secondes contiendra 1280 points temporels.
"""

import mne
import torch
from braindecode.models import SignalJEPA_Contextual


ASC_CHANNEL_NAMES = [
    "Fp1",
    "AF3",
    "F7",
    "F3",
    "FC1",
    "FC5",
    "T7",
    "C3",
    "CP1",
    "CP5",
    "P7",
    "P3",
    "Pz",
    "PO3",
    "O1",
    "Oz",
    "O2",
    "PO4",
    "P4",
    "P8",
    "CP6",
    "CP2",
    "C4",
    "T8",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "AF4",
    "Fp2",
    "Fz",
    "Cz",
]

CHECKPOINT_NAME = "braindecode/signal-jepa"
TARGET_SAMPLING_FREQUENCY = 128.0
WINDOW_DURATION_SECONDS = 10
NUMBER_OF_TIMEPOINTS = int(
    TARGET_SAMPLING_FREQUENCY * WINDOW_DURATION_SECONDS
)
NUMBER_OF_OUTPUTS = 1
FAKE_BATCH_SIZE = 5
RANDOM_SEED = 42


def create_asc_channels_info() -> list[dict]:
    """Construit les noms et positions 3D des 32 électrodes ASC.

    Les fichiers ``.npy`` ne contiennent pas les coordonnées spatiales.
    Elles sont donc obtenues à partir du montage standard 10-05 de MNE.
    L'ordre des canaux reste strictement identique à celui du dataset ASC.
    """

    info = mne.create_info(
        ch_names=ASC_CHANNEL_NAMES,
        sfreq=TARGET_SAMPLING_FREQUENCY,
        ch_types="eeg",
    )

    standard_montage = mne.channels.make_standard_montage(
        "standard_1005"
    )

    # on_missing="raise" garantit que le test s'arrête immédiatement
    # si un nom d'électrode ASC n'existe pas dans le montage standard.
    info.set_montage(
        standard_montage,
        match_case=False,
        on_missing="raise",
    )

    return info["chs"]


def create_pretrained_model(
    channels_info: list[dict],
) -> SignalJEPA_Contextual:
    """Charge l'encodeur pré-entraîné et crée une tête binaire neuve."""

    model = SignalJEPA_Contextual.from_pretrained(
        CHECKPOINT_NAME,
        chs_info=channels_info,
        n_times=NUMBER_OF_TIMEPOINTS,
        sfreq=TARGET_SAMPLING_FREQUENCY,
        n_outputs=NUMBER_OF_OUTPUTS,
        channel_embedding="pretrain_aligned",
        # La tête de classification YO/YF n'existe pas dans le checkpoint
        # auto-supervisé. Elle est donc initialisée aléatoirement.
        strict=False,
    )

    return model


def main() -> None:
    """Charge SignalJEPA et vérifie les dimensions d'un forward."""

    torch.manual_seed(RANDOM_SEED)

    channels_info = create_asc_channels_info()
    model = create_pretrained_model(channels_info)
    model.eval()

    # Les valeurs artificielles sont exprimées approximativement en µV,
    # uniquement pour rester proches de l'échelle du pré-entraînement.
    # Elles ne servent pas à mesurer les performances du modèle.
    fake_eeg_microvolts = (
        torch.randn(
            FAKE_BATCH_SIZE,
            len(ASC_CHANNEL_NAMES),
            NUMBER_OF_TIMEPOINTS,
        )
        * 20.0
    )

    with torch.no_grad():
        logits = model(fake_eeg_microvolts)

    expected_shape = (FAKE_BATCH_SIZE, NUMBER_OF_OUTPUTS)

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            "SignalJEPA a produit une forme inattendue : "
            f"{tuple(logits.shape)} au lieu de {expected_shape}."
        )

    # Le pipeline ASC accepte un logit par exemple. SignalJEPA retourne
    # [batch, 1] ; squeeze(dim=1) produit donc [batch] sans supprimer
    # accidentellement la dimension batch lorsque sa taille vaut 1.
    binary_logits = logits.squeeze(dim=1)
    probability_yf = torch.sigmoid(binary_logits)
    predictions = (probability_yf >= 0.5).long()

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    number_of_trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Checkpoint :", CHECKPOINT_NAME)
    print("Nombre d'électrodes :", len(channels_info))
    print("Fréquence cible :", TARGET_SAMPLING_FREQUENCY, "Hz")
    print("Nombre de points :", NUMBER_OF_TIMEPOINTS)
    print("Forme entrée :", tuple(fake_eeg_microvolts.shape))
    print("Forme sortie :", tuple(logits.shape))
    print("Probabilités YF :", probability_yf)
    print("Prédictions :", predictions)
    print("Paramètres totaux :", f"{number_of_parameters:,}")
    print(
        "Paramètres actuellement entraînables :",
        f"{number_of_trainable_parameters:,}",
    )
    print("Test de compatibilité réussi.")


if __name__ == "__main__":
    main()
