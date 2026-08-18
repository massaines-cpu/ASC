import torch
from braindecode.models import (
    SignalJEPA,
    SignalJEPA_Contextual,
)

from main import (
    create_asc_channels_info,
)


channels_info = create_asc_channels_info()


# ==========================================================
# 1. SignalJEPA original : extraction de représentations
# ==========================================================

base_model = SignalJEPA.from_pretrained(
    "braindecode/signal-jepa",
    chs_info=channels_info,
    channel_embedding="pretrain_aligned",
)

print("\nSIGNALJEPA ORIGINAL")
print(base_model)
print("\nCouche finale originale :")
print(base_model.final_layer)


# ==========================================================
# 2. SignalJEPA adapté à la classification YO/YF
# ==========================================================

classification_model = SignalJEPA_Contextual.from_pretrained(
    "braindecode/signal-jepa",
    chs_info=channels_info,
    n_times=1280,
    sfreq=128.0,
    n_outputs=1,
    channel_embedding="pretrain_aligned",
    strict=False,
)

print("\nSIGNALJEPA CONTEXTUAL")
print(classification_model)
print("\nNouvelle tête YO/YF :")
print(classification_model.final_layer)


# ==========================================================
# 3. Comparaison des sorties
# ==========================================================

fake_eeg = torch.randn(1, 32, 1280) * 20.0

base_model.eval()
classification_model.eval()

with torch.no_grad():
    original_features = base_model(fake_eeg)

    transferred_features = classification_model(
        fake_eeg,
        return_features=True,
    )["features"]

    classification_logits = classification_model(
        fake_eeg
    )

print("\nReprésentations du modèle original :")
print(original_features.shape)

print("\nReprésentations conservées dans le modèle Contextual :")
print(transferred_features.shape)

print("\nSortie après la nouvelle tête YO/YF :")
print(classification_logits.shape)

