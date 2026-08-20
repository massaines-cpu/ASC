import torch
from huggingface_hub import hf_hub_download

# ============================================================
# 1. Télécharger le checkpoint
# ============================================================

path = hf_hub_download(
    repo_id="PierreGtch/EEGNetv4",
    filename="EEGNetv4_Cho2017/model-params.pkl",
)

print("Checkpoint :", path)


# ============================================================
# 2. Charger avec PyTorch
# ============================================================

params = torch.load(
    path,
    map_location="cpu",
    weights_only=True,
)

print("\nType chargé :", type(params))


# ============================================================
# 3. Vérifier si c'est directement un state_dict
# ============================================================

if isinstance(params, dict) and "state_dict" in params:
    state_dict = params["state_dict"]
else:
    state_dict = params


# ============================================================
# 4. Afficher toutes les couches et leurs dimensions
# ============================================================

print("\n================ STATE DICT ================\n")

for nom, tenseur in state_dict.items():

    if isinstance(tenseur, torch.Tensor):
        print(
            f"{nom:65s} -> {list(tenseur.shape)}"
        )

    else:
        print(
            f"{nom:65s} -> {type(tenseur)}"
        )


# ============================================================
# 5. Chercher les couches intéressantes
# ============================================================

print("\n================ COUCHES SPATIALES ================\n")

for nom, tenseur in state_dict.items():

    if (
        "spatial" in nom.lower()
        or "depthwise" in nom.lower()
    ):
        print(
            nom,
            "->",
            list(tenseur.shape)
            if isinstance(tenseur, torch.Tensor)
            else type(tenseur)
        )


print("\n================ CLASSIFIEUR ================\n")

for nom, tenseur in state_dict.items():

    if (
        "classifier" in nom.lower()
        or "final" in nom.lower()
        or "fc" in nom.lower()
    ):
        print(
            nom,
            "->",
            list(tenseur.shape)
            if isinstance(tenseur, torch.Tensor)
            else type(tenseur)
        )