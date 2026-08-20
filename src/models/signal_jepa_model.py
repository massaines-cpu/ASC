"""SignalJEPA Contextual pour la classification binaire ASC YO/YF, 32 canaux.

Trois expériences sont possibles avec exactement la même architecture :

1. ``scratch`` + ``full_finetuning`` : tous les poids sont aléatoires ;
2. ``pretrained`` + ``classifier_only`` : l'encodeur SSL pré-entraîné
   (``feature_encoder`` + ``pos_encoder`` + ``transformer``) est gelé,
   seul ``final_layer`` apprend ;
3. ``pretrained`` + ``full_finetuning`` : tous les blocs s'adaptent à ASC.

Pourquoi ``SignalJEPA_Contextual`` et pas la classe ``SignalJEPA`` de base
--------------------------------------------------------------------------
``SignalJEPA`` seule n'est pas une architecture de classification : sa
documentation officielle le dit explicitement ("This model is not meant for
classification but for SSL pre-training") et sa sortie brute est un tenseur
d'embeddings par patch ``[batch, n_chans * n_patches, emb_dim]``, sans tête de
décision. Trois variantes existent pour la classification :
``SignalJEPA_Contextual``, ``SignalJEPA_PostLocal``, ``SignalJEPA_PreLocal``.

PreLocal et PostLocal ne réutilisent QUE les poids du ``feature_encoder`` :
ni le transformer, ni la table d'embeddings de canaux ne sont chargés
(cf. leur documentation : "no channel embedding nor transformer"). C'est
volontairement le choix fait pour l'expérience à 19 canaux
(``signal_jepa_prelocal_model.py``), qui utilise un montage réduit et
``signal-jepa_without-chans``.

``SignalJEPA_Contextual`` est la seule variante qui charge aussi le
transformer et la table d'embeddings pré-entraînée à 62 canaux. C'est
précisément ce qui a fait remonter ``braindecode/signal-jepa`` en tête du
benchmark de compatibilité ASC (statut "oui", nombre de canaux variable) :
un ``from_pretrained`` avec ``chs_info`` limité à un sous-ensemble du
montage pré-entraîné réutilise directement les embeddings appris pour ces
canaux, au lieu de les réinitialiser. Vérifié programmatiquement : les 32
noms d'électrodes ASC (voir ``ASC_CHANNEL_NAMES`` ci-dessous) sont tous
présents dans les 62 canaux de pré-entraînement de ``braindecode/signal-jepa``
— aucune électrode ASC n'est absente de la table.

Historique : un premier ``signal_jepa_model.py`` (disparu du dépôt) semble
avoir enveloppé la classe ``SignalJEPA`` de base plutôt que Contextual. Sur
le fold J1, les deux variantes (scratch et pretrained/classifier_only)
convergeaient vers une sortie constante (accuracy de validation exactement
à 0.5, toutes les prédictions dans une seule classe) : c'est la signature
d'un gradient qui ne transporte aucune information utile depuis l'entrée,
cohérent avec une tête de classification improvisée sur un tenseur
d'embeddings par patch non prévu pour ça, plutôt qu'un problème de données
ou d'hyperparamètres.

La sortie contient un seul logit associé à YF. L'entraînement utilise donc
``BCEWithLogitsLoss`` et la probabilité est calculée avec Sigmoid.
"""

import json
from typing import Literal

import torch
from braindecode.models import SignalJEPA_Contextual
from huggingface_hub import hf_hub_download
from torch import nn


ModelVariant = Literal["scratch", "pretrained"]
FreezeStrategy = Literal["full_finetuning", "classifier_only"]

DEFAULT_CHECKPOINT = "braindecode/signal-jepa"
NUMBER_OF_OUTPUTS = 1

# Ordre réel des 32 canaux ASC dans la deuxième dimension des fichiers
# préparés. Identique à celui utilisé dans prepare_signal_jepa_prelocal.py
# et src/eeg_model_registry/config.py : ne jamais trier alphabétiquement.
ASC_CHANNEL_NAMES = [
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3",
    "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
    "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
    "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
]


def _load_asc_channel_positions(checkpoint_name: str = DEFAULT_CHECKPOINT) -> list[dict]:
    """Associe chaque canal ASC à sa position 3D réelle du montage pré-entraîné.

    SignalJEPA_Contextual indexe sa table d'embeddings par nom de canal : il
    faut donc les mêmes noms ET les mêmes coordonnées que la table
    d'origine plutôt que des positions ASC recalculées séparément, sans quoi
    ``channel_embedding='pretrain_aligned'`` associerait un canal à la
    mauvaise position. La position vient toujours de ``braindecode/signal-jepa``
    (DEFAULT_CHECKPOINT), même en mode scratch : c'est un fait physique du
    montage 10-10, indépendant du chargement ou non des poids pré-entraînés.
    """

    config_path = hf_hub_download(checkpoint_name, "config.json")
    with open(config_path, encoding="utf-8") as config_file:
        pretrained_channels = {
            entry["ch_name"]: entry["loc"]
            for entry in json.load(config_file)["chs_info"]
        }

    missing_channels = [
        name for name in ASC_CHANNEL_NAMES if name not in pretrained_channels
    ]
    if missing_channels:
        raise ValueError(
            f"Canaux ASC absents de la table pré-entraînée {checkpoint_name} : "
            f"{missing_channels}. SignalJEPA_Contextual ne peut pas leur "
            "associer un embedding pré-entraîné."
        )

    return [
        {"ch_name": name, "loc": pretrained_channels[name]}
        for name in ASC_CHANNEL_NAMES
    ]


def configure_freezing(
    model: SignalJEPA_Contextual,
    freeze_strategy: FreezeStrategy,
) -> None:
    """Définit précisément les paramètres mis à jour par l'optimiseur."""

    if freeze_strategy == "full_finetuning":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    if freeze_strategy == "classifier_only":
        # Tout est d'abord gelé, puis les couches nouvelles sont réactivées.
        for parameter in model.parameters():
            parameter.requires_grad = False

        # Vérifié empiriquement en comparant les clés du modèle à celles du
        # checkpoint SSL téléchargé : feature_encoder, pos_encoder et
        # transformer sont intégralement fournis par le pré-entraînement.
        # Seul final_layer (spat_conv + linear) est absent du checkpoint et
        # donc initialisé aléatoirement : c'est la seule partie spécifique à
        # la tâche ASC.
        for parameter in model.final_layer.parameters():
            parameter.requires_grad = True
        return

    raise ValueError(f"Stratégie de gel inconnue : {freeze_strategy}.")


def create_signal_jepa_model(
    pretrained: bool,
    number_of_timepoints: int,
    sampling_frequency: float,
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    freeze_strategy: FreezeStrategy = "full_finetuning",
) -> nn.Module:
    """Crée un SignalJEPA Contextual neuf pour chaque fold LODO."""

    chs_info = _load_asc_channel_positions()

    common_arguments = {
        "chs_info": chs_info,
        "n_times": number_of_timepoints,
        "n_outputs": NUMBER_OF_OUTPUTS,
    }

    if pretrained:
        model = SignalJEPA_Contextual.from_pretrained(
            checkpoint_name,
            **common_arguments,
            # final_layer n'existe pas dans le checkpoint SSL : c'est attendu,
            # cf. configure_freezing ci-dessus.
            strict=False,
        )
        model_variant: ModelVariant = "pretrained"
    else:
        model = SignalJEPA_Contextual(
            **common_arguments,
            sfreq=sampling_frequency,
            # Valeur identique au tutoriel officiel et à signal_jepa_prelocal_model.py.
            drop_prob=0.0,
        )
        model_variant = "scratch"

    configure_freezing(model, freeze_strategy)

    # "signal_jepa" (pas "signal_jepa_contextual") : epoch_runs.py teste
    # cette chaîne exacte pour savoir s'il doit repasser feature_encoder /
    # pos_encoder / transformer en mode eval() pendant que final_layer reste
    # en train() — un détail nécessaire sur MPS (cf. epoch_runs.py) pour que
    # le gel de classifier_only soit réellement respecté.
    model.asc_model_family = "signal_jepa"
    model.asc_model_variant = model_variant
    model.asc_freeze_strategy = freeze_strategy

    return model


def prepare_binary_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalise la sortie Contextual de ``[batch, 1]`` vers ``[batch]``."""

    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits.squeeze(dim=1)

    raise ValueError(
        "SignalJEPA Contextual doit produire un logit YF par exemple, "
        f"mais a produit {tuple(logits.shape)}."
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Retourne le nombre total puis le nombre de paramètres entraînables."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def print_trainable_parameters(model: nn.Module) -> None:
    """Affiche les poids réellement transmis à l'optimiseur."""

    print("Paramètres entraînables :")
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(f"  {parameter_name:<55} {tuple(parameter.shape)}")
