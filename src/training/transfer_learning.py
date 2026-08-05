"""
Transfer learning avec EEGNet pré-entraîné (braindecode, checkpoint Lee2019 ERP).

CE QUI EST RÉELLEMENT TRANSFÉRABLE:

Le checkpoint pré-entraîné a été entraîné avec 19 électrodes et 128 points
temporels (n_chans=19, n_times=128). Notre tâche utilise 32 électrodes et
5120 points temporels. En comparant les state_dict des deux configurations
(vérifié empiriquement) :

    - conv_temporal, bnorm_temporal          : COMPATIBLES (indépendants
                                                du nombre d'électrodes et
                                                de points temporels)
    - conv_separable_depth/point, bnorm_2    : COMPATIBLES
    - conv_spatial                           : INCOMPATIBLE (dépend du
                                                nombre d'électrodes :
                                                19 vs 32) -> reste
                                                initialisé aléatoirement,
                                                jamais transféré
    - final_layer.conv_classifier            : INCOMPATIBLE (dépend du
                                                nombre de points temporels
                                                restants après pooling :
                                                4 vs 160) -> remplacé
                                                explicitement

CONSÉQUENCE SUR LES NIVEAUX DE FREEZE
---------------------------------------
Comme conv_spatial n'est JAMAIS pré-entraîné dans notre cas (il reste
toujours aléatoire), le geler tôt revient à figer une couche qui n'a
encore rien appris. On garde les 4 niveaux de freeze demandés pour
observer cet effet empiriquement, mais il faut s'attendre à ce que
geler le bloc spatial nuise aux performances, précisément parce qu'il
n'a jamais bénéficié du transfert.
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from braindecode.models import EEGNet
from huggingface_hub import hf_hub_download
from sklearn.metrics import confusion_matrix, classification_report

from src.dataset.labels import prepare_classification_table
from src.dataset.dataloader_participant import create_participant_dataloaders


# ==================================================================
# 1. CONFIGURATION GÉNÉRALE
# ==================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEVELOPMENT_DYADS = ["J1", "J2", "J4", "J5", "J7", "J8", "J10", 'J15']
TEST_DYADS = []

BATCH_SIZE = 5
NUMBER_OF_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 30
EARLY_STOPPING_MIN_DELTA = 1e-4
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

N_CHANNELS = 32
N_TIMES = 5120
N_CLASSES = 2

REPOSITORY_ID = "guido151/EEGNetv4"
CHECKPOINT_DIRECTORY = "EEGNetv4_Lee2019_ERP"

FREEZE_LEVELS = {
    0: "fine_tuning_complet",
    1: "freeze_temporal",
    2: "freeze_temporal_spatial",  # attendu : sous-optimal, cf. docstring
    3: "freeze_tout_sauf_classifieur",
}

EXPERIMENT_NAME = "transfer_learning_A_eegnet_braindecode"
RESULTS_DIR = PROJECT_ROOT / "results" / EXPERIMENT_NAME
MODELS_DIR = PROJECT_ROOT / "models" / EXPERIMENT_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_PATH = (PROJECT_ROOT/ "data"/ "data_toy_repaired")

criterion = nn.CrossEntropyLoss()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


metadata = pd.read_csv(PROJECT_ROOT / "data" / "all_metadata.csv")
classification_table = prepare_classification_table(
    metadata=metadata,
    target_column="eyes_code",
    allowed_classes=["YO", "YF"],
    label_map={"YO": 0, "YF": 1},
)


# ==================================================================
# 2. TÉLÉCHARGEMENT ET CHARGEMENT DU CHECKPOINT PRÉ-ENTRAÎNÉ
# ==================================================================

def download_pretrained_state_dict() -> dict[str, torch.Tensor]:
    """Télécharge directement le state_dict du checkpoint EEGNetv4.

    Le fichier params.pt contient les poids PyTorch sauvegardés par
    l'ancienne version du modèle. On lit directement ce dictionnaire
    sans tenter de le recharger dans une instance récente de Braindecode.

    Cela évite le conflit de nom entre :
        ancienne version : conv_spatial.weight
        version actuelle : conv_spatial.parametrizations.weight.original
    """

    parameters_path = hf_hub_download(
        repo_id=REPOSITORY_ID,
        filename=f"{CHECKPOINT_DIRECTORY}/params.pt",
    )

    checkpoint = torch.load(
        parameters_path,
        map_location="cpu",
        weights_only=True,
    )

    # Certains checkpoints contiennent directement le state_dict.
    # D'autres l'enveloppent dans une clé comme "state_dict" ou
    # "model_state_dict".
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Le checkpoint téléchargé ne contient pas "
            "un state_dict PyTorch valide."
        )

    # Certains modèles sauvegardés par Skorch ajoutent éventuellement
    # le préfixe « module. » ou « module_. » aux noms des paramètres.
    cleaned_state_dict = {}

    for parameter_name, parameter_value in state_dict.items():

        cleaned_name = parameter_name

        if cleaned_name.startswith("module_."):
            cleaned_name = cleaned_name.removeprefix("module_.")

        elif cleaned_name.startswith("module."):
            cleaned_name = cleaned_name.removeprefix("module.")

        cleaned_state_dict[cleaned_name] = parameter_value

    print(
        f"Checkpoint chargé : {len(cleaned_state_dict)} "
        "tenseurs trouvés."
    )

    return cleaned_state_dict


def load_pretrained_backbone(
    model: EEGNet,
    pretrained_state_dict: dict,
) -> EEGNet:
    """Charge uniquement les paramètres compatibles du checkpoint.

    Un paramètre est transféré seulement si :
    - son nom existe dans le modèle cible ;
    - sa forme est exactement identique ;
    - il n'appartient pas au classifieur final.
    """

    target_state_dict = model.state_dict()

    compatible_weights = {}
    skipped_weights = {}

    for name, pretrained_tensor in pretrained_state_dict.items():

        # La tête finale doit rester propre à la tâche YO/YF.
        if name.startswith("final_layer"):
            skipped_weights[name] = (
                "classifieur final volontairement ignoré"
            )
            continue

        # Le paramètre n'existe pas sous ce nom dans le modèle cible.
        if name not in target_state_dict:
            skipped_weights[name] = (
                "nom absent du modèle cible"
            )
            continue

        target_tensor = target_state_dict[name]

        # Même nom, mais dimensions différentes.
        if pretrained_tensor.shape != target_tensor.shape:
            skipped_weights[name] = (
                f"forme source {tuple(pretrained_tensor.shape)} "
                f"!= forme cible {tuple(target_tensor.shape)}"
            )
            continue

        compatible_weights[name] = pretrained_tensor

    if not compatible_weights:
        raise RuntimeError(
            "Aucun poids compatible n'a été trouvé dans le checkpoint."
        )

    model.load_state_dict(
        compatible_weights,
        strict=False,
    )

    print("\nPoids transférés :")

    for name in sorted(compatible_weights):
        print(
            f"  {name:<55} "
            f"{tuple(compatible_weights[name].shape)}"
        )

    print("\nPoids ignorés :")

    for name, reason in skipped_weights.items():
        print(
            f"  {name:<55} {reason}"
        )

    print(
        f"\nNombre de tenseurs transférés : "
        f"{len(compatible_weights)}"
    )

    print(
        f"Nombre de tenseurs ignorés : "
        f"{len(skipped_weights)}"
    )

    return model


# ==================================================================
# 3. GEL / DÉGEL DES BLOCS
# ==================================================================

def set_freeze_level(model: EEGNet, freeze_level: int) -> EEGNet:
    """Gèle les premiers blocs d'EEGNet selon le niveau demandé.

    Regroupement des couches en blocs logiques (noms des sous-modules
    vérifiés directement sur l'implémentation braindecode) :

        bloc temporel  : conv_temporal, bnorm_temporal
        bloc spatial   : conv_spatial, bnorm_1        (jamais pré-entraîné
                                                        dans notre cas)
        bloc séparable : conv_separable_depth, conv_separable_point, bnorm_2

    Le classifieur (final_layer) n'est jamais inclus dans les blocs
    gelables : il est toujours entraînable, puisqu'il vient d'être
    réinitialisé pour notre tâche.
    """
    temporal_block = [model.conv_temporal, model.bnorm_temporal]
    spatial_block = [model.conv_spatial, model.bnorm_1]
    separable_block = [
        model.conv_separable_depth,
        model.conv_separable_point,
        model.bnorm_2,
    ]

    blocks_in_order = [temporal_block, spatial_block, separable_block]

    for parameter in model.parameters():
        parameter.requires_grad = True

    for block in blocks_in_order[:freeze_level]:
        for module in block:
            for parameter in module.parameters():
                parameter.requires_grad = False

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


# ==================================================================
# 4. EXÉCUTION D'UNE EPOCH
# ==================================================================

def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    gradient_context = torch.enable_grad() if training else torch.no_grad()

    with gradient_context:
        for eeg, labels in loader:
            if training:
                optimizer.zero_grad()

            predictions = model(eeg)
            loss = criterion(predictions, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predicted_classes = predictions.argmax(dim=1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Le DataLoader ne contient aucun exemple.")

    return total_loss / total_examples, correct_predictions / total_examples


def collect_predictions(model, loader):
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for eeg, labels in loader:
            logits = model(eeg)
            predictions = logits.argmax(dim=1)
            all_labels.append(labels.numpy())
            all_predictions.append(predictions.numpy())
    return np.concatenate(all_labels), np.concatenate(all_predictions)


# ==================================================================
# 5. ENTRAÎNEMENT D'UN FOLD
# ==================================================================
def train_one_fold(
    validation_dyad: str,
    train_dyads: list[str],
    freeze_level: int,
    pretrained_state_dict: dict | None,
    results_dir: Path,
    models_dir: Path,
) -> dict:
    train_loader, validation_loader, _ = create_participant_dataloaders(
        classification_table=classification_table,
        dataset_root=DATASET_PATH,
        train_dyads=train_dyads,
        validation_dyads=[validation_dyad],
        test_dyads=TEST_DYADS,
        batch_size=BATCH_SIZE,
    )

    set_seed(RANDOM_SEED)

    model = EEGNet(n_chans=N_CHANNELS, n_outputs=N_CLASSES, n_times=N_TIMES)

    if pretrained_state_dict is not None:
        model = load_pretrained_backbone(model, pretrained_state_dict)
    else:
        print("[Baseline from-scratch] pas de chargement de poids pré-entraînés.")

    model = set_freeze_level(model, freeze_level)

    trainable, frozen = count_parameters(model)
    print(
        f"Fold {validation_dyad} | freeze_level={freeze_level} | "
        f"entraînables={trainable:,} | gelés={frozen:,}"
    )

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=LEARNING_RATE)

    best_validation_loss = float("inf")
    best_validation_accuracy = None
    best_epoch = None
    epochs_without_improvement = 0
    best_model_path = models_dir / f"best_model_fold_{validation_dyad}.pt"

    history = {"train_loss": [], "train_accuracy": [], "validation_loss": [], "validation_accuracy": []}

    for epoch_index in range(NUMBER_OF_EPOCHS):
        epoch_number = epoch_index + 1

        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            optimizer,
        )

        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        improved = (
                validation_loss
                < best_validation_loss - EARLY_STOPPING_MIN_DELTA
        )

        if improved:
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch_number

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        print(
            f"Fold {validation_dyad} | "
            f"Freeze {freeze_level} | "
            f"Epoch {epoch_number:03d}/{NUMBER_OF_EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.4f} | "
            f"Val loss: {validation_loss:.4f} | "
            f"Val acc: {validation_accuracy:.4f}"
        )

        print(
            f"Epochs sans amélioration : "
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping — fold {validation_dyad}, "
                f"freeze {freeze_level}, "
                f"meilleure epoch : {best_epoch}"
            )
            break
    # Vérifie qu'au moins une amélioration a été enregistrée.
    #
    # Normalement, la première époque est toujours enregistrée puisque
    # best_validation_loss est initialisée à l'infini. Cette vérification
    # permet néanmoins de détecter explicitement un problème numérique,
    # par exemple une validation loss égale à NaN.
    if best_epoch is None or best_validation_accuracy is None:
        raise RuntimeError(
            f"Aucun meilleur modèle n'a été enregistré pour le fold "
            f"{validation_dyad} avec le niveau de freeze {freeze_level}."
        )

    # Recharge les paramètres correspondant au meilleur checkpoint.
    #
    # L'évaluation finale doit porter sur l'époque ayant obtenu la
    # meilleure validation loss et non sur la dernière époque exécutée.
    best_state_dict = torch.load(
        best_model_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(best_state_dict)

    # Récupération des prédictions du meilleur modèle.
    validation_labels, validation_predictions = collect_predictions(
        model,
        validation_loader,
    )

    # Calcul des métriques détaillées.
    matrix = confusion_matrix(
        validation_labels,
        validation_predictions,
    )

    report = classification_report(
        validation_labels,
        validation_predictions,
        target_names=["YO", "YF"],
        digits=4,
        zero_division=0,
    )

    # Chaque combinaison fold/niveau de freeze possède son propre
    # dossier afin d'éviter d'écraser les résultats.
    fold_results_dir = results_dir / f"fold_{validation_dyad}"
    fold_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Sauvegarde des métriques textuelles.
    report_path = fold_results_dir / "evaluation_report.txt"

    with report_path.open(
        mode="w",
        encoding="utf-8",
    ) as report_file:
        report_file.write(
            f"Dyade de validation : {validation_dyad}\n"
        )
        report_file.write(
            f"Niveau de freeze : {freeze_level}\n"
        )
        report_file.write(
            f"Meilleure époque : {best_epoch}\n"
        )
        report_file.write(
            f"Meilleure validation loss : "
            f"{best_validation_loss:.6f}\n"
        )
        report_file.write(
            f"Accuracy correspondante : "
            f"{best_validation_accuracy:.6f}\n"
        )
        report_file.write(
            f"Paramètres entraînables : {trainable}\n"
        )
        report_file.write(
            f"Paramètres gelés : {frozen}\n"
        )

        report_file.write("\nMatrice de confusion :\n")
        report_file.write(str(matrix))

        report_file.write("\n\nClassification report :\n")
        report_file.write(report)

    # Sauvegarde de l'historique sous forme de tableau.
    history_table = pd.DataFrame(history)
    history_table.insert(
        0,
        "epoch",
        range(1, len(history_table) + 1),
    )
    history_table.to_csv(
        fold_results_dir / "training_history.csv",
        index=False,
    )

    # Courbes d'apprentissage.
    epochs = range(1, len(history["train_loss"]) + 1)

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
    )

    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Train",
    )
    axes[0].plot(
        epochs,
        history["validation_loss"],
        label="Validation",
    )
    axes[0].axvline(
        best_epoch,
        color="black",
        linestyle="--",
        label="Meilleur checkpoint",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()

    axes[1].plot(
        epochs,
        history["train_accuracy"],
        label="Train",
    )
    axes[1].plot(
        epochs,
        history["validation_accuracy"],
        label="Validation",
    )
    axes[1].axvline(
        best_epoch,
        color="black",
        linestyle="--",
        label="Meilleur checkpoint",
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    figure.suptitle(
        f"Fold {validation_dyad} — freeze {freeze_level}"
    )
    figure.tight_layout()
    figure.savefig(
        fold_results_dir / "training_curves.png",
        dpi=150,
    )
    plt.close(figure)

    print("\nMatrice de confusion :")
    print(matrix)

    print("\nClassification report :")
    print(report)

    print(
        f"\nFold {validation_dyad} terminé | "
        f"freeze {freeze_level} | "
        f"meilleure époque : {best_epoch} | "
        f"validation loss : {best_validation_loss:.4f} | "
        f"validation accuracy : {best_validation_accuracy:.4f}"
    )

    # Ce dictionnaire est récupéré dans main() sous le nom « row ».
    #
    # main() lui ajoutera ensuite les colonnes freeze_name et condition
    # avant de construire le tableau récapitulatif.
    return {
        "validation_dyad": validation_dyad,
        "freeze_level": freeze_level,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_validation_accuracy": best_validation_accuracy,
        "final_train_loss": history["train_loss"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_validation_loss": history["validation_loss"][-1],
        "final_validation_accuracy": (
            history["validation_accuracy"][-1]
        ),
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
    }


# ==================================================================
# 6. PROGRAMME PRINCIPAL
# ==================================================================

def main() -> None:
    set_seed(RANDOM_SEED)

    print("Téléchargement du checkpoint pré-entraîné...")
    pretrained_state_dict = download_pretrained_state_dict()

    all_rows = []

    # Niveaux 0 à 3 : transfer learning avec différents degrés de gel.
    for freeze_level, freeze_name in FREEZE_LEVELS.items():
        print("\n" + "#" * 70)
        print(f"NIVEAU DE FREEZE {freeze_level} — {freeze_name}")
        print("#" * 70)

        level_results_dir = RESULTS_DIR / freeze_name
        level_models_dir = MODELS_DIR / freeze_name

        level_results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        level_models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for validation_dyad in DEVELOPMENT_DYADS:
            train_dyads = [d for d in DEVELOPMENT_DYADS if d != validation_dyad]
            row = train_one_fold(
                validation_dyad=validation_dyad,
                train_dyads=train_dyads,
                freeze_level=freeze_level,
                pretrained_state_dict=pretrained_state_dict,
                results_dir=level_results_dir,
                models_dir=level_models_dir,
            )
            row["freeze_name"] = freeze_name
            row["condition"] = "pretrained"
            all_rows.append(row)

    # Référence from-scratch : MÊME architecture braindecode, mais sans
    # aucun poids pré-entraîné (comparaison à variable unique contrôlée).
    print("\n" + "#" * 70)
    print("RÉFÉRENCE — EEGNet (braindecode) from-scratch, sans transfert")
    print("#" * 70)

    scratch_results_dir = RESULTS_DIR / "from_scratch_baseline"
    scratch_models_dir = MODELS_DIR / "from_scratch_baseline"

    scratch_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scratch_models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for validation_dyad in DEVELOPMENT_DYADS:
        train_dyads = [d for d in DEVELOPMENT_DYADS if d != validation_dyad]
        row = train_one_fold(
            validation_dyad=validation_dyad,
            train_dyads=train_dyads,
            freeze_level=0,
            pretrained_state_dict=None,
            results_dir=scratch_results_dir,
            models_dir=scratch_models_dir,
        )
        row["freeze_name"] = "from_scratch_baseline"
        row["condition"] = "from_scratch"
        all_rows.append(row)

    summary_table = pd.DataFrame(all_rows)
    summary_table.to_csv(RESULTS_DIR / "full_comparison.csv", index=False)

    comparison = (
        summary_table.groupby(["condition", "freeze_name"], as_index=False)
        .agg(
            mean_accuracy=("best_validation_accuracy", "mean"),
            std_accuracy=("best_validation_accuracy", "std"),
            mean_loss=("best_validation_loss", "mean"),
            trainable_parameters=("trainable_parameters", "first"),
        )
    )

    print("\n" + "=" * 70)
    print("COMPARAISON FINALE")
    print("=" * 70)
    print(comparison.to_string(index=False))
    comparison.to_csv(RESULTS_DIR / "comparison_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(comparison["freeze_name"], comparison["mean_accuracy"], yerr=comparison["std_accuracy"], capsize=4)
    plt.ylabel("Accuracy moyenne (validation, LODO CV)")
    plt.title("EEGNet pré-entraîné : niveaux de freeze vs from-scratch")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison.png", dpi=150)
    plt.close()

    print(f"\nRésultats sauvegardés dans : {RESULTS_DIR}")


if __name__ == "__main__":
    main()