"""
Leave-One-Dyad-Out Cross-Validation — protocole B.

OBJECTIF
--------
Évaluer la capacité du modèle à généraliser vers une dyade qui n'a jamais
été utilisée pendant l'entraînement.

PROTOCOLE B
-----------
Les dyades de développement sont utilisées dans une cross-validation LODO :

    - une dyade est placée en validation ;
    - toutes les autres dyades de développement sont utilisées en train ;
    - on recommence jusqu'à ce que chaque dyade de développement ait été
      utilisée une fois comme validation.

La dyade J15 reste totalement séparée :

    - elle ne participe à aucun entraînement ;
    - elle ne participe à aucune validation ;
    - elle sera utilisée une seule fois à la fin du projet pour évaluer
      le modèle et l'architecture retenus.

Exemple pour le fold J1 :

    Train      : J2, J4, J5, J7, J8, J10
    Validation : J1
    Test final : J15, non utilisé à ce stade
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.dataset.labels import prepare_classification_table
from src.dataset.dataloader_participant import create_participant_dataloaders
from src.models.participant_model import SimpleParticipantClassifier


# ==================================================================
# 1. CONFIGURATION GÉNÉRALE
# ==================================================================

# __file__ correspond au chemin de ce script.
#
# Exemple :
# projet/src/training/cross_validation.py
#
# parents[0] = training
# parents[1] = src
# parents[2] = projet
#
# PROJECT_ROOT pointe donc vers la racine du projet.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Dyades utilisées pour développer et comparer les modèles.
#
# Chaque dyade sera utilisée exactement une fois comme validation.
DEVELOPMENT_DYADS = [
    "J1",
    "J2",
    "J4",
    "J5",
    "J7",
    "J8",
    "J10",
    'J15',
]

# J15 reste en dehors de toute la cross-validation.
#
# Attention :
# tant que nous sélectionnons l'architecture ou les hyperparamètres,
# nous ne devons pas interpréter les performances sur J15.
TEST_DYADS = []

# Nombre d'exemples transmis au modèle avant chaque mise à jour des poids.
BATCH_SIZE = 5


# Nombre maximal d'epochs pour chaque fold.
#
# Ici, on conserve 100 epochs pour observer entièrement le comportement
# de la baseline, notamment le surapprentissage.
NUMBER_OF_EPOCHS = 100


# Taux d'apprentissage de l'optimiseur Adam.
LEARNING_RATE = 0.001


# Seed utilisée pour rendre les expériences plus reproductibles.
#
# Avec la même seed, PyTorch initialise normalement les poids du modèle
# de la même manière lors de chaque lancement.
RANDOM_SEED = 42


# Dossiers de sortie.
#
# Les tableaux, historiques et graphiques sont placés dans results/protocol_B_j15_test.
# Les poids des meilleurs modèles sont placés dans models/protocol_B_j15_test.
EXPERIMENT_NAME = "protocol_A_all_dyads"
RESULTS_DIR = PROJECT_ROOT / "results" / EXPERIMENT_NAME
MODELS_DIR = PROJECT_ROOT / "models" / EXPERIMENT_NAME

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# La CrossEntropyLoss est adaptée à une classification multiclasse.
#
# Ici, le modèle retourne deux logits :
#
#     logits[:, 0] = score associé à YO
#     logits[:, 1] = score associé à YF
#
# Les labels attendus sont donc des entiers :
#
#     YO -> 0
#     YF -> 1
criterion = nn.CrossEntropyLoss()


# ==================================================================
# 2. REPRODUCTIBILITÉ
# ==================================================================

def set_seed(seed: int) -> None:
    """Fixe les générateurs pseudo-aléatoires utilisés dans le projet.

    Cela limite les variations entre deux exécutions dues à :

    - l'initialisation aléatoire des poids ;
    - le mélange des données ;
    - certaines opérations internes de PyTorch.

    Une seed ne garantit pas toujours une reproductibilité parfaite sur
    tous les matériels, mais elle améliore fortement la comparabilité.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Cette ligne concerne l'utilisation éventuelle d'un GPU NVIDIA.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================================================================
# 3. CHARGEMENT DES MÉTADONNÉES ET CRÉATION DES LABELS
# ==================================================================

metadata_path = PROJECT_ROOT / "data" / "all_metadata.csv"

# Lecture du tableau qui contient les informations sur les participants,
# les dyades et la classe cible.
metadata = pd.read_csv(metadata_path)


# Transformation des métadonnées brutes en tableau adapté à la tâche
# de classification.
#
# Seuls les participants dont eyes_code vaut YO ou YF sont conservés.
#
# Le mapping convertit les classes textuelles en entiers utilisables
# par CrossEntropyLoss :
#
#     YO -> 0
#     YF -> 1
classification_table = prepare_classification_table(
    metadata=metadata,
    target_column="eyes_code",
    allowed_classes=["YO", "YF"],
    label_map={
        "YO": 0,
        "YF": 1,
    },
)


# ==================================================================
# 4. EXÉCUTION D'UNE EPOCH
# ==================================================================

def run_epoch(model: nn.Module,loader, optimizer: torch.optim.Optimizer | None = None,) :
    """Exécute une passe complète sur un DataLoader.

    Le comportement dépend de la présence de l'optimiseur.

    Cas 1 : optimizer est fourni
    ----------------------------
    La fonction effectue une epoch d'entraînement :

        - model.train()
        - calcul des prédictions ;
        - calcul de la loss ;
        - calcul des gradients ;
        - mise à jour des poids.

    Cas 2 : optimizer vaut None
    ---------------------------
    La fonction effectue une epoch de validation :

        - model.eval()
        - aucun calcul de gradients ;
        - aucune modification des poids.

    Parameters
    ----------
    model:
        Réseau de neurones à entraîner ou évaluer.

    loader:
        DataLoader fournissant des couples :

            eeg, labels

        eeg contient les tenseurs EEG.
        labels contient les classes attendues, 0 ou 1.

    optimizer:
        Optimiseur utilisé seulement pendant l'entraînement.

    Returns
    -------
    average_loss:
        Loss moyenne par exemple sur toute l'epoch.

    accuracy:
        Proportion d'exemples correctement classés.
    """

    # Si un optimiseur est fourni, nous sommes en entraînement.
    training = optimizer is not None

    if training:
        # Active le comportement d'entraînement du modèle.
        #
        # Cela est important pour les couches qui se comportent
        # différemment en train et en validation, comme Dropout ou
        # BatchNorm.
        model.train()
    else:
        # Désactive les comportements propres à l'entraînement.
        model.eval()

    # Somme des losses pondérées par le nombre d'exemples.
    total_loss = 0.0

    # Nombre total de prédictions correctes.
    correct_predictions = 0

    # Nombre total d'exemples parcourus.
    total_examples = 0

    # En entraînement, PyTorch doit conserver le graphe de calcul afin
    # de pouvoir calculer les gradients.
    #
    # En validation, torch.no_grad() évite de stocker ce graphe :
    # cela économise de la mémoire et accélère les calculs.
    gradient_context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with gradient_context:

        # Le DataLoader fournit les données batch par batch.
        for eeg, labels in loader:

            # En entraînement, on efface les gradients calculés lors
            # du batch précédent.
            #
            # PyTorch additionne les gradients par défaut. Sans cette
            # instruction, les gradients de plusieurs batches seraient
            # cumulés involontairement.
            if training:
                optimizer.zero_grad()

            # Passage avant :
            # le modèle transforme les EEG en logits.
            #
            # Exemple de forme possible :
            #
            #     eeg         : [batch_size, 32, 5120]
            #     predictions : [batch_size, 2]
            predictions = model(eeg)

            # Comparaison entre les logits prédits et les labels réels.
            loss = criterion(predictions, labels)

            if training:
                # Calcule les gradients de la loss par rapport aux
                # paramètres du modèle.
                loss.backward()

                # Modifie les poids du modèle avec Adam.
                optimizer.step()

            # Taille réelle du batch.
            #
            # Le dernier batch peut contenir moins de BATCH_SIZE
            # exemples. Il faut donc utiliser sa taille réelle.
            current_batch_size = labels.size(0)

            # loss.item() est la moyenne de la loss dans le batch.
            #
            # On la multiplie par la taille du batch pour obtenir une
            # somme, avant de diviser plus tard par le nombre total
            # d'exemples.
            total_loss += loss.item() * current_batch_size

            # argmax sélectionne l'indice du logit le plus élevé :
            #
            #     indice 0 -> classe YO
            #     indice 1 -> classe YF
            predicted_classes = predictions.argmax(dim=1)

            # Comparaison élément par élément entre prédictions et labels.
            correct_predictions += (
                predicted_classes == labels
            ).sum().item()

            total_examples += current_batch_size

    # Une absence d'exemples indiquerait généralement un problème
    # dans les listes de dyades ou dans le Dataset.
    if total_examples == 0:
        raise ValueError(
            "Le DataLoader ne contient aucun exemple. "
            "Vérifie les dyades et la classification_table."
        )

    # Moyenne exacte de la loss sur les exemples, et non simple moyenne
    # des batches.
    average_loss = total_loss / total_examples

    # Exemple :
    #
    #     60 prédictions correctes / 100 exemples = 0.60
    accuracy = correct_predictions / total_examples

    return average_loss, accuracy


# ==================================================================
# 5. SAUVEGARDE DES COURBES D'UN FOLD
# ==================================================================

def save_fold_results(
    validation_dyad: str,
    history: dict[str, list[float]],
) -> None:
    """Sauvegarde l'historique et les graphiques d'un fold.

    Pour chaque dyade de validation, un dossier séparé est créé :

        results/protocol_B_j15_test/fold_J1/
            history.csv
            loss_curve.png
            accuracy_curve.png
    """

    fold_dir = RESULTS_DIR / f"fold_{validation_dyad}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    number_of_recorded_epochs = len(history["train_loss"])
    epochs = range(1, number_of_recorded_epochs + 1)

    # Transformation de l'historique Python en tableau.
    history_table = pd.DataFrame(history)

    # Ajout d'une colonne d'epoch commençant à 1.
    history_table.insert(
        loc=0,
        column="epoch",
        value=epochs,
    )

    history_table.to_csv(
        fold_dir / "history.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Figure 1 : train loss contre validation loss
    # --------------------------------------------------------------
    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        epochs,
        history["validation_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Loss — dyade de validation : {validation_dyad}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        fold_dir / "loss_curve.png",
        dpi=150,
    )

    # Fermer la figure évite de conserver toutes les figures en mémoire
    # pendant l'exécution des sept folds.
    plt.close()

    # --------------------------------------------------------------
    # Figure 2 : train accuracy contre validation accuracy
    # --------------------------------------------------------------
    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy",
    )

    plt.plot(
        epochs,
        history["validation_accuracy"],
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        f"Accuracy — dyade de validation : {validation_dyad}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        fold_dir / "accuracy_curve.png",
        dpi=150,
    )

    plt.close()


# ==================================================================
# 6. ENTRAÎNEMENT D'UN SEUL FOLD
# ==================================================================

def train_one_fold(
    validation_dyad: str,
    train_dyads: list[str],
) -> tuple[dict, float, float, int]:
    """Entraîne un modèle neuf pour un fold de la LODO.

    Exemple pour validation_dyad = "J4" :

        train_dyads     = J1, J2, J5, J7, J8, J10
        validation_dyad = J4
        test_dyads      = J15

    Le modèle entraîné dans ce fold ne doit jamais voir les exemples
    de J4 pendant l'optimisation.

    Returns
    -------
    history:
        Toutes les métriques enregistrées à chaque epoch.

    best_validation_loss:
        Plus faible validation loss observée.

    best_validation_accuracy:
        Accuracy observée à l'exact même epoch que la meilleure loss.

    best_epoch:
        Numéro de l'epoch où la meilleure validation loss a été observée.
    """

    # Vérification contre une fuite directe :
    # la dyade de validation ne doit pas se trouver dans le train.
    if validation_dyad in train_dyads:
        raise ValueError(
            f"{validation_dyad} est présente à la fois "
            "dans train_dyads et dans validation_dyad."
        )

    # Vérification que le test final ne se retrouve pas dans le train.
    overlap_train_test = set(train_dyads) & set(TEST_DYADS)

    if overlap_train_test:
        raise ValueError(
            "Certaines dyades sont présentes à la fois dans le train "
            f"et le test : {overlap_train_test}"
        )

    # Création des trois DataLoaders.
    #
    # Le test_loader est créé par la fonction, mais il ne sera pas utilisé
    # pendant la cross-validation.
    train_loader, validation_loader, _ = (
        create_participant_dataloaders(
            classification_table=classification_table,
            dataset_root=PROJECT_ROOT / "data" / "data_toy",
            train_dyads=train_dyads,
            validation_dyads=[validation_dyad],
            test_dyads=TEST_DYADS,
            batch_size=BATCH_SIZE,
        )
    )

    # Réinitialisation de la seed avant chaque modèle.
    #
    # De cette manière, les folds commencent avec la même initialisation
    # des poids. Cela facilite leur comparaison.
    set_seed(RANDOM_SEED)

    # Création d'un modèle entièrement neuf.
    #
    # Il est indispensable de recréer le modèle dans chaque fold :
    # réutiliser le modèle du fold précédent provoquerait une fuite
    # d'information.
    model = SimpleParticipantClassifier()

    # Création d'un optimiseur entièrement neuf, associé uniquement
    # aux paramètres du modèle de ce fold.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Historique des métriques, utilisé pour construire les courbes.
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    # float("inf") garantit que la loss de la première epoch sera
    # inférieure à la valeur initiale.
    best_validation_loss = float("inf")

    # Accuracy observée à la même epoch que best_validation_loss.
    best_validation_accuracy = None

    # Numéro de la meilleure epoch.
    best_epoch = None

    # Chaque fold possède son propre fichier de poids.
    best_model_path = (
        MODELS_DIR
        / f"best_model_fold_{validation_dyad}.pt"
    )

    for epoch_index in range(NUMBER_OF_EPOCHS):

        # epoch_index commence à 0, tandis qu'on souhaite afficher les
        # epochs à partir de 1.
        epoch_number = epoch_index + 1

        # Entraînement sur toutes les dyades de train.
        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
        )

        # Évaluation sur la dyade laissée de côté.
        #
        # Aucun optimiseur n'est fourni : run_epoch passe donc
        # automatiquement en mode validation.
        validation_loss, validation_accuracy = run_epoch(
            model=model,
            loader=validation_loader,
        )

        # Sauvegarde des métriques de cette epoch.
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(
            validation_accuracy
        )

        # Le meilleur modèle est défini selon la validation loss.
        #
        # On utilise la loss plutôt que l'accuracy, car la loss prend
        # également en compte le niveau de confiance des prédictions.
        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            # Très important :
            # cette accuracy correspond exactement au même modèle et à
            # la même epoch que la meilleure validation loss.
            best_validation_accuracy = validation_accuracy

            best_epoch = epoch_number

            # On sauvegarde uniquement les paramètres du modèle.
            torch.save(
                model.state_dict(),
                best_model_path,
            )

        print(
            f"Fold {validation_dyad} | "
            f"Epoch {epoch_number:03d}/{NUMBER_OF_EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.4f} | "
            f"Val loss: {validation_loss:.4f} | "
            f"Val acc: {validation_accuracy:.4f}"
        )

    # Ces valeurs ne devraient jamais rester à None puisque le premier
    # epoch améliore forcément une valeur initiale égale à l'infini.
    if best_epoch is None or best_validation_accuracy is None:
        raise RuntimeError(
            "Aucun meilleur modèle n'a été enregistré."
        )

    save_fold_results(
        validation_dyad=validation_dyad,
        history=history,
    )

    print(
        f"\n[Fold validation={validation_dyad}] "
        f"meilleure validation loss = "
        f"{best_validation_loss:.4f} | "
        f"accuracy correspondante = "
        f"{best_validation_accuracy:.4f} | "
        f"epoch = {best_epoch}\n"
    )

    return (
        history,
        best_validation_loss,
        best_validation_accuracy,
        best_epoch,
    )


# ==================================================================
# 7. GRAPHIQUES GLOBAUX DE COMPARAISON
# ==================================================================

def save_global_comparison(
    all_histories: dict[str, dict[str, list[float]]],
) -> None:
    """Compare les performances de validation de tous les folds."""

    epochs = range(1, NUMBER_OF_EPOCHS + 1)

    # --------------------------------------------------------------
    # Validation loss de toutes les dyades
    # --------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for validation_dyad, history in all_histories.items():
        plt.plot(
            epochs,
            history["validation_loss"],
            label=f"Dyade {validation_dyad}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(
        "Validation Loss par fold "
        "(Leave-One-Dyad-Out)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "all_folds_validation_loss.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------------
    # Validation accuracy de toutes les dyades
    # --------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for validation_dyad, history in all_histories.items():
        plt.plot(
            epochs,
            history["validation_accuracy"],
            label=f"Dyade {validation_dyad}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(
        "Validation Accuracy par fold "
        "(Leave-One-Dyad-Out)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "all_folds_validation_accuracy.png",
        dpi=150,
    )

    plt.close()


# ==================================================================
# 8. PROGRAMME PRINCIPAL
# ==================================================================

def main() -> None:
    """Exécute les sept folds du protocole B."""

    set_seed(RANDOM_SEED)

    # Dictionnaire utilisé pour conserver l'historique complet de
    # chaque dyade et produire ensuite les graphiques globaux.
    all_histories = {}

    # Liste de dictionnaires qui sera transformée en DataFrame.
    fold_summary = []

    for validation_dyad in DEVELOPMENT_DYADS:

        # Pour un fold donné, toutes les dyades sauf celle de validation
        # sont placées dans le train.
        train_dyads = [
            dyad
            for dyad in DEVELOPMENT_DYADS
            if dyad != validation_dyad
        ]

        print("=" * 70)
        print(
            f"FOLD - dyade de validation : {validation_dyad}"
        )
        print(f"Train dyads      : {train_dyads}")
        print(f"Validation dyad  : {[validation_dyad]}")
        print(f"Test dyads isolés: {TEST_DYADS}")
        print("=" * 70)

        (
            history,
            best_validation_loss,
            best_validation_accuracy,
            best_epoch,
        ) = train_one_fold(
            validation_dyad=validation_dyad,
            train_dyads=train_dyads,
        )

        all_histories[validation_dyad] = history

        # Les métriques "best" correspondent toutes à la même epoch.
        #
        # Les métriques "final" correspondent à la dernière epoch.
        # Elles sont conservées uniquement pour analyser le comportement
        # du modèle après 100 epochs.
        fold_summary.append(
            {
                "validation_dyad": validation_dyad,
                "best_epoch": best_epoch,
                "best_validation_loss": (
                    best_validation_loss
                ),
                "best_validation_accuracy": (
                    best_validation_accuracy
                ),
                "final_train_loss": (
                    history["train_loss"][-1]
                ),
                "final_train_accuracy": (
                    history["train_accuracy"][-1]
                ),
                "final_validation_loss": (
                    history["validation_loss"][-1]
                ),
                "final_validation_accuracy": (
                    history["validation_accuracy"][-1]
                ),
            }
        )

    # Création du tableau récapitulatif des sept folds.
    summary_table = pd.DataFrame(fold_summary)

    # Sauvegarde du tableau détaillé.
    summary_table.to_csv(
        RESULTS_DIR / "lodo_cv_summary.csv",
        index=False,
    )

    # Calcul des statistiques de cross-validation.
    mean_accuracy = summary_table[
        "best_validation_accuracy"
    ].mean()

    std_accuracy = summary_table[
        "best_validation_accuracy"
    ].std()

    mean_loss = summary_table[
        "best_validation_loss"
    ].mean()

    std_loss = summary_table[
        "best_validation_loss"
    ].std()

    print("\n" + "=" * 70)
    print("RÉSUMÉ GLOBAL DE LA CROSS-VALIDATION")
    print("=" * 70)
    print(summary_table.to_string(index=False))
    print()
    print(
        "Best validation accuracy moyenne : "
        f"{mean_accuracy:.4f}"
    )
    print(
        "Écart-type de l'accuracy : "
        f"{std_accuracy:.4f}"
    )
    print(
        "Best validation loss moyenne : "
        f"{mean_loss:.4f}"
    )
    print(
        "Écart-type de la loss : "
        f"{std_loss:.4f}"
    )

    save_global_comparison(all_histories)


# Cette condition empêche la cross-validation de se lancer
# automatiquement si le fichier est importé depuis un autre script.
#
# Elle ne s'exécute que lorsque le fichier est lancé directement :
#
#     python cross_validation.py
if __name__ == "__main__":
    main()