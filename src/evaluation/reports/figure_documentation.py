"""Produit la fiche d'explication de chaque figure présentée à Amel.

Chaque ligne répond aux quatre questions obligatoires : input, opération,
dimension d'agrégation et output. Les formules sont écrites explicitement pour
éviter les formulations vagues comme « on a fait une moyenne ».
"""

from __future__ import annotations

import pandas as pd

from src.config.settings import REPORT_OUTPUT_ROOT


FIGURE_DOCUMENTATION = (
    {
        "figure": "amplitudes_by_participant.png",
        "input": "Chaque fichier EEG [2 participants, 32 canaux, 5120 temps]",
        "operation": (
            "Pour chaque fichier et participant : max(abs(x)) sur les 32×5120 "
            "valeurs, puis moyenne de ces maxima sur les fichiers d'une dyade."
        ),
        "dimensions_reduced": (
            "max sur axes canal et temps ; moyenne sur axe fichier"
        ),
        "output": (
            "Une barre par couple dyade/participant : amplitude absolue "
            "maximale moyenne."
        ),
        "x_axis": "Dyade",
        "y_axis": "Amplitude absolue maximale moyenne, échelle logarithmique",
        "unit": "Volt si les fichiers source sont en volts",
        "warning": (
            "Une grande barre localise une amplitude atypique, mais ne prouve "
            "pas à elle seule une cause d'acquisition."
        ),
    },
    {
        "figure": "heatmap_electrodes_outliers.png",
        "input": "Pour chaque participant/canal : 5120 amplitudes temporelles",
        "operation": (
            "n = somme_t 1(|x_t| > seuil), puis somme sur participants et "
            "fichiers appartenant à la même dyade."
        ),
        "dimensions_reduced": "somme sur temps, participants et fichiers",
        "output": "Chaque cellule = nombre total de valeurs hors seuil",
        "x_axis": "Électrode, dans l'ordre réel des 32 canaux",
        "y_axis": "Dyade",
        "unit": "Nombre de points temporels",
        "warning": (
            "Le seuil absolu 0,01 reste exploratoire ; la figure décrit ce "
            "critère et ne valide pas sa pertinence physiologique."
        ),
    },
    {
        "figure": "outliers_by_region.png",
        "input": "Nombre de valeurs hors seuil par électrode",
        "operation": (
            "Somme des nombres d'outliers des électrodes rattachées à une "
            "même région large."
        ),
        "dimensions_reduced": "somme sur électrodes d'une région",
        "output": "Une barre = total de points hors seuil dans la région",
        "x_axis": "Nombre total de valeurs hors seuil",
        "y_axis": "Région cérébrale",
        "unit": "Nombre de points temporels",
        "warning": (
            "Les régions ne contiennent pas toutes le même nombre "
            "d'électrodes ; le total brut dépend aussi de cette taille."
        ),
    },
    {
        "figure": "acquisition_vs_physiology.png",
        "input": "Nombre d'outliers P1 et P2 agrégé par dyade",
        "operation": "ratio = outliers_P1 / (outliers_P1 + outliers_P2)",
        "dimensions_reduced": "somme préalable sur fichiers, canaux et temps",
        "output": "Une barre = part des outliers portée par P1",
        "x_axis": "Dyade",
        "y_axis": "Ratio compris entre 0 et 1",
        "unit": "Sans unité",
        "warning": (
            "Un ratio extrême localise l'anomalie sur un participant. Il ne "
            "permet pas, seul, de distinguer acquisition et physiologie."
        ),
    },
    {
        "figure": "electrode_positions_and_neighbors.png",
        "input": "Dictionnaire externe de coordonnées 2D des 32 électrodes",
        "operation": (
            "Distance euclidienne via cdist ; filtrage même région ; sélection "
            "des quatre plus proches ; interpolation pondérée par 1/distance."
        ),
        "dimensions_reduced": "aucune statistique EEG sur cette figure",
        "output": "Montage 2D et segments entre FC5 et ses voisins retenus",
        "x_axis": "Coordonnée gauche-droite relative",
        "y_axis": "Coordonnée arrière-avant relative",
        "unit": "Coordonnées relatives",
        "warning": "Les positions ne sont pas stockées dans les fichiers .npy.",
    },
    {
        "figure": "boxplot_<groupe>.png",
        "input": "8 best_validation_accuracy, une par fold LODO",
        "operation": (
            "Médiane, Q1, Q3 et moustaches du boxplot ; tous les folds sont "
            "aussi affichés comme points ; moyenne affichée par un losange."
        ),
        "dimensions_reduced": "distribution sur l'axe dyade",
        "output": "Une boîte par configuration expérimentale",
        "x_axis": "Configuration",
        "y_axis": "Accuracy au checkpoint de loss minimale (%)",
        "unit": "Pourcentage",
        "warning": (
            "Les huit dyades ne sont pas des répétitions totalement "
            "indépendantes : leurs ensembles d'entraînement se recouvrent."
        ),
    },
    {
        "figure": "<experience>_loss_8folds.png",
        "input": "Historique train_loss et validation_loss à chaque epoch",
        "operation": "Aucune moyenne entre folds ; une sous-figure par dyade",
        "dimensions_reduced": "loss déjà moyennée sur les exemples du loader",
        "output": "Deux courbes et une ligne verticale au meilleur checkpoint",
        "x_axis": "Epoch",
        "y_axis": "BCE moyenne",
        "unit": "Sans unité",
        "warning": (
            "Pour SignalJEPA, les courbes train/validation sont au niveau "
            "fenêtre ; le checkpoint est sélectionné au niveau participant."
        ),
    },
    {
        "figure": "<experience>_accuracy_8folds.png",
        "input": "Historique train_accuracy et validation_accuracy par epoch",
        "operation": "Aucune moyenne entre folds ; une sous-figure par dyade",
        "dimensions_reduced": "moyenne des décisions correctes sur les exemples",
        "output": "Courbes train/validation et époque du meilleur checkpoint",
        "x_axis": "Epoch",
        "y_axis": "Accuracy entre 0 et 1",
        "unit": "Proportion",
        "warning": "La sélection du checkpoint repose sur la loss, pas l'accuracy.",
    },
    {
        "figure": "architecture_<modele>.png",
        "input": "Tenseur factice [1, 32, 5120]",
        "operation": "Forward hooks sur chaque couche feuille du modèle",
        "dimensions_reduced": "selon les convolutions, poolings et flatten",
        "output": "Ordre réel, forme de sortie et paramètres de chaque couche",
        "x_axis": "Sans objet",
        "y_axis": "Ordre des couches",
        "unit": "Nombre de tenseurs / paramètres",
        "warning": "Le schéma décrit l'architecture, pas les performances.",
    },
)


def main() -> None:
    """Sauvegarde la fiche en CSV et en Markdown lisible."""

    REPORT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    documentation_table = pd.DataFrame(FIGURE_DOCUMENTATION)

    csv_path = REPORT_OUTPUT_ROOT / "figure_documentation.csv"
    markdown_path = REPORT_OUTPUT_ROOT / "figure_documentation.md"

    documentation_table.to_csv(csv_path, index=False)
    markdown_path.write_text(
        "# Documentation scientifique des figures\n\n"
        + documentation_table.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )

    print(f"Documentation CSV      : {csv_path}")
    print(f"Documentation Markdown : {markdown_path}")


if __name__ == "__main__":
    main()

