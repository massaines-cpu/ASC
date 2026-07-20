#fonctions de creation labels
from pathlib import Path
import pandas as pd

#fichier préparation de données

# A partir du fichier all_metadata.csv, on :
#   1. garde uniquement les classes qui nous intéressent
#   2. transforme les labels textuels en labels numériques
#   3. retourne une nouvelle table prête à être utilisée

def prepare_classification_table(metadata,target_column,allowed_classes,label_map):
    """
        Prépare une table de classification.

        Paramètres:
        metadata : DataFrame
            table contenant toutes les métadonnées

        target_column : str
            nom de la colonne qui contient les classes à prédire
            exemple : eyes_code

        allowed_classes : list
            liste des classes que l'on souhaite conserver
            exemple :
                ["YO", "YF"]

        label_map : dict
            dictionnaire permettant de convertir les classes textuelles
            en entiers
            exemple :
                {
                    "YO": 0,
                    "YF": 1
                }

        Return:
        DataFrame
            nouvelle table filtrée contenant une colonne "label"
        """
# vérifie que la colonne demandée existe bien dans le DataFrame
# sinon, on arrête immédiatement le programme

    if target_column not in metadata.columns:
        raise ValueError(f'la colonne {target_column} n\'existe pas')
    # On garde uniquement les lignes appartenant aux classes
    # qui nous intéressent
    table_filtree = metadata[metadata[target_column].isin(allowed_classes)].copy()
    # Création d'une nouvelle colonne "label".
    # map() remplace automatiquement chaque classe textuelle
    # par son entier correspondant
    table_filtree['label'] = table_filtree[target_column].map(label_map)

    # Vérifie que toutes les classes ont bien reçu un label.
    #
    # Si une classe n'est pas présente dans label_map,
    # map() crée une valeur NaN.

    if table_filtree['label'].isna().any():
        raise ValueError('certaines classes n\'ont pas de label numeriques')
    # Après le filtrage, les indices ne sont plus forcément continus.

    # reset_index(drop=True) au lieu de 0 1 4 6 on remet zero 0 1 2 3 4
    table_filtree = table_filtree.reset_index(drop=True)

    return table_filtree

# Partie exécutée uniquement si ce fichier est lancé directement

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]
    # Construction du chemin vers all_metadata.csv

    metadata_path = project_root / "data" / "all_metadata.csv"
    # Lecture du fichier CSV dans un DataFrame pandas

    metadata = pd.read_csv(metadata_path)
    # Préparation de la classification "Eyes Open" vs
    # "Eyes Closed"
    eyes_table = prepare_classification_table(
        metadata=metadata,
        target_column="eyes_code",
        allowed_classes=["YO", "YF"],
        label_map={
            "YO": 0,
            "YF": 1,
        },
    )
    # Affiche uniquement quelques colonnes utiles
    # pour vérifier rapidement que tout fonctionne.

    print(eyes_table[
        [
            "filename",
            "dyad_id",
            "eyes_code",
            "label",
        ]
    ].head())

    print("\nrépartition des labels :")
    print(eyes_table["label"].value_counts())