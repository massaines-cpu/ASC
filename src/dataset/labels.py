#fonctions de creation labels
from pathlib import Path
import pandas as pd


# Prépare un DataFrame pour une tâche de classification.
  #
  # Parameters
  # metadata
  #     DataFrame contenant toutes les métadonnées.
  #
  # target_column
  #     Colonne utilisée comme cible.
  #     Exemple : "eyes_code", "task", "condition_code".
  #
  # allowed_classes
  #     Classes que l'on souhaite conserver.
  #     Exemple : ["YO", "YF"].
  #
  # label_map
  #     Conversion des classes texte en nombres.
  #     Exemple : {"YO": 0, "YF": 1}.
  #
  # Returns
  # DataFrame filtré avec une colonne numérique "label".

def prepare_classification_table(metadata,target_column,allowed_classes,label_map):
    if target_column not in metadata.columns:
        raise ValueError(f'la colonne {target_column} n\'existe pas')

    table_filtree = metadata[metadata[target_column].isin(allowed_classes)].copy()

    table_filtree['label'] = table_filtree[target_column].map(label_map)

    if table_filtree['label'].isna().any():
        raise ValueError('certaines classes n\'ont pas de label numeriques')

    table_filtree = table_filtree.reset_index(drop=True)

    return table_filtree

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]
    metadata_path = project_root / "data" / "all_metadata.csv"

    metadata = pd.read_csv(metadata_path)

    eyes_table = prepare_classification_table(
        metadata=metadata,
        target_column="eyes_code",
        allowed_classes=["YO", "YF"],
        label_map={
            "YO": 0,
            "YF": 1,
        },
    )

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