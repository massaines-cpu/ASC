"""Affiche les candidats les mieux documentés depuis la base SQLite."""

from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "database" / "eeg_model_catalog.sqlite3"


def main() -> None:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            "La base n'existe pas encore. Lance d'abord run_registry.py."
        )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT
            "Nom du modèle",
            "Poids réellement disponibles",
            "Nombre de canaux du pré-entraînement",
            "Canaux fixes ou variables",
            "Peut recevoir mes 32 canaux ASC",
            "Score de compatibilité",
            "Confiance des métadonnées"
        FROM asc_model_catalog
        ORDER BY "Score de compatibilité" DESC
        LIMIT 30
        """
    ).fetchall()

    print("\nCANDIDATS LES MIEUX CLASSÉS\n")
    for row in rows:
        print(
            f"{row['Nom du modèle']}\n"
            f"  Poids : {row['Poids réellement disponibles']} | "
            f"Canaux : {row['Nombre de canaux du pré-entraînement']} "
            f"({row['Canaux fixes ou variables']})\n"
            f"  ASC 32 : {row['Peut recevoir mes 32 canaux ASC']} | "
            f"Score : {row['Score de compatibilité']} | "
            f"Confiance : {row['Confiance des métadonnées']}\n"
        )

    connection.close()


if __name__ == "__main__":
    main()

