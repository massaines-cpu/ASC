"""Lance les audits et figures qui ne nécessitent aucun nouvel entraînement."""

from src.evaluation.audits import (
    audit_best_checkpoints,
    audit_binary_pipeline,
)
from src.evaluation.reports import figure_documentation
from src.evaluation.visualizations import (
    generate_architecture_diagrams,
    generate_boxplots,
    generate_fold_grids,
)

def main() -> None:
    """Exécute chaque étape dans un ordre qui facilite le diagnostic."""

    print("\n1/6 — Vérification de la sortie binaire")
    audit_binary_pipeline.main()

    print("\n2/6 — Vérification des meilleurs checkpoints")
    audit_best_checkpoints.main()

    print("\n3/6 — Boxplots LODO")
    generate_boxplots.main()

    print("\n4/6 — Grilles des huit folds")
    generate_fold_grids.main()

    print("\n5/6 — Diagrammes d'architecture")
    generate_architecture_diagrams.main()

    print("\n6/6 — Documentation des figures")
    figure_documentation.main()

    print("\nTous les rapports ont été générés.")


if __name__ == "__main__":
    main()

