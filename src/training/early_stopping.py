class EarlyStopping:
    """Suit la validation loss et détecte l'arrêt anticipé."""

    def __init__(
        self,
        patience: int,
        min_delta: float,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0

    def update(self, validation_loss: float) -> bool:
        """Retourne True lorsque la loss s'améliore."""

        improved = (
            validation_loss
            < self.best_loss - self.min_delta
        )

        if improved:
            self.best_loss = validation_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return improved

    def should_stop(self) -> bool:
        """Indique si la patience est épuisée."""

        return (
            self.epochs_without_improvement
            >= self.patience
        )