import torch
from torch import nn
from pathlib import Path
import pandas as pd

from src.dataset.labels import prepare_classification_table
from src.dataset.dataloader_participant import create_participant_dataloaders

PROJECT_ROOT = Path(__file__).resolve().parents[2]

metadata = pd.read_csv(
    PROJECT_ROOT / "data" / "all_metadata.csv"
)

classification_table = prepare_classification_table(
    metadata,
    target_column="eyes_code",
    allowed_classes=["YO", "YF"],
    label_map={
        "YO": 0,
        "YF": 1,
    },
)

train_loader, validation_loader, test_loader = create_participant_dataloaders(
    classification_table=classification_table,
    dataset_root=PROJECT_ROOT / "data" / "data_toy",
    train_dyads=["J2", "J4", "J5", "J7", "J8", "J1"],
    validation_dyads=["J10"],
    test_dyads=["J15"],
    batch_size=5,
)
class SimpleParticipantClassifier(nn.Module):

    def __init__(self, number_of_channels=32, number_of_timepoints=5120):
        super().__init__()

        self.flatten = nn.Flatten()
        # calcul 2 nombre car 2 classes, YO YF
        self.classifier = nn.Linear(
            number_of_channels * number_of_timepoints,
            2,
        )

    def forward(self, eeg):
        eeg = self.flatten(eeg)
        predictions = self.classifier(eeg)

        return predictions

eeg, labels = next(iter(train_loader))
model = SimpleParticipantClassifier()

predictions = model(eeg)

print("EEG :", eeg.shape)
print("Prédictions :", predictions.shape)
print("Labels :", labels.shape)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)