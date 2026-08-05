import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import hf_hub_download


history_path = hf_hub_download(
    repo_id="guido151/EEGNetv4",
    filename="EEGNetv4_Lee2019_ERP/history.json",
)


with Path(history_path).open(
    mode="r",
    encoding="utf-8",
) as history_file:
    history_data = json.load(history_file)


history_table = pd.DataFrame(history_data)

print("Colonnes disponibles :")
print(history_table.columns.tolist())

print("\nPremières lignes :")
print(history_table.head())

plt.figure(figsize=(10, 5))

plt.plot(
    history_table.index + 1,
    history_table["train_loss"],
    label="Train loss",
)

plt.plot(
    history_table.index + 1,
    history_table["valid_loss"],
    label="Validation loss",
)

best_epoch_index = history_table["valid_loss"].idxmin()

plt.axvline(
    best_epoch_index + 1,
    color="black",
    linestyle="--",
    label="Meilleur checkpoint",
)

plt.xlabel("Époque")
plt.ylabel("Loss")
plt.title("Entraînement EEGNetv4 Lee2019 ERP")
plt.legend()
plt.tight_layout()
plt.show()