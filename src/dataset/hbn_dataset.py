


class HBNDataset(Dataset):
    def __getitem__(self, idx):
        eeg_epoch = ...
        return eeg_epoch

class HBNDataset(Dataset):

    def __getitem__(self, idx):

        eeg = load_eeg(...)

        return eeg

class TemporalShufflingDataset(Dataset):

    def __getitem__(self, idx):

        ...

        if ordre_correct:
            label = 1
        else:
            label = 0

        return x1, x2, x3, label