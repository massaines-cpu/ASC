import torch
import torch.nn as nn

class ShallowConvNet(nn.Module):
    def __init__(self, n_channels=61, n_times=501, embedding_dim=100):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=25),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)


if __name__ == "__main__":
    x = torch.randn(16, 61, 501)
    model = ShallowConvNet()
    z = model(x)
    print(z.shape)