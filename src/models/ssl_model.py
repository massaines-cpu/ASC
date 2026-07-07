class SSLTemporalShufflingModel(nn.Module):
    def forward(self, x1, x2, x3):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z3 = self.encoder(x3)
        prediction = ...
        return prediction