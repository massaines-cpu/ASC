class MultiBrainModel(nn.Module):
    def forward(self, eeg_A, eeg_B):
        emb_A = self.encoder(eeg_A)
        emb_B = self.encoder(eeg_B)
        fusion = torch.cat([emb_A, emb_B], dim=1)
        prediction = self.classifier(fusion)
        return prediction