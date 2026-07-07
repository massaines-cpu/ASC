def train_ssl(model, dataloader, optimizer, loss_fn):
    ...

torch.save(model.encoder.state_dict(), "encoder.pt")