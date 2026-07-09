def train_ssl(model, dataloader, optimizer, loss_fn):
    ...

torch.save(model.encoder.state_dict(), "encoder.pt")

dataset = HBNDataset()

dataset = TemporalShufflingDataset(dataset)

loader = DataLoader(dataset)

encoder = ShallowConvNet()

for epoch in range(nb_epochs):

    for x1, x2, x3, label in loader:

        prediction = encoder(x1, x2, x3)

        loss = criterion(prediction, label)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

torch.save(
    encoder.state_dict(),
    "encoder.pt"
)