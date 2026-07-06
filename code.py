def train_HBN(model):

    for triplet in HBN:

        x1, x2, x3, label = create_temporal_shuffling_triplet(triplet)

        prediction = model(x1, x2, x3)

        loss = loss_function(prediction, label)

        loss.backward()

        optimizer.step()