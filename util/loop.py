from time import time
import numpy as np
import torch.optim as optim
from sklearn.metrics import f1_score


def train(model, dataloader, n_epoch, learning_rate):
    opt = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epoch): 
        time_start = time()

        dataloader.shuffle()
        loss_total = 0
        for ids in dataloader:
            opt.zero_grad()
            feats = dataloader.get_feats(ids)
            embeds = model(feats, ids)

            loss = model.loss(embeds, ids)
            loss.backward()
            opt.step()

            loss_total += loss.item()

        print(f"{epoch}: {time() - time_start:.1f}s | {loss_total:.8f}")

    return model


def validate(model, dataloader, max_steps):
    model.eval()
    dataloader.shuffle()

    n_steps = 0
    scores = []
    for ids in dataloader:
        feats = dataloader.get_feats(ids)
        embeds = model(feats, ids)

        preds = (embeds.detach().numpy() >= .5)
        real = dataloader.get_classes(ids).detach().numpy()

        score = f1_score(preds, real, average="samples", zero_division=0)
        scores.append(score)

        n_steps += 1
        if n_steps > max_steps:
            break

    return np.mean(scores)
