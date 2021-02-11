"""
PyTorch recreation of the GraphSAGE model.

http://snap.stanford.edu/graphsage/
"""
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util.aggregator import MeanAggregator
from util.graph_loader import GraphLoader
from sklearn.metrics import f1_score

torch.manual_seed(0)


class GraphSAGE(nn.Module):
    """
    Supervised GraphSAGE.

    http://snap.stanford.edu/graphsage/
    """
    def __init__(self, layer_sizes, n_neighbors, dataloader, concat=True):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.dataloader = dataloader
        self.concat = concat
        self.layers = nn.ModuleList([nn.Linear(size * ([1, 2][self.concat]), layer_sizes[i+1]) for i, size in enumerate(layer_sizes[:-1])])
        self.aggregators = nn.ModuleList([MeanAggregator() for _ in range(len(layer_sizes)-1)])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(len(self.layers))])

    def forward(self, x, nodes, limit=9999):
        """
        Parameters
        ----------
        x: tensor[batch_size, feat_size]
            Features for nodes given.
        nodes: tensor[batch_size]
            IDs of nodes.
        """
        h_prev = x.detach().clone()
        if len(h_prev.shape) == 2:
            h_prev = h_prev.view(h_prev.shape[0], -1, h_prev.shape[1])

        # TODO only need to remove uniques here!

        for i, layer in enumerate(self.layers[:limit]):
            neighs = self.dataloader.neighbors(nodes, self.n_neighbors[i])
            neigh_feats = self.dataloader.get_feats(neighs)
            if i == 1:
                neigh_feats = self.forward(neigh_feats.reshape(-1, self.dataloader.n_feats), neighs.flatten(), limit=1).reshape(len(nodes), -1, h_prev.shape[-1])
            h = self.aggregators[i](h_prev, neigh_feats)
            h = layer(torch.cat([h_prev, h], -1))
            h = self.activations[i](h)
            h_prev = h

        return h.squeeze()

    def loss(self, z, expected):
        """
        Simple supervised loss.

        Parameters
        ----------
        z: tensor[batch_size, n_class]
            Embeddings for nodes given.
        expected: tensor[batch_size, n_class]
            Class vector for same nodes as z.
        """
        criterion = nn.BCEWithLogitsLoss()
        return criterion(z, expected.to(torch.float)).mean()


if __name__ == '__main__':
    N_EPOCH = 5
    BATCH_SIZE = 512

    LEARNING_RATE = .1
    HIDDEN_SIZE = 128
    MAX_DEGREE = 128

    dataloader = GraphLoader('ppi', BATCH_SIZE, max_degree=MAX_DEGREE)

    model = GraphSAGE([dataloader.n_feats, HIDDEN_SIZE, dataloader.n_classes], [25, 10], dataloader)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    n_steps = 0
    for epoch in range(N_EPOCH): 
        time_start = time()

        dataloader.shuffle()
        loss_total = 0
        for ids in dataloader:
            opt.zero_grad()
            feats = dataloader.get_feats(ids)
            embeds = model(feats, ids)

            loss = model.loss(embeds, dataloader.get_classes(ids))
            loss.backward()
            opt.step()

            loss_total += loss.item()
            n_steps += 1

        print(f"{epoch}: {time() - time_start:.1f}s | {loss_total:.8f}")

    dataloader = GraphLoader('ppi', BATCH_SIZE, max_degree=MAX_DEGREE, test=True)
    dataloader.shuffle()
    model.eval()
    model.dataloader = dataloader

    MAX_STEPS = 5000
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
        if n_steps > MAX_STEPS:
            break

    print(f"F1: {np.mean(scores) * 100:.1f}")
