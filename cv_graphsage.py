"""
PyTorch recreation of the GraphSAGE model.

http://snap.stanford.edu/graphsage/
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.aggregator import MeanAggregator
from util.graph_loader import GraphLoader
from util.loop import train, validate

torch.manual_seed(0)


def neighs_with_historical(nodes, n_neighs, dataloader, history):
    """
    Sample neighbors with history.

    Returns
    -------
    ndarray[dtype=int, shape=(nodes, n_neighs)]
    """
    neighs = [dataloader._connections[node] for node in nodes]
    not_history_nodes = torch.where((history == 0).all(-1))[0]
    neighs = [[v for v in neigh if v not in not_history_nodes] for neigh in neighs]
    min_neighs = min([len(neigh) for neigh in neighs])
    neighs_output = torch.empty((nodes.shape[0], min_neighs), dtype=torch.long)
    for i, neigh in enumerate(neighs):
        neighs_output[i] = torch.Tensor(np.random.choice(neigh, size=min_neighs, replace=(len(neigh) != min_neighs)))
    return neighs_output


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

        self.history = torch.full((self.dataloader.n_nodes, layer_sizes[1]), 0.)

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

        nodes, unique_idx, unique_reverse = np.unique(nodes, return_index=True, return_inverse=True)
        h_prev = h_prev[unique_idx]

        for i, layer in enumerate(self.layers[:limit]):
            neighs = self.dataloader.neighbors(nodes, self.n_neighbors[i])
            neigh_feats = self.dataloader.get_feats(neighs)
            if i == 1:
                neigh_feats = self.forward(neigh_feats.reshape(-1, self.dataloader.n_feats), neighs.flatten(), limit=i).reshape(len(nodes), -1, h_prev.shape[-1])
                neigh_original = neigh_feats
            
                N_HISTORICAL = 5
                h_nodes = neighs_with_historical(nodes, N_HISTORICAL, self.dataloader, self.history)
                neigh_historical = self.history[h_nodes].detach()
                
                neigh_delta = neigh_original - self.history[neighs].detach()
                neigh_delta *= neigh_historical.shape[1] / self.n_neighbors[i]
                neigh_feats = torch.cat([neigh_delta, neigh_historical], axis=1)

                self.history[neighs] = neigh_original

            h = self.aggregators[i](h_prev, neigh_feats)
            h = layer(torch.cat([h_prev, h], -1))
            h = self.activations[i](h)
            h_prev = h

        h = h.squeeze()[unique_reverse]
        return h

    def loss(self, z, ids):
        """
        Simple supervised loss.

        Parameters
        ----------
        z: tensor[batch_size, n_class]
            Embeddings for nodes given.
        ids: tensor[batch_size]
            Node ids.
        """
        expected = self.dataloader.get_classes(ids).to(torch.float)
        criterion = nn.BCEWithLogitsLoss()
        return criterion(z, expected).mean()


if __name__ == '__main__':
    N_EPOCH = 10
    BATCH_SIZE = 512

    LEARNING_RATE = .001
    HIDDEN_SIZE = 256

    dataloader = GraphLoader('ppi', BATCH_SIZE)
    model = GraphSAGE([dataloader.n_feats, HIDDEN_SIZE, dataloader.n_classes], [25, 10], dataloader)

    model = train(model, dataloader, N_EPOCH, LEARNING_RATE)

    dataloader = GraphLoader('ppi', BATCH_SIZE, test=True)
    model.dataloader = dataloader
    MAX_STEPS = 5000

    f1 = validate(model, dataloader, MAX_STEPS)

    print(f"F1: {f1 * 100:.1f}")
