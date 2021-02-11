"""
PyTorch recreation of the GraphSAGE model.

http://snap.stanford.edu/graphsage/
"""
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from util.aggregator import MeanAggregator
from util.graph_loader import GraphLoader

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

        return h

    def loss(self, z, nodes):
        """
        Encourage nearby nodes to have similar representations while
        enforcing that disparate nodes are highly distinct.

        loss = -log(sigma(z_u.T *z_v))-Q E_vn~P_n(v)(log(sigma(-z_u.T*z_vn)))
        z = embedding of node
        v = node that coocurs near u on fixed length randon walk.
        P_n is negative sampling dist, Q = num negative sampels.

        Parameters
        ----------
        z: tensor[batch_size, embed_dim]
            Embeddings for nodes given.
        nodes: tensor[batch_size]
            IDs of nodes.
        """
        walk_len = 3
        num_pos = 1
        num_negs = 20

        pos_nodes = self.dataloader.random_walk(nodes, num_pos, walk_len)
        pos_feats = self.dataloader.get_feats(pos_nodes)
        pos_embeds = self.forward(pos_feats, pos_nodes)
        neg_nodes = self.dataloader.sample(num_negs * len(nodes))
        neg_feats = self.dataloader.get_feats(neg_nodes)
        neg_embeds = self.forward(neg_feats, neg_nodes).view(len(nodes), num_negs, -1)

        pos_score = -torch.log(torch.sigmoid(torch.sum(z * pos_embeds, dim=-1))).squeeze()
        neg_score = -torch.log(torch.sigmoid(torch.sum(z * neg_embeds, dim=-1))).sum(-1)

        return (pos_score + num_negs * neg_score).mean()


if __name__ == '__main__':
    N_EPOCH = 5
    BATCH_SIZE = 512
    MAX_STEPS = 1e10

    LEARNING_RATE = .01
    EMBED_DIM = 128
    HIDDEN_SIZE = 128
    MAX_DEGREE = 128

    dataloader = GraphLoader('ppi', BATCH_SIZE, max_degree=MAX_DEGREE)

    model = GraphSAGE([dataloader.n_feats, HIDDEN_SIZE, EMBED_DIM], [25, 10], dataloader)
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

            loss = model.loss(embeds, ids)
            loss.backward()
            opt.step()

            loss_total += loss.item()
            n_steps += 1

        print(f"{epoch}: {time() - time_start:.1f}s | {loss_total:.8f}")
        if n_steps > MAX_STEPS:
            break

    # TODO validate
    model.eval()
    N_STEP = 5000
    BATCH_SIZE = 256
