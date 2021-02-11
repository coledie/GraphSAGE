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
from graphsage import GraphSAGE

torch.manual_seed(0)


class UnsupGraphSAGE(GraphSAGE):
    """
    Unsupervised GraphSAGE.

    http://snap.stanford.edu/graphsage/
    """
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

    model = UnsupGraphSAGE([dataloader.n_feats, HIDDEN_SIZE, EMBED_DIM], [25, 10], dataloader)
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
