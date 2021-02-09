"""
PyTorch recreation of the GraphSAGE model.

http://snap.stanford.edu/graphsage/
"""
import os
import json
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

torch.manual_seed(0)


class MeanAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_self, x_neigh):
        if len(x_self.shape) == 2:
            x_self = x_self.view(x_self.shape[0], -1, x_self.shape[1])

        return torch.cat([x_self, x_neigh], axis=1).mean(1).view(x_self.shape[0], 1, x_self.shape[2])


class GraphLoader(Dataset):
    """
    Graph dataset loader with uniform neighbor sampling.

    Works with preprocessed graphs in format
    folder/
    - *id_map.json: {"id": feats_idx} (ideally {"0": 0, ...})
    - *G.json: {
        "nodes": [{"id": 0, "test": false}, ...],
        "links": [{"source": 0, "target": 372}, ...]
        }
    - *feats.npy: ndarray[n_nodes, n_feats]
    - *class_map.json: {"50088": [0, 1], "44884": [1, 1]}
    eg PPO or reddit preprocessed dataset from,
    http://snap.stanford.edu/graphsage/
    """
    def __init__(self, folder, batch_size, max_degree=9999, test=False):
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.test = test

        for filename in os.listdir(folder):
            filename_full = os.path.join(folder, filename)
            if filename.endswith('feats.npy'):
                self._feats = torch.Tensor(np.load(filename_full))
                self.n_nodes = self._feats.shape[0]
            else:
                file = open(filename_full)
                if filename.endswith('G.json'):
                    data = json.load(file)
                    types = data['nodes']
                    edges = data['links']
                elif filename.endswith('id_map.json'):
                    id_map = json.load(file)
                elif filename.endswith('class_map.json'):
                    classes = json.load(file)
                file.close()

        self._types = torch.empty(self.n_nodes, dtype=torch.bool)
        for values in types:
            node_id, is_test = str(values['id']), values['test']
            self._types[id_map[node_id]] = is_test
        del types
        self._connections = [[i] for i in range(self.n_nodes)]
        for edge in edges:
            self._connections[edge['source']].append(edge['target'])
        self._connections = [torch.Tensor(row) for row in self._connections]
        del edges
        self._classes = torch.empty((self.n_nodes, len(classes[node_id])), dtype=torch.float)
        for node_id, idx in id_map.items():
            self._classes[idx] = torch.Tensor(classes[node_id])
        del classes, id_map

        self.n_feats = self._feats.shape[-1]
        self.n_classes = self._classes.shape[-1]
        self._selected_ids = torch.arange(self.n_nodes, dtype=torch.long)[self.test == self._types]

    def __iter__(self):
        """
        Iterate through dataset.
        """
        for _ in range(self.n_nodes // self.batch_size):
            yield self.sample()

    def get_feats(self, ids):
        """
        Get features for node ids given.
        """
        return self._feats[ids]

    def get_classes(self, ids):
        """
        Get features for node ids given.
        """
        return self._classes[ids]

    def shuffle(self):
        """
        Shuffle dataset, also serves as a reset function.
        """

    def sample(self, n_samples=None):
        """
        Sample nodes names from graph.
        """
        n_samples = n_samples or self.batch_size
        return self._selected_ids[torch.randint(self._selected_ids.shape[0], (n_samples,))]

    def neighbors(self, nodes, n_neighbors):
        """
        Sample n_neighbors neighbor names of nodes given.
        """
        neighs = torch.empty((nodes.shape[0], n_neighbors), dtype=torch.long)
        for i, node in enumerate(nodes):
            neigh = self._connections[node]
            neighs[i] = neigh[torch.randint(neigh.shape[-1], (n_neighbors,))]
        return neighs

    def random_walk(self, nodes, n_samples, walk_len):
        """
        Fixed length random walk starting at given nodes.
        """
        assert n_samples == 1, "Random walk not implemented for n_samples>1!"
        neighs = nodes
        for _ in range(walk_len):
            neighs = self.neighbors(neighs, 1)
        return neighs


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
        if self.concat:
            layer_sizes = [s * 2 for s in layer_sizes]#[layer_sizes[0]] + [s * 2 for s in layer_sizes[1:]]
        self.layers = nn.ModuleList([nn.Linear(size, layer_sizes[i+1]) for i, size in enumerate(layer_sizes[:-1])])
        self.aggregators = nn.ModuleList([MeanAggregator() for _ in range(len(layer_sizes)-1)])
        self.activations = [nn.ReLU(), nn.ReLU()]

    def forward(self, x, nodes):
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

        for i, layer in enumerate(self.layers):
            neighs = self.dataloader.neighbors(nodes, self.n_neighbors[i])
            neigh_feats = self.dataloader.get_feats(neighs)
            h = self.aggregators[i](h_prev, neigh_feats)
            h = layer(torch.cat([h_prev, h], -1))
            h = self.activations[i](h)
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
        walk_len = 3  # TODO
        num_pos = 1  # TODO - Assumed since no mult on pos_score
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
    N_EPOCH = 10
    BATCH_SIZE = 512
    MAX_STEPS = 1e10

    LEARNING_RATE = .01
    EMBED_DIM = 128
    HIDDEN_SIZE = 128
    MAX_DEGREE = 128

    dataloader = GraphLoader('ppi', BATCH_SIZE, max_degree=MAX_DEGREE)

    model = GraphSAGE([dataloader.n_feats, EMBED_DIM], [25, 10], dataloader)
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

        print(f"{epoch}: {time() - time_start:.1f}s | {loss_total:.2f}")
        if n_steps > MAX_STEPS:
            break

    # TODO validate
    model.eval()
    N_STEP = 5000
    BATCH_SIZE = 256
