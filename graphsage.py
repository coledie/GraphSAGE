"""
PyTorch recreation of the GraphSAGE model.

http://snap.stanford.edu/graphsage/
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

torch.manual_seed(0)


class MeanAggregator(nn.Module):
    def __init__(self, concat=False):
        super().__init__()
        self.concat = concat

    def forward(self, x_self, x_neigh):
        x_neigh = x_neigh.mean(-2)

        if not self.concat:
            output = x_self + x_neigh
        else:
            output = torch.cat([x_self, x_neigh], axis=1)

        return output


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
        self._connections = torch.zeros((self.n_nodes, self.n_nodes), dtype=torch.bool)
        for edge in edges:
            self._connections[edge['source'], edge['target']] = 1
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

    def sample(self):
        """
        Sample nodes names from graph.
        """
        return self._selected_ids[torch.randint(self._selected_ids.shape[0], (self.batch_size,))]
    
    def neighbors(self, nodes, n_neighbors):
        """
        Sample n_neighbors neighbor names of nodes given.
        """
        neighs = torch.empty((nodes.shape[0], n_neighbors), dtype=torch.int)
        for i, node in enumerate(nodes):
            neigh_mask = self._connections[node][self.test == self._types]
            neigh = torch.cat([torch.Tensor([node]), self._selected_ids[neigh_mask]], axis=-1)
            neighs[i] = neigh[torch.randint(neigh.shape[-1], (n_neighbors,))]
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
            layer_sizes = [layer_sizes[0]] + [s * 2 for s in layer_sizes[1:]]
        self.layers = nn.ModuleList([nn.Linear(size, layer_sizes[i+1]) for i, size in enumerate(layer_sizes[:-1])])
        self.aggregators = nn.ModuleList([MeanAggregator(self.concat) for _ in range(len(layer_sizes)-1)])

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
        for i, layer in enumerate(self.layers):
            neighs = self.dataloader.neighbors(nodes, self.n_neighbors[i])
            neigh_feats = self.dataloader.get_feats(neighs)
            h = self.aggregators[i](h_prev, neigh_feats)
            h = layer(torch.cat(h_prev, h))
            h = torch.max(h, 0)
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
        return


if __name__ == '__main__':
    N_EPOCH = 10
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

        print(epoch, loss_total)
        if n_steps > MAx_STEPS:
            break

    # TODO validate
    model.eval()
    N_STEP = 5000
    BATCH_SIZE = 256
