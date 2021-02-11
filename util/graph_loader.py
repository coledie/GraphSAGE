import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


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
    def __init__(self, folder, batch_size, test=False):
        self.batch_size = batch_size
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
        self._connections = [torch.Tensor(row).to(torch.long) for row in self._connections]
        del edges
        self._classes = torch.empty((self.n_nodes, len(classes[node_id])), dtype=torch.long)
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
