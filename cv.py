"""
Wrapper that applies control variate, variance reduction
to graph convolutional neural networks.

For supervised graphsage,
```bash
python graphsage.py
```

For unsupervised graphsage,
```bash
python cv.py --unsup
```

https://arxiv.org/abs/1710.10568
"""
import numpy as np
import torch
from util.graph_loader import GraphLoader
from util.loop import train, validate

np.random.seed(0)
torch.manual_seed(0)


def cv_forward_wrap(forward, n_historical, n_nodes, embed_dim):
    historicals = torch.full((n_nodes, embed_dim), -1.)

    def cv_forward(x, nodes, limit=9999):
        """
        Parameters
        ----------
        x: tensor[batch_size, feat_size]
            Features for nodes given.
        nodes: tensor[batch_size]
            IDs of nodes.
        """
        if limit == 1:
            historical_mask = (historicals[nodes] != -1).all(-1)
            if historical_mask.sum() > n_historical:
                n_found = int(historical_mask.sum())
                historical_locs = np.where(historical_mask.detach().numpy())[0]
                np.random.randint(0, n_found, size=n_found - n_historical)
                remove_locs = historical_locs[np.random.randint(0, n_found, size=n_found - n_historical)]
                historical_mask[remove_locs] = False

            h_normal = forward(x[~historical_mask], nodes[~historical_mask], limit)
            h_historical = historicals[nodes[historical_mask]].detach()

            h = torch.zeros((len(nodes), embed_dim), dtype=h_normal.dtype)
            h[~historical_mask] = h_normal
            h[historical_mask] = h_historical

            historicals[nodes[~historical_mask]] = h_normal
        else:
            h = forward(x, nodes, limit)

        return h

    return cv_forward


def cv_neighbors_wrap(neighbors, n_historical):
    def cv_neighbors(nodes, n_neighbors):
        neighs = neighbors(nodes, n_neighbors + n_historical)
        return neighs

    return cv_neighbors


def cv_wrapper(model, n_historical, embed_dim):
    """
    Add control variate to receptive field generation
    of given graph conv network model.
    """
    n_nodes = model.dataloader.n_nodes

    model.forward = cv_forward_wrap(model.forward, n_historical, n_nodes, embed_dim)
    model.dataloader.neighbors = cv_neighbors_wrap(model.dataloader.neighbors, n_historical)
    return model


if __name__ == '__main__':
    import sys

    N_EPOCH = 10
    BATCH_SIZE = 512

    LEARNING_RATE = .01
    HIDDEN_SIZE = 256
    N_HISTORICAL = 5

    dataloader = GraphLoader('ppi', BATCH_SIZE)
    model_args = [[dataloader.n_feats, HIDDEN_SIZE, dataloader.n_classes], [25, 5], dataloader]
    if len(sys.argv) > 1 and sys.argv[1] == '--unsup':
        from graphsage_unsup import UnsupGraphSAGE
        model = UnsupGraphSAGE(*model_args)
    else:
        from graphsage import GraphSAGE
        model = GraphSAGE(*model_args)
    model = cv_wrapper(model, N_HISTORICAL, HIDDEN_SIZE)
    model = train(model, dataloader, N_EPOCH, LEARNING_RATE)

    if not (len(sys.argv) > 1 and sys.argv[1] == '--unsup'):
        dataloader = GraphLoader('ppi', BATCH_SIZE, test=True)
        model.dataloader = dataloader
        MAX_STEPS = 5000
        f1 = validate(model, dataloader, MAX_STEPS)
        print(f"F1: {f1 * 100:.1f}")
